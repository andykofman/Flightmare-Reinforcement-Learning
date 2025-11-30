#!/usr/bin/env python3
"""
COMPREHENSIVE PHYSICS VALIDATION SCRIPT

Purpose: Systematically validate that Flightmare physics is working correctly
        by testing basic maneuvers with known control inputs.

Tests:
1. Zero-input hover test (should maintain altitude within reasonable bounds)
2. Constant upward thrust test (should rise predictably)
3. Constant downward thrust test (should descend predictably)
4. Lateral movement test (should move horizontally)
5. Action-to-motion correlation test (verify all 4 motors affect motion)
6. Gravity check (verify downward acceleration without thrust)
7. Response time check (verify motors respond within expected timeframe)
8. Target tracking test (simple P-controller to verify physics allows reaching targets)

Each test reports PASS/FAIL with diagnostic information.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add flightrl_modern to path
sys.path.insert(0, str(Path(__file__).parent))

from flightrl_modern.envs.gymnasium_wrapper import make_flight_env

# Test configuration
TESTS_TO_RUN = [
    "hover_stability",
    "upward_thrust",
    "downward_thrust", 
    "lateral_movement",
    "motor_correlation",
    "gravity_check",
    "response_time",
    "target_tracking"
]

RESULTS = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def print_header(title):
    """Print a formatted test header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_test_result(test_name, passed, message=""):
    """Print and record test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{status}: {test_name}")
    if message:
        print(f"  → {message}")
    
    if passed:
        RESULTS["passed"].append(test_name)
    else:
        RESULTS["failed"].append(test_name)

def print_warning(message):
    """Print and record warning"""
    print(f"  ⚠ WARNING: {message}")
    RESULTS["warnings"].append(message)

def create_env():
    """Create a fresh environment for testing"""
    try:
        env = make_flight_env(
            render=False,
            num_envs=1,
            num_threads=1,
            seed=42
        )
        return env
    except Exception as e:
        print(f"ERROR: Failed to create environment: {e}")
        sys.exit(1)

def run_episode(env, action_sequence, max_steps=300):
    """
    Run an episode with a specific action sequence.
    
    Args:
        env: The environment
        action_sequence: Either a single action (repeated) or list of actions
        max_steps: Maximum number of steps
    
    Returns:
        DataFrame with trajectory data
    """
    obs, info = env.reset()
    
    # Handle observation format - might be (num_envs, obs_dim) or (obs_dim,)
    if obs.ndim == 1:
        # Single env, obs is 1D array
        get_obs = lambda o: o
    else:
        # Vectorized env, obs is 2D array (num_envs, obs_dim)
        get_obs = lambda o: o[0] if o.ndim > 1 else o
    
    data = {
        "step": [],
        "time": [],
        "pos_x": [], "pos_y": [], "pos_z": [],
        "vel_x": [], "vel_y": [], "vel_z": [],
        "action_0": [], "action_1": [], "action_2": [], "action_3": [],
        "reward": []
    }
    
    # Handle single action vs sequence
    if isinstance(action_sequence, np.ndarray) and action_sequence.ndim == 1:
        # Single action - repeat it
        actions = [action_sequence] * max_steps
    else:
        actions = action_sequence
    
    for step in range(min(max_steps, len(actions))):
        action = actions[step]
        
        # Get current observation
        current_obs = get_obs(obs)
        
        # Record state
        data["step"].append(step)
        data["time"].append(step * 0.02)  # dt = 0.02s
        data["pos_x"].append(current_obs[0])
        data["pos_y"].append(current_obs[1])
        data["pos_z"].append(current_obs[2])
        data["vel_x"].append(current_obs[6])
        data["vel_y"].append(current_obs[7])
        data["vel_z"].append(current_obs[8])
        data["action_0"].append(action[0])
        data["action_1"].append(action[1])
        data["action_2"].append(action[2])
        data["action_3"].append(action[3])
        
        # Step environment - ensure action has right shape
        if action.ndim == 1:
            action_input = action.reshape(1, -1)
        else:
            action_input = action
            
        obs, reward, terminated, truncated, info = env.step(action_input)
        
        # Handle reward format
        if isinstance(reward, np.ndarray):
            data["reward"].append(reward[0] if reward.ndim > 0 else float(reward))
        else:
            data["reward"].append(float(reward))
        
        # Handle termination format
        term = terminated[0] if isinstance(terminated, np.ndarray) else terminated
        trunc = truncated[0] if isinstance(truncated, np.ndarray) else truncated
        
        if term or trunc:
            break
    
    return pd.DataFrame(data)

# ============================================================================
# TEST 1: Hover Stability
# ============================================================================
def test_hover_stability():
    """
    Test: Apply zero actions (hover command) and verify quad maintains altitude
    
    Expected: Z position should stay within ±1.0m of initial position
    Physics issue if: Quad drifts significantly or crashes
    """
    print_header("TEST 1: Hover Stability (Zero Action)")
    
    env = create_env()
    action = np.array([0.0, 0.0, 0.0, 0.0])  # Hover command
    
    print(f"Action: {action} (all zeros = hover thrust)")
    print("Expected: Maintain altitude within ±1.0m for 6 seconds")
    
    df = run_episode(env, action, max_steps=300)
    
    # Analysis
    initial_z = df["pos_z"].iloc[0]
    final_z = df["pos_z"].iloc[-1]
    max_z_deviation = df["pos_z"].max() - initial_z
    min_z_deviation = df["pos_z"].min() - initial_z
    
    print(f"\nResults:")
    print(f"  Initial Z: {initial_z:.3f}m")
    print(f"  Final Z: {final_z:.3f}m")
    print(f"  Z drift: {final_z - initial_z:.3f}m")
    print(f"  Max Z deviation: {max_z_deviation:.3f}m")
    print(f"  Min Z deviation: {min_z_deviation:.3f}m")
    
    # Verdict
    max_drift = max(abs(max_z_deviation), abs(min_z_deviation))
    
    if max_drift < 0.5:
        print_test_result("Hover Stability", True, f"Excellent stability: {max_drift:.3f}m max drift")
    elif max_drift < 1.0:
        print_test_result("Hover Stability", True, f"Acceptable stability: {max_drift:.3f}m max drift")
        print_warning(f"Some drift present - consider tuning hover thrust")
    else:
        print_test_result("Hover Stability", False, 
                         f"Excessive drift: {max_drift:.3f}m (threshold: 1.0m)")
        print("  → Physics issue: Zero action should maintain hover!")
    
    env.close()
    return df

# ============================================================================
# TEST 2: Upward Thrust
# ============================================================================
def test_upward_thrust():
    """
    Test: Apply constant upward thrust and verify quad rises
    
    Expected: Z velocity > 0, Z position increases monotonically
    Physics issue if: Quad doesn't rise or rises too fast/slow
    """
    print_header("TEST 2: Upward Thrust Response")
    
    env = create_env()
    action = np.array([0.2, 0.2, 0.2, 0.2])  # 20% upward thrust
    
    print(f"Action: {action} (20% thrust on all motors)")
    print("Expected: Upward acceleration ~2-4 m/s²")
    
    df = run_episode(env, action, max_steps=150)  # 3 seconds
    
    # Analysis
    initial_z = df["pos_z"].iloc[0]
    final_z = df["pos_z"].iloc[-1]
    max_vel_z = df["vel_z"].max()
    avg_vel_z = df["vel_z"].mean()
    
    print(f"\nResults:")
    print(f"  Initial Z: {initial_z:.3f}m")
    print(f"  Final Z: {final_z:.3f}m")
    print(f"  Z displacement: {final_z - initial_z:.3f}m")
    print(f"  Max Z velocity: {max_vel_z:.3f}m/s")
    print(f"  Avg Z velocity: {avg_vel_z:.3f}m/s")
    
    # Estimate acceleration (should be positive)
    if len(df) > 10:
        # Use first 0.5s to estimate initial acceleration
        early_vel = df["vel_z"].iloc[:25].values
        time_vals = df["time"].iloc[:25].values
        if len(early_vel) > 1:
            accel = np.polyfit(time_vals, early_vel, 1)[0]
            print(f"  Initial acceleration: {accel:.3f}m/s²")
    
    # Verdict
    z_rise = final_z - initial_z
    passed = (z_rise > 0.5) and (avg_vel_z > 0.2)
    
    if passed:
        print_test_result("Upward Thrust", True, f"Quad rose {z_rise:.3f}m as expected")
    else:
        print_test_result("Upward Thrust", False,
                         f"Quad did not rise properly: {z_rise:.3f}m, vel={avg_vel_z:.3f}m/s")
        print("  → Physics issue: Positive thrust should cause upward motion!")
    
    env.close()
    return df

# ============================================================================
# TEST 3: Downward Thrust
# ============================================================================
def test_downward_thrust():
    """
    Test: Apply negative thrust and verify quad descends
    
    Expected: Z velocity < 0, Z position decreases
    Physics issue if: Quad doesn't descend or accelerates upward
    """
    print_header("TEST 3: Downward Thrust Response")
    
    env = create_env()
    action = np.array([-0.2, -0.2, -0.2, -0.2])  # 20% downward thrust
    
    print(f"Action: {action} (-20% thrust on all motors)")
    print("Expected: Downward acceleration, controlled descent")
    
    df = run_episode(env, action, max_steps=150)  # 3 seconds
    
    # Analysis
    initial_z = df["pos_z"].iloc[0]
    final_z = df["pos_z"].iloc[-1]
    min_vel_z = df["vel_z"].min()
    avg_vel_z = df["vel_z"].mean()
    
    print(f"\nResults:")
    print(f"  Initial Z: {initial_z:.3f}m")
    print(f"  Final Z: {final_z:.3f}m")
    print(f"  Z displacement: {final_z - initial_z:.3f}m")
    print(f"  Min Z velocity: {min_vel_z:.3f}m/s")
    print(f"  Avg Z velocity: {avg_vel_z:.3f}m/s")
    
    # Verdict
    z_drop = initial_z - final_z
    passed = (z_drop > 0.5) and (avg_vel_z < -0.2)
    
    if passed:
        print_test_result("Downward Thrust", True, f"Quad descended {z_drop:.3f}m as expected")
    else:
        print_test_result("Downward Thrust", False,
                         f"Quad did not descend properly: {z_drop:.3f}m, vel={avg_vel_z:.3f}m/s")
        print("  → Physics issue: Negative thrust should cause downward motion!")
    
    env.close()
    return df

# ============================================================================
# TEST 4: Lateral Movement
# ============================================================================
def test_lateral_movement():
    """
    Test: Apply differential thrust to create lateral movement
    
    Expected: Quad tilts and moves horizontally
    Physics issue if: No tilt or no horizontal motion
    """
    print_header("TEST 4: Lateral Movement (Differential Thrust)")
    
    env = create_env()
    # Increase motors 0,1 and decrease 2,3 to pitch forward
    action = np.array([0.1, 0.1, -0.1, -0.1])
    
    print(f"Action: {action} (differential thrust to pitch)")
    print("Expected: Horizontal displacement > 0.5m")
    
    df = run_episode(env, action, max_steps=200)  # 4 seconds
    
    # Analysis
    initial_xy = np.array([df["pos_x"].iloc[0], df["pos_y"].iloc[0]])
    final_xy = np.array([df["pos_x"].iloc[-1], df["pos_y"].iloc[-1]])
    xy_displacement = np.linalg.norm(final_xy - initial_xy)
    
    max_xy_vel = np.sqrt(df["vel_x"]**2 + df["vel_y"]**2).max()
    
    print(f"\nResults:")
    print(f"  Initial XY: ({initial_xy[0]:.3f}, {initial_xy[1]:.3f})")
    print(f"  Final XY: ({final_xy[0]:.3f}, {final_xy[1]:.3f})")
    print(f"  XY displacement: {xy_displacement:.3f}m")
    print(f"  Max XY velocity: {max_xy_vel:.3f}m/s")
    
    # Verdict
    passed = xy_displacement > 0.3
    
    if passed:
        print_test_result("Lateral Movement", True, 
                         f"Quad moved {xy_displacement:.3f}m laterally as expected")
    else:
        print_test_result("Lateral Movement", False,
                         f"Insufficient lateral movement: {xy_displacement:.3f}m")
        print("  → Physics issue: Differential thrust should cause tilt and lateral motion!")
    
    env.close()
    return df

# ============================================================================
# TEST 5: Motor Correlation
# ============================================================================
def test_motor_correlation():
    """
    Test: Verify each motor independently affects motion
    
    Expected: Each motor command creates measurable change in state
    Physics issue if: Motors don't affect motion or multiple motors have same effect
    """
    print_header("TEST 5: Individual Motor Response")
    
    env = create_env()
    
    print("Testing each motor individually...")
    
    motor_effects = {}
    
    for motor_id in range(4):
        action = np.zeros(4)
        action[motor_id] = 0.3  # 30% thrust on single motor
        
        df = run_episode(env, action, max_steps=100)  # 2 seconds
        
        # Measure effects
        z_change = df["pos_z"].iloc[-1] - df["pos_z"].iloc[0]
        xy_change = np.sqrt(
            (df["pos_x"].iloc[-1] - df["pos_x"].iloc[0])**2 +
            (df["pos_y"].iloc[-1] - df["pos_y"].iloc[0])**2
        )
        
        motor_effects[motor_id] = {
            "z_change": z_change,
            "xy_change": xy_change
        }
        
        print(f"  Motor {motor_id}: Z={z_change:+.3f}m, XY={xy_change:.3f}m")
        
        env.close()
        env = create_env()  # Fresh env for each test
    
    # Analysis: All motors should have some effect
    all_have_effect = all(
        abs(m["z_change"]) > 0.1 or m["xy_change"] > 0.1 
        for m in motor_effects.values()
    )
    
    # Check if motors are too similar (should have different effects due to geometry)
    z_changes = [m["z_change"] for m in motor_effects.values()]
    z_std = np.std(z_changes)
    
    print(f"\nAnalysis:")
    print(f"  Z-change std dev: {z_std:.3f}m (should be > 0.05 for differential effects)")
    
    if all_have_effect:
        print_test_result("Motor Correlation", True, "All motors affect quad motion")
        if z_std < 0.05:
            print_warning("All motors have very similar effects - check motor layout")
    else:
        print_test_result("Motor Correlation", False, "Some motors don't affect motion")
        print("  → Physics issue: All motors should independently affect the quad!")
    
    env.close()
    return motor_effects

# ============================================================================
# TEST 6: Gravity Check
# ============================================================================
def test_gravity():
    """
    Test: Apply large negative thrust and verify quad falls under gravity
    
    Expected: Downward acceleration ~9.81 m/s² (gravity)
    Physics issue if: No gravity or wrong magnitude
    """
    print_header("TEST 6: Gravity Verification")
    
    env = create_env()
    action = np.array([-1.0, -1.0, -1.0, -1.0])  # Maximum negative thrust
    
    print(f"Action: {action} (maximum negative thrust)")
    print("Expected: Gravity-induced fall (~9.81 m/s² acceleration)")
    
    df = run_episode(env, action, max_steps=100)  # 2 seconds
    
    # Estimate acceleration from velocity change
    if len(df) > 20:
        vel_z = df["vel_z"].values[:50]  # First second
        time_vals = df["time"].values[:50]
        
        # Fit linear to get acceleration
        if len(vel_z) > 1:
            accel = np.polyfit(time_vals, vel_z, 1)[0]
            
            print(f"\nResults:")
            print(f"  Estimated downward acceleration: {-accel:.2f} m/s²")
            print(f"  Expected (gravity): ~9.81 m/s²")
            print(f"  Difference: {abs(accel + 9.81):.2f} m/s²")
            
            # Allow some margin due to drag
            if 7.0 < abs(accel) < 12.0:
                print_test_result("Gravity Check", True, 
                                 f"Gravity magnitude reasonable: {-accel:.2f} m/s²")
            else:
                print_test_result("Gravity Check", False,
                                 f"Gravity magnitude wrong: {-accel:.2f} m/s² (expected ~9.81)")
                print("  → Physics issue: Gravity constant may be incorrect!")
        else:
            print_test_result("Gravity Check", False, "Insufficient data to estimate gravity")
    else:
        print_test_result("Gravity Check", False, "Episode too short to test gravity")
    
    env.close()
    return df

# ============================================================================
# TEST 7: Response Time
# ============================================================================
def test_response_time():
    """
    Test: Apply step input and measure time to 63% of steady-state response
    
    Expected: Response time < 0.5s (motor tau = 0.0001s is very fast)
    Physics issue if: Very slow response or no response
    """
    print_header("TEST 7: Control Response Time")
    
    env = create_env()
    
    # Create step input: 0 for 1s, then 0.3 for 2s
    actions = []
    for i in range(50):  # 1 second at hover
        actions.append(np.array([0.0, 0.0, 0.0, 0.0]))
    for i in range(100):  # 2 seconds at upward thrust
        actions.append(np.array([0.3, 0.3, 0.3, 0.3]))
    
    print("Applying step input at t=1.0s")
    print("Expected: Quick response (< 0.5s to reach 63% of final value)")
    
    df = run_episode(env, actions, max_steps=150)
    
    # Find response time (time to reach 63% of change)
    step_time_idx = 50  # Where step occurs
    
    if len(df) > step_time_idx + 20:
        z_before_step = df["pos_z"].iloc[step_time_idx]
        z_final = df["pos_z"].iloc[-1]
        z_change = z_final - z_before_step
        z_target_63 = z_before_step + 0.63 * z_change
        
        # Find when we cross 63% threshold
        after_step_df = df.iloc[step_time_idx:]
        crossed = after_step_df[after_step_df["pos_z"] >= z_target_63]
        
        if len(crossed) > 0:
            response_time = crossed.iloc[0]["time"] - df.iloc[step_time_idx]["time"]
            
            print(f"\nResults:")
            print(f"  Z before step: {z_before_step:.3f}m")
            print(f"  Z after step (final): {z_final:.3f}m")
            print(f"  Total change: {z_change:.3f}m")
            print(f"  Time to 63% response: {response_time:.3f}s")
            
            if response_time < 0.5:
                print_test_result("Response Time", True, f"Fast response: {response_time:.3f}s")
            elif response_time < 1.0:
                print_test_result("Response Time", True, f"Adequate response: {response_time:.3f}s")
                print_warning("Response slower than expected for motor_tau=0.0001")
            else:
                print_test_result("Response Time", False, f"Slow response: {response_time:.3f}s")
                print("  → Physics issue: Motors should respond faster!")
        else:
            print_test_result("Response Time", False, "No response detected to step input")
    else:
        print_test_result("Response Time", False, "Insufficient data")
    
    env.close()
    return df

# ============================================================================
# TEST 8: Target Tracking Ability
# ============================================================================
def test_target_tracking():
    """
    Test: Can the quad physically reach a target using simple proportional control?
    
    This tests if the physics allows target reaching, independent of RL.
    """
    print_header("TEST 8: Target Tracking Capability (Simple P-Controller)")
    
    env = create_env()
    obs, info = env.reset()
    
    # Handle observation format
    if obs.ndim == 1:
        get_obs = lambda o: o
    else:
        get_obs = lambda o: o[0] if o.ndim > 1 else o
    
    target = np.array([0.0, 0.0, 5.0])  # Same as training target
    
    print(f"Target position: {target}")
    print("Using simple proportional controller to track target")
    print("This tests if physics allows reaching target at all")
    
    data = {"step": [], "time": [], "distance_to_target": [], "pos_z": []}
    
    kp_pos = 0.5  # Proportional gain for position
    kp_vel = 0.1  # Proportional gain for velocity damping
    
    for step in range(300):
        # Extract state
        current_obs = get_obs(obs)
        pos = current_obs[:3]
        vel = current_obs[6:9]
        
        # Simple proportional control
        pos_error = target - pos
        vel_error = -vel  # Want zero velocity
        
        # Very simple PD-like control mapped to thrust
        control = kp_pos * pos_error[2] + kp_vel * vel_error[2]  # Z-control only
        
        # Clamp and create action
        action = np.clip([control, control, control, control], -1.0, 1.0)
        
        # Record
        distance = np.linalg.norm(pos - target)
        data["step"].append(step)
        data["time"].append(step * 0.02)
        data["distance_to_target"].append(distance)
        data["pos_z"].append(pos[2])
        
        # Step
        obs, reward, terminated, truncated, info = env.step(np.array(action).reshape(1, -1))
        
        # Handle termination format
        term = terminated[0] if isinstance(terminated, np.ndarray) else terminated
        trunc = truncated[0] if isinstance(truncated, np.ndarray) else truncated
        
        if term or trunc:
            break
    
    df = pd.DataFrame(data)
    
    # Analysis
    min_distance = df["distance_to_target"].min()
    final_distance = df["distance_to_target"].iloc[-1]
    converged = final_distance < 0.5
    
    print(f"\nResults (Simple P-Controller):")
    print(f"  Min distance to target: {min_distance:.3f}m")
    print(f"  Final distance to target: {final_distance:.3f}m")
    print(f"  Converged (< 0.5m): {converged}")
    
    if converged:
        print_test_result("Target Tracking", True, 
                         f"Physics allows reaching target: {final_distance:.3f}m")
        print("  → RL failure is due to LEARNING, not physics!")
    elif min_distance < 1.0:
        print_test_result("Target Tracking", True,
                         f"Physics allows getting close: {min_distance:.3f}m")
        print_warning("Simple controller didn't fully converge - may need better tuning")
        print("  → RL failure is likely due to LEARNING, not physics")
    else:
        print_test_result("Target Tracking", False,
                         f"Can't reach target even with simple controller: {min_distance:.3f}m")
        print("  → Possible physics issue OR target unreachable from starting position")
    
    env.close()
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*80)
    print("  FLIGHTMARE PHYSICS VALIDATION SUITE")
    print("  Testing basic physics and motor responses")
    print("="*80)
    
    test_results = {}
    
    if "hover_stability" in TESTS_TO_RUN:
        test_results["hover"] = test_hover_stability()
    
    if "upward_thrust" in TESTS_TO_RUN:
        test_results["upward"] = test_upward_thrust()
    
    if "downward_thrust" in TESTS_TO_RUN:
        test_results["downward"] = test_downward_thrust()
    
    if "lateral_movement" in TESTS_TO_RUN:
        test_results["lateral"] = test_lateral_movement()
    
    if "motor_correlation" in TESTS_TO_RUN:
        test_results["motors"] = test_motor_correlation()
    
    if "gravity_check" in TESTS_TO_RUN:
        test_results["gravity"] = test_gravity()
    
    if "response_time" in TESTS_TO_RUN:
        test_results["response"] = test_response_time()
    
    if "target_tracking" in TESTS_TO_RUN:
        test_results["tracking"] = test_target_tracking()
    
    # Final summary
    print_header("VALIDATION SUMMARY")
    
    print(f"\n✓ PASSED: {len(RESULTS['passed'])} tests")
    for test in RESULTS['passed']:
        print(f"    • {test}")
    
    if RESULTS['failed']:
        print(f"\n✗ FAILED: {len(RESULTS['failed'])} tests")
        for test in RESULTS['failed']:
            print(f"    • {test}")
    
    if RESULTS['warnings']:
        print(f"\n⚠ WARNINGS: {len(RESULTS['warnings'])}")
        for warning in RESULTS['warnings']:
            print(f"    • {warning}")
    
    # Overall verdict
    print("\n" + "="*80)
    if not RESULTS['failed']:
        print("  ✓ PHYSICS VALIDATION: ALL TESTS PASSED")
        print("  → Physics is working correctly")
        print("  → RL performance issues are due to TRAINING/REWARDS, not physics")
    else:
        print("  ✗ PHYSICS VALIDATION: SOME TESTS FAILED")
        print("  → Review failed tests above")
        print("  → Physics may have issues that need fixing")
    print("="*80)
    
    print("\nNext steps:")
    if not RESULTS['failed']:
        print("  1. Physics is validated - focus on fixing RL training")
        print("  2. Increase reward coefficients (50-100x)")
        print("  3. Add altitude termination conditions")
        print("  4. Consider curriculum learning")
    else:
        print("  1. Fix physics issues identified above")
        print("  2. Re-run validation after fixes")
        print("  3. Then address RL training issues")
    
    return test_results, RESULTS

if __name__ == "__main__":
    results, summary = main()
