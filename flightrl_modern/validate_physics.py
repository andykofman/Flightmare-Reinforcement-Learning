#!/usr/bin/env python3
"""
FLIGHTMARE PHYSICS VALIDATION SUITE - Production Version

A comprehensive test suite to validate quadrotor physics simulation.
All tests use physically-derived thresholds based on the configured parameters.

Configuration (from quadrotor_env.yaml):
- Mass: 0.73 kg
- Gravity: 9.81 m/s²
- sim_dt: 0.02s (50 Hz)
- motor_tau: 0.0001s (near-instant response)

Test Categories:
1. Fundamental Physics (gravity, thrust, mass)
2. Control Response (motors, dynamics)
3. Kinematic Consistency (position/velocity/acceleration)
4. System Integration (multi-axis, tracking)

Exit Codes:
- 0: All tests passed
- 1: One or more tests failed
- 2: Critical error (environment creation failed)
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add flightrl_modern to path
sys.path.insert(0, str(Path(__file__).parent))

from flightrl_modern.envs.gymnasium_wrapper import make_flight_env

# =============================================================================
# CONFIGURATION - Physics Constants & Test Parameters
# =============================================================================

@dataclass
class PhysicsConfig:
    """Expected physics parameters from configuration"""
    mass: float = 0.73  # kg
    gravity: float = 9.81  # m/s²
    sim_dt: float = 0.02  # seconds
    motor_tau: float = 0.0001  # seconds (motor time constant)
    
    # Derived values
    @property
    def weight(self) -> float:
        return self.mass * self.gravity
    
    @property
    def hover_thrust_per_motor(self) -> float:
        return self.weight / 4.0


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    criteria: Dict[str, str] = field(default_factory=dict)


@dataclass 
class ValidationReport:
    """Complete validation report"""
    timestamp: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return self.passed / self.total_tests if self.total_tests > 0 else 0.0
    
    @property
    def all_passed(self) -> bool:
        return self.failed == 0


PHYSICS = PhysicsConfig()
REPORT = ValidationReport(timestamp=datetime.now().isoformat())

# =============================================================================
# TEST THRESHOLDS - Physically Derived
# =============================================================================

class Thresholds:
    """Strict but physically justified test thresholds"""
    
    # Hover stability: acceleration should be < 5% of gravity
    HOVER_ACCEL_MAX = 0.5  # m/s² (≈5% of g)
    
    # Gravity: should be within 5% of 9.81
    GRAVITY_MIN = 9.31  # m/s² (9.81 - 5%)
    GRAVITY_MAX = 10.31  # m/s² (9.81 + 5%)
    
    # Thrust response: 20% action should give ~20% extra thrust
    # Extra accel = (0.2 * 2 * weight) / mass = 0.2 * 2 * g ≈ 3.92 m/s²
    UPWARD_ACCEL_MIN = 2.5  # m/s² (allow some margin)
    UPWARD_ACCEL_MAX = 6.0  # m/s²
    
    # Downward: negative thrust reduces lift, gravity dominates
    # At -0.2 action: thrust ≈ 0.6 * hover, net accel ≈ -4 m/s²
    DOWNWARD_ACCEL_MIN = -8.0  # m/s²
    DOWNWARD_ACCEL_MAX = -2.0  # m/s²
    
    # Motor response time (63% rise time)
    MOTOR_RESPONSE_MAX = 0.1  # seconds (motor_tau is 0.0001s)
    
    # Position update: must see measurable change with thrust
    MIN_DISPLACEMENT = 0.5  # meters (after 3 seconds of thrust)
    
    # Velocity change threshold
    MIN_VELOCITY_CHANGE = 1.0  # m/s
    
    # Symmetry: opposite motors should have similar effects
    SYMMETRY_TOLERANCE = 0.3  # 30% difference allowed
    
    # Lateral movement from differential thrust
    LATERAL_MIN_DISPLACEMENT = 0.5  # meters
    
    # Angular response
    MIN_ANGULAR_RESPONSE = 0.1  # rad/s


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_env(seed: int = 42):
    """Create environment with error handling"""
    try:
        env = make_flight_env(
            render=False,
            num_envs=1,
            num_threads=1,
            seed=seed
        )
        return env
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create environment: {e}")
        sys.exit(2)


def extract_obs(obs: np.ndarray) -> np.ndarray:
    """Extract single observation from potentially batched array"""
    if obs.ndim == 1:
        return obs
    return obs[0] if obs.ndim > 1 else obs


def estimate_acceleration(times: np.ndarray, velocities: np.ndarray) -> float:
    """Estimate acceleration from velocity time series using linear fit"""
    if len(times) < 3:
        return 0.0
    coeffs = np.polyfit(times, velocities, 1)
    return coeffs[0]


def run_trajectory(env, action: np.ndarray, num_steps: int) -> Dict[str, np.ndarray]:
    """Run trajectory and collect data"""
    obs, _ = env.reset()
    
    data = {
        'time': [], 'pos': [], 'vel': [], 'reward': []
    }
    
    for step in range(num_steps):
        current_obs = extract_obs(obs)
        
        data['time'].append(step * PHYSICS.sim_dt)
        data['pos'].append(current_obs[:3].copy())
        data['vel'].append(current_obs[6:9].copy())
        
        action_input = action.reshape(1, -1) if action.ndim == 1 else action
        obs, reward, terminated, truncated, _ = env.step(action_input)
        
        r = reward[0] if isinstance(reward, np.ndarray) else float(reward)
        data['reward'].append(r)
        
        term = terminated[0] if isinstance(terminated, np.ndarray) else terminated
        trunc = truncated[0] if isinstance(truncated, np.ndarray) else truncated
        if term or trunc:
            break
    
    return {k: np.array(v) for k, v in data.items()}


def print_header(title: str):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def record_result(name: str, passed: bool, message: str, 
                  metrics: Dict = None, criteria: Dict = None):
    """Record test result"""
    result = TestResult(
        name=name,
        passed=passed,
        message=message,
        metrics=metrics or {},
        criteria=criteria or {}
    )
    REPORT.results.append(result)
    REPORT.total_tests += 1
    if passed:
        REPORT.passed += 1
    else:
        REPORT.failed += 1
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{status}: {name}")
    print(f"  → {message}")
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")


# =============================================================================
# TEST 1: HOVER THRUST BALANCE
# =============================================================================

def test_hover_thrust_balance():
    """
    Verify that action=[0,0,0,0] produces thrust equal to weight.
    
    Physics: At hover, thrust = weight = m*g
    With action=0: thrust_per_motor = act_mean = (m*g)/4
    Net force should be zero → acceleration ≈ 0
    
    Criterion: |acceleration| < 0.5 m/s² (< 5% of g)
    """
    print_header("TEST 1: Hover Thrust Balance")
    print("Verifying action=[0,0,0,0] produces thrust ≈ weight")
    print(f"Expected: |acceleration| < {Thresholds.HOVER_ACCEL_MAX} m/s²")
    
    env = create_env()
    action = np.array([0.0, 0.0, 0.0, 0.0])
    
    data = run_trajectory(env, action, num_steps=150)  # 3 seconds
    env.close()
    
    # Estimate vertical acceleration
    times = data['time']
    vel_z = data['vel'][:, 2]
    accel_z = estimate_acceleration(times, vel_z)
    
    # Check velocity consistency
    vel_z_std = np.std(vel_z)
    vel_z_mean = np.mean(vel_z)
    
    passed = abs(accel_z) < Thresholds.HOVER_ACCEL_MAX
    
    record_result(
        name="Hover Thrust Balance",
        passed=passed,
        message=f"Z acceleration = {accel_z:.3f} m/s² (threshold: ±{Thresholds.HOVER_ACCEL_MAX})",
        metrics={
            "z_acceleration_m/s²": accel_z,
            "z_velocity_mean_m/s": vel_z_mean,
            "z_velocity_std_m/s": vel_z_std,
        },
        criteria={
            "max_acceleration": f"±{Thresholds.HOVER_ACCEL_MAX} m/s²"
        }
    )
    
    return passed


# =============================================================================
# TEST 2: GRAVITY MAGNITUDE
# =============================================================================

def test_gravity_magnitude():
    """
    Verify gravity acceleration is correct (~9.81 m/s²).
    
    Physics: With zero thrust (action=-1 gives minimum thrust ≈ 0),
    acceleration should be -g = -9.81 m/s²
    
    Criterion: 9.31 < |a| < 10.31 m/s² (±5% of g)
    """
    print_header("TEST 2: Gravity Magnitude")
    print("Verifying gravity ≈ 9.81 m/s² with minimal thrust")
    print(f"Expected: {Thresholds.GRAVITY_MIN} < |a| < {Thresholds.GRAVITY_MAX} m/s²")
    
    env = create_env()
    action = np.array([-1.0, -1.0, -1.0, -1.0])  # Minimum thrust
    
    data = run_trajectory(env, action, num_steps=50)  # 1 second
    env.close()
    
    times = data['time']
    vel_z = data['vel'][:, 2]
    accel_z = estimate_acceleration(times, vel_z)
    
    # Gravity should cause negative acceleration
    gravity_magnitude = abs(accel_z)
    
    passed = Thresholds.GRAVITY_MIN < gravity_magnitude < Thresholds.GRAVITY_MAX
    
    record_result(
        name="Gravity Magnitude",
        passed=passed,
        message=f"Measured gravity = {gravity_magnitude:.3f} m/s² (expected: ~9.81)",
        metrics={
            "gravity_magnitude_m/s²": gravity_magnitude,
            "expected_m/s²": 9.81,
            "error_percent": abs(gravity_magnitude - 9.81) / 9.81 * 100
        },
        criteria={
            "range": f"{Thresholds.GRAVITY_MIN} - {Thresholds.GRAVITY_MAX} m/s²"
        }
    )
    
    return passed


# =============================================================================
# TEST 3: UPWARD THRUST RESPONSE
# =============================================================================

def test_upward_thrust():
    """
    Verify positive thrust causes upward acceleration.
    
    Physics: action=0.2 gives thrust = act_mean + 0.2*act_std
    = (m*g)/4 + 0.2*(2*m*g)/4 = 1.4*(m*g)/4 per motor
    Total thrust = 1.4*m*g, Net force = 0.4*m*g upward
    Acceleration = 0.4*g ≈ 3.92 m/s²
    
    Criterion: 2.5 < a < 6.0 m/s² (allowing margin for initial conditions)
    """
    print_header("TEST 3: Upward Thrust Response")
    print("Verifying action=[0.2,0.2,0.2,0.2] causes upward acceleration")
    print(f"Expected: {Thresholds.UPWARD_ACCEL_MIN} < a < {Thresholds.UPWARD_ACCEL_MAX} m/s²")
    
    env = create_env()
    action = np.array([0.2, 0.2, 0.2, 0.2])
    
    data = run_trajectory(env, action, num_steps=100)  # 2 seconds
    env.close()
    
    times = data['time'][:50]  # First second for cleaner estimate
    vel_z = data['vel'][:50, 2]
    accel_z = estimate_acceleration(times, vel_z)
    
    # Check displacement
    pos_z = data['pos'][:, 2]
    displacement = pos_z[-1] - pos_z[0]
    
    passed = (Thresholds.UPWARD_ACCEL_MIN < accel_z < Thresholds.UPWARD_ACCEL_MAX 
              and displacement > Thresholds.MIN_DISPLACEMENT)
    
    record_result(
        name="Upward Thrust Response",
        passed=passed,
        message=f"Acceleration = {accel_z:.3f} m/s², displacement = {displacement:.2f}m",
        metrics={
            "z_acceleration_m/s²": accel_z,
            "z_displacement_m": displacement,
            "expected_accel_m/s²": 0.4 * PHYSICS.gravity,
        },
        criteria={
            "acceleration_range": f"{Thresholds.UPWARD_ACCEL_MIN} - {Thresholds.UPWARD_ACCEL_MAX} m/s²",
            "min_displacement": f"{Thresholds.MIN_DISPLACEMENT}m"
        }
    )
    
    return passed


# =============================================================================
# TEST 4: DOWNWARD THRUST RESPONSE  
# =============================================================================

def test_downward_thrust():
    """
    Verify negative thrust causes increased descent rate.
    
    Physics: action=-0.2 gives thrust = act_mean - 0.2*act_std
    = (m*g)/4 - 0.2*(2*m*g)/4 = 0.6*(m*g)/4 per motor
    Total thrust = 0.6*m*g, Net force = -0.4*m*g downward
    Acceleration = -0.4*g ≈ -3.92 m/s² (plus gravity effects)
    
    Criterion: -8.0 < a < -2.0 m/s²
    """
    print_header("TEST 4: Downward Thrust Response")
    print("Verifying action=[-0.2,-0.2,-0.2,-0.2] causes descent")
    print(f"Expected: {Thresholds.DOWNWARD_ACCEL_MIN} < a < {Thresholds.DOWNWARD_ACCEL_MAX} m/s²")
    
    env = create_env()
    action = np.array([-0.2, -0.2, -0.2, -0.2])
    
    data = run_trajectory(env, action, num_steps=100)
    env.close()
    
    times = data['time'][:50]
    vel_z = data['vel'][:50, 2]
    accel_z = estimate_acceleration(times, vel_z)
    
    pos_z = data['pos'][:, 2]
    displacement = pos_z[-1] - pos_z[0]
    
    passed = (Thresholds.DOWNWARD_ACCEL_MIN < accel_z < Thresholds.DOWNWARD_ACCEL_MAX
              and displacement < -Thresholds.MIN_DISPLACEMENT)
    
    record_result(
        name="Downward Thrust Response",
        passed=passed,
        message=f"Acceleration = {accel_z:.3f} m/s², displacement = {displacement:.2f}m",
        metrics={
            "z_acceleration_m/s²": accel_z,
            "z_displacement_m": displacement,
        },
        criteria={
            "acceleration_range": f"{Thresholds.DOWNWARD_ACCEL_MIN} - {Thresholds.DOWNWARD_ACCEL_MAX} m/s²"
        }
    )
    
    return passed


# =============================================================================
# TEST 5: THRUST LINEARITY
# =============================================================================

def test_thrust_linearity():
    """
    Verify thrust scales linearly with action.
    
    Physics: acceleration should increase linearly with action
    a(action) = (2*action)*g approximately
    
    Test actions: -0.3, 0, 0.3 and check linear relationship
    """
    print_header("TEST 5: Thrust Linearity")
    print("Verifying thrust scales linearly with action input")
    
    test_actions = [-0.3, 0.0, 0.3]
    accelerations = []
    
    for act_val in test_actions:
        env = create_env(seed=42 + int(act_val * 100))
        action = np.array([act_val, act_val, act_val, act_val])
        data = run_trajectory(env, action, num_steps=75)
        env.close()
        
        times = data['time'][:50]
        vel_z = data['vel'][:50, 2]
        accel = estimate_acceleration(times, vel_z)
        accelerations.append(accel)
        print(f"  Action={act_val:.1f}: acceleration={accel:.3f} m/s²")
    
    # Check linearity: fit line and check R²
    coeffs = np.polyfit(test_actions, accelerations, 1)
    predicted = np.polyval(coeffs, test_actions)
    ss_res = np.sum((np.array(accelerations) - predicted) ** 2)
    ss_tot = np.sum((np.array(accelerations) - np.mean(accelerations)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Check slope is positive (more thrust = more upward accel)
    slope = coeffs[0]
    
    passed = r_squared > 0.9 and slope > 0
    
    record_result(
        name="Thrust Linearity",
        passed=passed,
        message=f"R² = {r_squared:.4f}, slope = {slope:.3f} m/s² per unit action",
        metrics={
            "r_squared": r_squared,
            "slope": slope,
            "accelerations": accelerations,
        },
        criteria={
            "min_r_squared": "0.9",
            "slope_sign": "positive"
        }
    )
    
    return passed


# =============================================================================
# TEST 6: MOTOR DIFFERENTIAL EFFECTS
# =============================================================================

def test_motor_differential():
    """
    Verify each motor independently affects the quadrotor.
    
    Physics: Single motor thrust creates both lift and torque.
    Different motors should create different torque directions.
    """
    print_header("TEST 6: Motor Differential Effects")
    print("Verifying each motor creates unique response")
    
    motor_responses = []
    
    for motor_id in range(4):
        env = create_env(seed=100 + motor_id)
        action = np.zeros(4)
        action[motor_id] = 0.5  # 50% on one motor
        
        data = run_trajectory(env, action, num_steps=75)
        env.close()
        
        # Measure response
        pos = data['pos']
        vel = data['vel']
        
        z_change = pos[-1, 2] - pos[0, 2]
        xy_change = np.sqrt((pos[-1, 0] - pos[0, 0])**2 + (pos[-1, 1] - pos[0, 1])**2)
        
        motor_responses.append({
            'motor': motor_id,
            'z_change': z_change,
            'xy_change': xy_change,
            'final_vel': vel[-1].copy()
        })
        
        print(f"  Motor {motor_id}: ΔZ={z_change:+.3f}m, ΔXY={xy_change:.3f}m")
    
    # Check all motors have effect
    all_active = all(
        abs(r['z_change']) > 0.1 or r['xy_change'] > 0.2 
        for r in motor_responses
    )
    
    # Check motors create different effects (not all identical)
    z_changes = [r['z_change'] for r in motor_responses]
    z_variance = np.var(z_changes)
    
    passed = all_active and z_variance > 0.01
    
    record_result(
        name="Motor Differential Effects",
        passed=passed,
        message=f"All motors active: {all_active}, Z variance: {z_variance:.4f}",
        metrics={
            "all_motors_active": all_active,
            "z_variance": z_variance,
            "z_changes": z_changes,
        },
        criteria={
            "all_active": "True",
            "min_z_variance": "0.01"
        }
    )
    
    return passed


# =============================================================================
# TEST 7: PITCH/ROLL FROM DIFFERENTIAL THRUST
# =============================================================================

def test_differential_thrust_rotation():
    """
    Verify differential thrust creates rotation and lateral movement.
    
    Physics: Asymmetric thrust creates torque → rotation → lateral force
    """
    print_header("TEST 7: Differential Thrust Rotation")
    print("Verifying asymmetric thrust causes pitch/roll and lateral motion")
    
    env = create_env()
    # Motors 0,1 high, 2,3 low → should create pitch and forward motion
    action = np.array([0.15, 0.15, -0.15, -0.15])
    
    data = run_trajectory(env, action, num_steps=150)
    env.close()
    
    pos = data['pos']
    
    # Measure lateral displacement
    xy_displacement = np.sqrt(
        (pos[-1, 0] - pos[0, 0])**2 + 
        (pos[-1, 1] - pos[0, 1])**2
    )
    
    # Measure max lateral velocity
    vel = data['vel']
    xy_vel = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
    max_xy_vel = np.max(xy_vel)
    
    passed = xy_displacement > Thresholds.LATERAL_MIN_DISPLACEMENT
    
    record_result(
        name="Differential Thrust Rotation",
        passed=passed,
        message=f"Lateral displacement = {xy_displacement:.3f}m, max velocity = {max_xy_vel:.3f}m/s",
        metrics={
            "xy_displacement_m": xy_displacement,
            "max_xy_velocity_m/s": max_xy_vel,
        },
        criteria={
            "min_displacement": f"{Thresholds.LATERAL_MIN_DISPLACEMENT}m"
        }
    )
    
    return passed


# =============================================================================
# TEST 8: POSITION-VELOCITY CONSISTENCY
# =============================================================================

def test_position_velocity_consistency():
    """
    Verify position changes are consistent with integrated velocity.
    
    Physics: Δposition ≈ ∫velocity dt
    This catches bugs where position/velocity are decoupled.
    """
    print_header("TEST 8: Position-Velocity Consistency")
    print("Verifying position change matches integrated velocity")
    
    env = create_env()
    action = np.array([0.25, 0.25, 0.25, 0.25])  # Upward thrust
    
    data = run_trajectory(env, action, num_steps=100)
    env.close()
    
    pos = data['pos']
    vel = data['vel']
    times = data['time']
    
    # Actual position change
    actual_z_change = pos[-1, 2] - pos[0, 2]
    
    # Integrated velocity (trapezoidal rule)
    dt = PHYSICS.sim_dt
    integrated_z_change = np.trapz(vel[:, 2], dx=dt)
    
    # Error
    error = abs(actual_z_change - integrated_z_change)
    relative_error = error / max(abs(actual_z_change), 0.1)
    
    passed = relative_error < 0.15  # 15% tolerance
    
    record_result(
        name="Position-Velocity Consistency",
        passed=passed,
        message=f"Actual ΔZ={actual_z_change:.3f}m, Integrated={integrated_z_change:.3f}m, Error={relative_error*100:.1f}%",
        metrics={
            "actual_z_change_m": actual_z_change,
            "integrated_z_change_m": integrated_z_change,
            "relative_error_percent": relative_error * 100,
        },
        criteria={
            "max_relative_error": "15%"
        }
    )
    
    return passed


# =============================================================================
# TEST 9: VELOCITY-ACCELERATION CONSISTENCY
# =============================================================================

def test_velocity_acceleration_consistency():
    """
    Verify velocity changes are consistent with acceleration.
    
    Physics: Δvelocity ≈ acceleration * Δt
    """
    print_header("TEST 9: Velocity-Acceleration Consistency")
    print("Verifying velocity change matches expected from acceleration")
    
    env = create_env()
    action = np.array([0.3, 0.3, 0.3, 0.3])  # Strong upward
    
    data = run_trajectory(env, action, num_steps=50)
    env.close()
    
    vel = data['vel']
    times = data['time']
    
    # Actual velocity change
    actual_vel_change = vel[-1, 2] - vel[0, 2]
    
    # Expected from constant acceleration model
    accel = estimate_acceleration(times, vel[:, 2])
    expected_vel_change = accel * (times[-1] - times[0])
    
    error = abs(actual_vel_change - expected_vel_change)
    relative_error = error / max(abs(actual_vel_change), 0.1)
    
    passed = relative_error < 0.2  # 20% tolerance
    
    record_result(
        name="Velocity-Acceleration Consistency",
        passed=passed,
        message=f"Actual Δv={actual_vel_change:.3f}m/s, Expected={expected_vel_change:.3f}m/s",
        metrics={
            "actual_vel_change_m/s": actual_vel_change,
            "expected_vel_change_m/s": expected_vel_change,
            "measured_accel_m/s²": accel,
            "relative_error_percent": relative_error * 100,
        },
        criteria={
            "max_relative_error": "20%"
        }
    )
    
    return passed


# =============================================================================
# TEST 10: SYMMETRIC MOTOR ACCELERATION
# =============================================================================

def test_motor_symmetry():
    """
    Verify all motors produce similar ACCELERATION when activated equally.
    
    Physics: Each motor should produce approximately the same thrust.
    We test acceleration (not position) because it's independent of initial velocity.
    
    Method: Compare acceleration from [0.4,0,0,0] vs [0,0.4,0,0] etc.
    All should produce similar Z-acceleration magnitude.
    """
    print_header("TEST 10: Motor Thrust Symmetry")
    print("Verifying all motors produce similar acceleration")
    print("(Testing acceleration, not position, to be independent of initial conditions)")
    
    motor_accelerations = {}
    
    for motor_id in range(4):
        # Use SAME seed for all motors - fair comparison
        env = create_env(seed=42)
        action = np.zeros(4)
        action[motor_id] = 0.5  # 50% thrust on single motor
        
        data = run_trajectory(env, action, num_steps=50)
        env.close()
        
        # Measure Z acceleration (independent of initial velocity)
        times = data['time'][:40]
        vel_z = data['vel'][:40, 2]
        accel_z = estimate_acceleration(times, vel_z)
        
        motor_accelerations[motor_id] = accel_z
        print(f"  Motor {motor_id}: Z acceleration = {accel_z:.3f} m/s²")
    
    # All motors should produce similar acceleration magnitude
    accels = list(motor_accelerations.values())
    accel_mean = np.mean(accels)
    accel_std = np.std(accels)
    accel_range = max(accels) - min(accels)
    
    # Coefficient of variation (std/mean) should be small
    cv = abs(accel_std / accel_mean) if abs(accel_mean) > 0.1 else accel_std
    
    # Pass if variation is < 30%
    passed = cv < Thresholds.SYMMETRY_TOLERANCE
    
    record_result(
        name="Motor Thrust Symmetry",
        passed=passed,
        message=f"Acceleration CV = {cv*100:.1f}% (mean={accel_mean:.2f}, std={accel_std:.2f} m/s²)",
        metrics={
            "motor_0_accel_m/s²": motor_accelerations[0],
            "motor_1_accel_m/s²": motor_accelerations[1],
            "motor_2_accel_m/s²": motor_accelerations[2],
            "motor_3_accel_m/s²": motor_accelerations[3],
            "mean_accel_m/s²": accel_mean,
            "std_accel_m/s²": accel_std,
            "coefficient_of_variation": cv,
        },
        criteria={
            "max_cv": f"{Thresholds.SYMMETRY_TOLERANCE*100}%"
        }
    )
    
    return passed


# =============================================================================
# TEST 11: PD CONTROLLER CONVERGENCE
# =============================================================================

def test_pd_controller_convergence():
    """
    Verify a simple PD controller can stabilize to a target.
    
    This tests that physics supports basic control, independent of RL.
    """
    print_header("TEST 11: PD Controller Convergence")
    print("Verifying physics supports basic altitude control")
    
    env = create_env()
    obs, _ = env.reset()
    
    target_z = 5.0
    kp = 0.8
    kd = 0.3
    
    distances = []
    
    for step in range(250):  # 5 seconds
        current_obs = extract_obs(obs)
        pos_z = current_obs[2]
        vel_z = current_obs[8]
        
        # PD control
        error = target_z - pos_z
        control = kp * error - kd * vel_z
        action = np.clip([control] * 4, -1.0, 1.0)
        
        distances.append(abs(error))
        
        obs, _, terminated, truncated, _ = env.step(np.array(action).reshape(1, -1))
        
        term = terminated[0] if isinstance(terminated, np.ndarray) else terminated
        trunc = truncated[0] if isinstance(truncated, np.ndarray) else truncated
        if term or trunc:
            break
    
    env.close()
    
    min_distance = min(distances)
    final_distance = distances[-1] if distances else float('inf')
    
    # Check for convergence (reaches within 0.5m at some point)
    converged = min_distance < 0.5
    
    passed = converged
    
    record_result(
        name="PD Controller Convergence",
        passed=passed,
        message=f"Min distance to target: {min_distance:.3f}m, Final: {final_distance:.3f}m",
        metrics={
            "min_distance_m": min_distance,
            "final_distance_m": final_distance,
            "target_z_m": target_z,
            "converged": converged,
        },
        criteria={
            "convergence_threshold": "0.5m"
        }
    )
    
    return passed


# =============================================================================
# TEST 12: PHYSICS EQUATION CONSISTENCY
# =============================================================================

def test_physics_equations():
    """
    Verify physics equations are consistent throughout a trajectory.
    
    Test: F = ma should hold. With constant action, acceleration should be constant.
    We measure acceleration at different points in the trajectory and verify consistency.
    
    Note: We don't test cross-run determinism because reset() has internal randomness.
    Instead, we verify the physics EQUATIONS are consistently applied.
    """
    print_header("TEST 12: Physics Equation Consistency")
    print("Verifying F=ma holds throughout trajectory (constant action → constant acceleration)")
    
    env = create_env()
    action = np.array([0.15, 0.15, 0.15, 0.15])  # Constant thrust
    
    data = run_trajectory(env, action, num_steps=100)
    env.close()
    
    times = data['time']
    vel_z = data['vel'][:, 2]
    
    # Measure acceleration in first half and second half
    mid = len(times) // 2
    
    accel_first_half = estimate_acceleration(times[:mid], vel_z[:mid])
    accel_second_half = estimate_acceleration(times[mid:], vel_z[mid:])
    
    # They should be very similar (constant thrust → constant acceleration)
    accel_diff = abs(accel_first_half - accel_second_half)
    avg_accel = (abs(accel_first_half) + abs(accel_second_half)) / 2
    relative_diff = accel_diff / max(avg_accel, 0.1)
    
    # Also check that velocity increases linearly (constant accel)
    # Fit quadratic to position and check residuals
    pos_z = data['pos'][:, 2]
    coeffs = np.polyfit(times, pos_z, 2)  # Should be parabolic
    predicted = np.polyval(coeffs, times)
    residuals = np.abs(pos_z - predicted)
    max_residual = np.max(residuals)
    
    passed = relative_diff < 0.15 and max_residual < 0.5
    
    record_result(
        name="Physics Equation Consistency",
        passed=passed,
        message=f"Acceleration consistency: {relative_diff*100:.1f}% diff, Position fit residual: {max_residual:.3f}m",
        metrics={
            "accel_first_half_m/s²": accel_first_half,
            "accel_second_half_m/s²": accel_second_half,
            "relative_difference_percent": relative_diff * 100,
            "position_fit_max_residual_m": max_residual,
        },
        criteria={
            "max_accel_difference": "15%",
            "max_position_residual": "0.5m"
        }
    )
    
    return passed


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_tests():
    """Run all physics validation tests"""
    
    print("\n" + "="*80)
    print("  FLIGHTMARE PHYSICS VALIDATION SUITE")
    print("  Production Version - Strict Criteria")
    print("="*80)
    print(f"\nPhysics Configuration:")
    print(f"  Mass: {PHYSICS.mass} kg")
    print(f"  Gravity: {PHYSICS.gravity} m/s²")
    print(f"  Weight: {PHYSICS.weight:.3f} N")
    print(f"  Hover thrust/motor: {PHYSICS.hover_thrust_per_motor:.3f} N")
    print(f"  Simulation dt: {PHYSICS.sim_dt} s")
    
    # Run all tests
    tests = [
        test_hover_thrust_balance,
        test_gravity_magnitude,
        test_upward_thrust,
        test_downward_thrust,
        test_thrust_linearity,
        test_motor_differential,
        test_differential_thrust_rotation,
        test_position_velocity_consistency,
        test_velocity_acceleration_consistency,
        test_motor_symmetry,  # Now tests acceleration symmetry
        test_pd_controller_convergence,
        test_physics_equations,  # Tests F=ma consistency
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            record_result(
                name=test_func.__name__,
                passed=False,
                message=f"Test crashed: {str(e)}",
                metrics={"error": str(e)}
            )
    
    # Print summary
    print_header("VALIDATION SUMMARY")
    
    print(f"\n  Total Tests: {REPORT.total_tests}")
    print(f"  ✓ Passed: {REPORT.passed}")
    print(f"  ✗ Failed: {REPORT.failed}")
    print(f"  Success Rate: {REPORT.success_rate*100:.1f}%")
    
    if REPORT.failed > 0:
        print(f"\n  Failed Tests:")
        for result in REPORT.results:
            if not result.passed:
                print(f"    • {result.name}")
    
    print("\n" + "="*80)
    if REPORT.all_passed:
        print("  ✓ PHYSICS VALIDATION: ALL TESTS PASSED")
        print("  → Physics simulation is working correctly")
        print("  → Safe to proceed with RL training")
    else:
        print("  ✗ PHYSICS VALIDATION: SOME TESTS FAILED")
        print(f"  → {REPORT.failed}/{REPORT.total_tests} tests failed")
        print("  → Review failed tests and fix physics issues")
    print("="*80)
    
    # Save report to JSON
    report_path = Path(__file__).parent / "physics_validation_report.json"
    with open(report_path, 'w') as f:
        report_dict = asdict(REPORT)
        json.dump(report_dict, f, indent=2, default=str)
    print(f"\nDetailed report saved to: {report_path}")
    
    return REPORT


if __name__ == "__main__":
    report = run_all_tests()
    sys.exit(0 if report.all_passed else 1)
