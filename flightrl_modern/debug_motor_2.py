#!/usr/bin/env python3
"""
Debug script to trace Motor 2 failure
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from flightrl_modern.envs.gymnasium_wrapper import make_flight_env

def test_individual_motors():
    """Test each motor individually to see what actions produce"""
    print("="*80)
    print("  MOTOR DEBUG: Individual Motor Action Mapping")
    print("="*80)
    
    for motor_id in range(4):
        print(f"\n--- Motor {motor_id} ---")
        
        env = make_flight_env(render=False, num_envs=1, num_threads=1, seed=42)
        obs, info = env.reset()
        
        # Create action with only this motor activated
        action = np.zeros(4)
        action[motor_id] = 0.3  # 30% positive thrust
        
        print(f"Input action: {action}")
        
        # Step once to see what happens
        obs_new, reward, term, trunc, info = env.step(action.reshape(1, -1))
        
        if obs.ndim > 1:
            obs = obs[0]
            obs_new = obs_new[0]
        
        # Check position change
        pos_change = obs_new[:3] - obs[:3]
        vel_change = obs_new[6:9] - obs[6:9]
        
        print(f"Position change: {pos_change}")
        print(f"Velocity change: {vel_change}")
        print(f"Total motion: {np.linalg.norm(pos_change):.6f}m")
        
        if np.linalg.norm(pos_change) < 0.0001:
            print("❌ MOTOR NOT RESPONDING!")
        else:
            print("✅ Motor responds")
        
        env.close()
    
    # Now test action normalization
    print("\n" + "="*80)
    print("  ACTION NORMALIZATION TEST")
    print("="*80)
    
    env = make_flight_env(render=False, num_envs=1, num_threads=1, seed=42)
    obs, info = env.reset()
    
    test_actions = [
        np.array([0.0, 0.0, 0.0, 0.0]),  # Zero action (hover)
        np.array([0.2, 0.2, 0.2, 0.2]),  # +20% all
        np.array([-0.2, -0.2, -0.2, -0.2]),  # -20% all
        np.array([0.3, 0.0, 0.0, 0.0]),  # Motor 0 only
        np.array([0.0, 0.3, 0.0, 0.0]),  # Motor 1 only
        np.array([0.0, 0.0, 0.3, 0.0]),  # Motor 2 only ← SUSPECT
        np.array([0.0, 0.0, 0.0, 0.3]),  # Motor 3 only
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\nTest {i}: action = {action}")
        
        # We'd need to inspect the C++ side to see actual thrust values
        # But we can at least check if the action is accepted
        try:
            obs_new, reward, term, trunc, info = env.step(action.reshape(1, -1))
            print(f"  ✓ Action accepted, reward={reward}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    env.close()

if __name__ == "__main__":
    test_individual_motors()
