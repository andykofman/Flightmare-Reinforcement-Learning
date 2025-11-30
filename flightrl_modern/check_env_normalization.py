#!/usr/bin/env python3
"""
Print what the environment is actually using for normalization
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from flightrl_modern.envs.gymnasium_wrapper import make_flight_env

# Create environment to see what parameters it's using
env = make_flight_env(render=False, num_envs=1, num_threads=1, seed=42)

# The environment doesn't expose act_mean/act_std directly
# But we can infer it from the action space

print("Environment created successfully")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Test what thrust values actually result from actions
test_actions = {
    "Zero (hover)": np.array([[0.0, 0.0, 0.0, 0.0]]),
    "+20%": np.array([[0.2, 0.2, 0.2, 0.2]]),
    "+100%": np.array([[1.0, 1.0, 1.0, 1.0]]),
}

env.reset()

for name, action in test_actions.items():
    # We can't directly see the thrust values from Python
    # But we know: quad_act = action * act_std + act_mean
    #
    # With mass=0.73, Gz=-9.81:
    # act_mean = 0.73 * 9.81 / 4 = 1.790 N
    # act_std = 0.73 * 2 * 9.81 / 4 = 3.581 N
    
    expected_thrust =action[0][0] * 3.581 + 1.790
    print(f"\n{name}: action={action[0]}")
    print(f"  Expected thrust/motor: {expected_thrust:.3f} N")
    print(f"  Expected total: {expected_thrust * 4:.3f} N")

env.close()

print("\n" + "="*60)
print("CHECKING IF THERE'S A CONTROL MODE BUG")
print("="*60)

# From previous conversation history, there was a control_mode bug
# Let me check if it's documented

print("\nSearching for control_mode issues in development log...")
