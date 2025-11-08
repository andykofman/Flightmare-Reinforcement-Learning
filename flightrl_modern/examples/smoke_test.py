#!/usr/bin/env python3
"""
Quick smoke test for Flightmare RL

Runs a very short training loop to verify everything is working.
This is used in Docker build verification and CI.

Usage:
    python smoke_test.py
"""

import sys
import os


def smoke_test():
    """Run a quick smoke test of the RL stack"""
    
    print("\n" + "="*60)
    print("Flightmare RL Smoke Test")
    print("="*60 + "\n")
    
    # Test 1: Import flightgym
    print("[1/5] Testing flightgym import...")
    try:
        import flightgym
        print("  ✓ flightgym imported successfully")
        print(f"    Available: {dir(flightgym)}")
    except ImportError as e:
        print(f"  ✗ Failed to import flightgym: {e}")
        return False
    
    # Test 2: Import PyTorch
    print("\n[2/5] Testing PyTorch import...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ Failed to import torch: {e}")
        return False
    
    # Test 3: Import Stable-Baselines3
    print("\n[3/5] Testing Stable-Baselines3 import...")
    try:
        import stable_baselines3 as sb3
        print(f"  ✓ Stable-Baselines3 {sb3.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import stable_baselines3: {e}")
        return False
    
    # Test 4: Create environment
    print("\n[4/5] Testing environment creation...")
    try:
        from flightrl_modern.envs.gymnasium_wrapper import make_flight_env
        
        # Set FLIGHTMARE_PATH if not set (for testing)
        if "FLIGHTMARE_PATH" not in os.environ:
            # Try to find it relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            flightmare_path = os.path.abspath(os.path.join(script_dir, "..", ".."))
            os.environ["FLIGHTMARE_PATH"] = flightmare_path
            print(f"  Setting FLIGHTMARE_PATH={flightmare_path}")
        
        env = make_flight_env(render=False, num_envs=1, num_threads=1)
        print("  ✓ Environment created successfully")
        
        # Test reset and step
        obs, info = env.reset(seed=42)
        print(f"    Observation shape: {obs.shape}")
        print(f"    Observation space: {env.observation_space}")
        print(f"    Action space: {env.action_space}")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"    Step executed successfully")
        print(f"      Reward: {reward:.4f}")
        print(f"      Terminated: {terminated}, Truncated: {truncated}")
        
        env.close()
        print("  ✓ Environment test passed")
        
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Quick SAC training
    print("\n[5/5] Testing SAC training (100 steps)...")
    try:
        from stable_baselines3 import SAC
        from flightrl_modern.envs.gymnasium_wrapper import make_flight_env_for_sb3
        from stable_baselines3.common.monitor import Monitor
        
        env = make_flight_env_for_sb3(render=False, seed=42)
        env = Monitor(env)
        
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=10000,
            learning_starts=50,
            batch_size=64,
            verbose=0,
            seed=42,
        )
        
        print("  Training for 100 steps...")
        model.learn(total_timesteps=100, progress_bar=False)
        print("  ✓ SAC training test passed")
        
        # Test prediction
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print(f"    Prediction test: action shape = {action.shape}")
        
        env.close()
        
    except Exception as e:
        print(f"  ✗ SAC training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✓ All smoke tests passed!")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    success = smoke_test()
    sys.exit(0 if success else 1)
