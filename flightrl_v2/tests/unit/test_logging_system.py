#! /usr/bin/env python3

"""
Test the Unified Logging System

Quick test script to verify the logging system is working correctly.

Usage:  
    python test_logging_system.py


"""

import sys
import os
from pathlib import Path

#Add parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Test that all imports work"""
    print("[TEST] Testing imports...")
    try:
        from flightrl_modern.utils import LogManager, resolve_model_alias, list_runs, print_runs_table
        print("[PASS] All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_log_manager():
    """ Test LogManager creation"""
    print("\n[TEST] Testing LogManager...")
    try: 
        from flightrl_v2.utils import LogManager
        
        # Create a test logger
        logger = LogManager(
            project="test_project",
            algorithm="Test",
            config={"test_param", 123},
            tags=["test", 'validation'],
            notes="Testing logging system",
            run_name="test_run"
        )

        print(f"[PASS] LogManager created")
        print(f"       Run directory: {logger.run_dir}")
        print(f"       Checkpoints: {logger.checkpoints_dir}")
        print(f"       TensorBoard: {logger.tensorboard_dir}")
        
        # Test metadata
        logger.log_metrics({"test_metric": 0.95})
        logger.finalize(status="completed")
        
        print("[PASS] Metadata logging successful")
        
        # Cleanup test directory
        import shutil
        test_dir = Path("logs/test_project")
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("[INFO] Test directory cleaned up")
        
        return True

    except Exception as e:
        print(f"❌ LogManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_list_runs():
    """Test list_runs functionality"""
    print("\n[TEST] Testing list_runs...")
    try:
        from flightrl_v2.utils import list_runs, print_runs_table

        # check if any projects exist

        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("[WARN] No logs directory found (this is OK for first run)")
            return True


        projects = [d.name for d in logs_dir.iterdir() if d.is_dir]
        if not projects:
            print("[WARN] No projects found in logs (this is OK for first run)")
            return True


        # list runs for first project
        project = projects[0]
        runs = list_runs(project, base_dir="logs")

        if runs:
            print(f"[INFO] Found {len(runs)} runs in '{project}'")
            print("\nSample run data:")
            print_runs_table(runs[:3])  # Show first 3 runs
        else:
            print(f"[WARN] No runs found in '{project}' (this is OK for first run)")
        
        return True

    except Exception as e:
        print(f"❌ list_runs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alias_resolution():
    """Test alias resolution"""
    print("\n[TEST] Testing alias resolution...")
    try:
        from flightrl_modern.utils import resolve_model_alias
        
        # Check if any projects have aliases
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("[WARN] No logs directory (alias test skipped)")
            return True
        
        projects = [d.name for d in logs_dir.iterdir() if d.is_dir()]
        
        for project in projects:
            # Check for 'latest' link
            latest_link = logs_dir / project / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                try:
                    path = resolve_model_alias("@latest", project=project)
                    print(f"[PASS] Resolved @latest for '{project}'")
                    print(f"       Path: {path}")
                    return True
                except Exception as e:
                    print(f"⚠️  Could not resolve @latest: {e}")
        
        print("[WARN] No projects with 'latest' link found (this is OK for first run)")
        return True
        
    except Exception as e:
        print(f"❌ Alias resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("UNIFIED LOGGING SYSTEM - TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("LogManager", test_log_manager()))
    results.append(("List Runs", test_list_runs()))
    results.append(("Alias Resolution", test_alias_resolution()))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name:<20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*70)
    
    if all_passed:
        print("\n[SUCCESS] ALL TESTS PASSED")
        print("\nThe logging system is ready to use.")
        print("\nNext steps:")
        print("  1. Start training: python target_seeking/examples/train_sac_velocity_validated.py --total-steps 10000")
        print("  2. List runs: python list_runs.py target_seeking")
        print("  3. Evaluate: python target_seeking/examples/evaluate_target_seeking.py --model @latest --episodes 1")
        print("\nSee docs/LOGGING_SYSTEM.md for full documentation.")
    else:
        print("\n[WARN] SOME TESTS FAILED")
        print("\nPlease check the error messages above.")
        print("You may need to install missing dependencies or fix import paths.")
    
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

        