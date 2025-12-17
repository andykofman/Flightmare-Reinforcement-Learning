#!/usr/bin/env python3
"""
PLACEHOLDER - Performance benchmarking script.

Usage:
    python benchmark.py --model ./models/sac/best_model.zip
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Benchmark model performance")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--iterations", "-n", type=int, default=1000)

    args = parser.parse_args()

    print("Benchmarking not yet implemented")
    print("Basic evaluation available via evaluate.py")
    return 1


if __name__ == "__main__":
    exit(main())