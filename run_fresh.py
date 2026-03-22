"""
Single command: purge old dataset → generate fresh data → train.

Usage:
    python run_fresh.py              # default 100 samples
    python run_fresh.py --samples 200
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path("generator_adv/advanced_dataset")


def purge_dataset():
    if DATASET_DIR.exists():
        print(f"Purging old dataset at '{DATASET_DIR}' ...")
        shutil.rmtree(DATASET_DIR)
        print("Done.\n")
    else:
        print("No existing dataset found, skipping purge.\n")


def generate_dataset(num_samples: int):
    print(f"Generating {num_samples} fresh samples ...")
    result = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, 'generator_adv'); "
         f"from adv_generator import generate_advanced_dataset; "
         f"generate_advanced_dataset(num_samples={num_samples}, output_dir='advanced_dataset')"],
        check=True,
    )
    print()


def run_training():
    print("Starting training ...")
    subprocess.run([sys.executable, "train.py"], check=True)


def main():
    parser = argparse.ArgumentParser(description="Fresh data + train pipeline")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to generate (default: 100)")
    args = parser.parse_args()

    purge_dataset()
    generate_dataset(args.samples)
    run_training()


if __name__ == "__main__":
    main()
