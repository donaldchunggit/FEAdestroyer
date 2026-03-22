"""
Single command: purge old dataset → generate fresh data → train.

Usage:
    python run_fresh.py                        # auto-detects GPU/CPU, uses FEM generator
    python run_fresh.py --samples 500          # override sample count
    python run_fresh.py --generator fast       # use fast analytical generator (no FEM)
    python run_fresh.py --generator fem        # use real FEM generator (default)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import torch
    _HAS_NVIDIA = torch.cuda.is_available()
except ImportError:
    _HAS_NVIDIA = False

DEFAULT_SAMPLES = 500 if _HAS_NVIDIA else 100

# Dataset dirs for each generator
_DIRS = {
    "fem":  Path("generator_fem/fem_dataset"),
    "fast": Path("generator_adv/advanced_dataset"),
}
_STRAY = Path("advanced_dataset")  # leftover from early bug


def purge_dataset(generator: str):
    purged = False
    for d in (_DIRS[generator], _STRAY):
        if d.exists():
            print(f"Purging old dataset at '{d}' ...")
            shutil.rmtree(d)
            purged = True
    print("Done.\n" if purged else "No existing dataset found, skipping purge.\n")


def generate_dataset(num_samples: int, generator: str):
    device = "NVIDIA GPU" if _HAS_NVIDIA else "CPU"
    print(f"Device: {device} — generating {num_samples} samples "
          f"({'FEM' if generator == 'fem' else 'analytical'} generator)\n")

    if generator == "fem":
        output_dir = "generator_fem/fem_dataset"
        code = (
            "import sys; sys.path.insert(0, 'generator_fem'); "
            f"from fem_generator import generate_fem_dataset; "
            f"generate_fem_dataset(num_samples={num_samples}, output_dir='{output_dir}')"
        )
    else:
        output_dir = "generator_adv/advanced_dataset"
        code = (
            "import sys; sys.path.insert(0, 'generator_adv'); "
            f"from adv_generator import generate_advanced_dataset; "
            f"generate_advanced_dataset(num_samples={num_samples}, output_dir='{output_dir}')"
        )

    subprocess.run([sys.executable, "-c", code], check=True)
    print()


def run_training():
    print("Starting training ...")
    subprocess.run([sys.executable, "train.py"], check=True)


def main():
    parser = argparse.ArgumentParser(description="Fresh data + train pipeline")
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES,
        help=f"Number of samples to generate (auto: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--generator", choices=["fem", "fast"], default="fem",
        help="fem = real FEM solver (better quality, slower);  fast = analytical (instant)",
    )
    args = parser.parse_args()

    purge_dataset(args.generator)
    generate_dataset(args.samples, args.generator)
    run_training()


if __name__ == "__main__":
    main()
