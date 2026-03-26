"""
Direct FEM dataset generator.
Uses multiprocessing.Pool with an initializer so worker processes
can find all modules on Windows (spawn method).

Usage:
    python generate.py --samples 100
    python generate.py --samples 100 --generator fast
    python generate.py --samples 100 --workers 12
"""
import sys
import os
import argparse
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Set PYTHONPATH so spawned worker processes inherit it
_extra = [
    str(ROOT / "generator_fem"),
    str(ROOT / "generator_adv"),
    str(ROOT / "Cantilever_Generator"),
    str(ROOT),
]
_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(_extra) + (os.pathsep + _existing if _existing else "")
for _p in _extra:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _worker_init():
    """Runs in each worker process before any tasks — sets up sys.path."""
    for p in _extra:
        if p not in sys.path:
            sys.path.insert(0, p)


def _run_fast(num_samples, output_dir):
    from adv_generator import generate_advanced_dataset
    generate_advanced_dataset(num_samples=num_samples, output_dir=output_dir)


def _run_fem(num_samples, output_dir, n_workers):
    import multiprocessing as mp
    import numpy as np
    import pandas as pd
    from fem_generator import _worker, MATERIALS

    n_train = int(num_samples * 0.8)
    train_dir = os.path.join(output_dir, "train")
    val_dir   = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    rng        = np.random.default_rng(42)
    n_buf      = num_samples * 2
    mat_keys   = list(MATERIALS.keys())
    seeds      = rng.integers(0, 2**31, size=n_buf)
    mats       = rng.choice(mat_keys, size=n_buf)
    lengths    = rng.uniform(0.5, 2.0, size=n_buf)
    wfracs     = rng.uniform(0.06, 0.18, size=n_buf)
    hfracs     = rng.uniform(0.06, 0.18, size=n_buf)
    fmags      = rng.uniform(5_000, 50_000, size=n_buf)
    ax         = rng.uniform(-20, 20, size=n_buf) * np.pi / 180
    ay         = rng.uniform(-20, 20, size=n_buf) * np.pi / 180
    n_targets  = rng.integers(60, 90, size=n_buf)

    def _args(idx, sid, split_dir):
        L = lengths[idx]; w = wfracs[idx]*L; h = hfracs[idx]*L
        F = fmags[idx]
        force = np.array([F*np.sin(ay[idx]), F*np.sin(ax[idx])*np.cos(ay[idx]), 0.0])
        return (sid, split_dir, int(n_targets[idx]), int(seeds[idx]), mats[idx], L, w, h, force)

    print(f"Generating {num_samples} FEM samples with {n_workers} workers ...")

    metadata      = []
    failed        = 0
    completed     = 0
    param_idx     = 0
    sample_ctr    = 0
    t_start       = time.time()

    with mp.Pool(processes=n_workers, initializer=_worker_init) as pool:
        pending = {}

        def _submit():
            nonlocal sample_ctr, param_idx
            if sample_ctr < num_samples and param_idx < n_buf:
                split = train_dir if sample_ctr < n_train else val_dir
                sid   = f"sample_{sample_ctr:04d}"
                res   = pool.apply_async(_worker, (_args(param_idx, sid, split),))
                pending[res] = (sample_ctr, param_idx)
                sample_ctr += 1
                param_idx  += 1

        # Fill pool
        for _ in range(min(n_workers * 2, num_samples)):
            _submit()

        while completed < num_samples and (pending or sample_ctr < num_samples):
            ready = [f for f in list(pending) if f.ready()]

            if not ready:
                time.sleep(0.05)
                continue

            for fut in ready:
                s_idx, _ = pending.pop(fut)
                try:
                    info, err = fut.get()
                except Exception as exc:
                    info, err = None, str(exc)

                if info is None:
                    failed += 1
                    print(f"  [FAIL] {err}")
                    if completed + len(pending) < num_samples and param_idx < n_buf:
                        _submit()
                else:
                    metadata.append(info)
                    completed += 1
                    elapsed = time.time() - t_start
                    rate    = elapsed / completed
                    remain  = rate * (num_samples - completed)
                    print(f"  {completed}/{num_samples}  {rate:.1f}s/sample  ~{remain:.0f}s left"
                          + (f"  ({failed} failed)" if failed else ""))
                    if completed + len(pending) < num_samples:
                        _submit()

    if not metadata:
        print("No samples generated. Check errors above.")
        return

    import pandas as pd
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    total = time.time() - t_start
    print(f"\nDone — {len(metadata)} samples in {total/60:.1f} min  ({total/len(metadata):.1f}s/sample effective)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",   type=int, default=100)
    parser.add_argument("--generator", choices=["fem", "fast"], default="fem")
    parser.add_argument("--workers",   type=int, default=None)
    args = parser.parse_args()

    n_workers = args.workers or max(1, os.cpu_count() - 1)

    if args.generator == "fast":
        out = str(ROOT / "generator_adv" / "advanced_dataset")
        if os.path.exists(out):
            shutil.rmtree(out)
        _run_fast(args.samples, out)
    else:
        out = str(ROOT / "generator_fem" / "fem_dataset")
        if os.path.exists(out):
            shutil.rmtree(out)
        _run_fem(args.samples, out, n_workers)
