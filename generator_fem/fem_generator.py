"""
FEM-backed dataset generator.

Replaces the analytical beam-theory approximations in adv_generator.py with
real linear-elastic FEM solves (physics_engine.py).  Ground-truth displacements
and Von Mises stress come from solving KU = F on a tetrahedral mesh, giving
noise-free physics data that can push Pearson R above 0.95.

Performance optimisations (without touching physics_engine.py):
  1. Multiprocessing — each sample is independent; all CPU cores used in parallel.
  2. Smaller meshes — 120-180 nodes instead of 200-400; the FEM assembly is O(n_elems)
     in pure Python, so halving node count roughly halves generation time with minimal
     accuracy loss on simple cantilever geometry.
  3. Degenerate-element filtering — Delaunay always produces tiny slivered tets at
     convex-hull edges; removing them before the solve cuts ~20-30 % of elements and
     improves solver conditioning.
"""

import os
import sys
import shutil
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import meshio
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

warnings.filterwarnings("ignore")

# Make physics_engine importable from Cantilever_Generator/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "Cantilever_Generator"))

from physics_engine import PhysicsEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Material database
# ---------------------------------------------------------------------------
MATERIALS = {
    "Structural Steel":    {"E": 200e9,   "nu": 0.30, "yield_stress": 250e6},
    "High-Strength Steel": {"E": 210e9,   "nu": 0.28, "yield_stress": 550e6},
    "Aluminum 6061-T6":    {"E":  68.9e9, "nu": 0.33, "yield_stress": 276e6},
    "Aluminum 7075-T6":    {"E":  71.7e9, "nu": 0.33, "yield_stress": 503e6},
    "Titanium Grade 5":    {"E": 114e9,   "nu": 0.34, "yield_stress": 880e6},
    "Stainless Steel 304": {"E": 193e9,   "nu": 0.29, "yield_stress": 215e6},
}
_MAT_KEYS = list(MATERIALS.keys())


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def _make_beam_mesh(length: float, width: float, height: float, n_target: int):
    """
    Perturbed structured grid → scipy Delaunay tetrahedral mesh.

    Smaller n_target (120-180) is intentional: the physics_engine FEM assembly
    runs in pure Python O(n_elements * 144), so keeping element count low is
    the single biggest lever we have without modifying physics_engine.py.
    """
    ratio_x = width  / length
    ratio_y = height / length
    nz = max(5, round((n_target / max(ratio_x * ratio_y, 1e-4)) ** (1 / 3)))
    nx = max(3, round(nz * ratio_x))
    ny = max(3, round(nz * ratio_y))

    xs = np.linspace(-width  / 2,  width  / 2, nx)
    ys = np.linspace(-height / 2,  height / 2, ny)
    zs = np.linspace(0, length, nz)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)

    jitter = np.array([width / (nx * 6), height / (ny * 6), length / (nz * 6)])
    pts += np.random.randn(*pts.shape) * jitter
    pts[:, 0] = np.clip(pts[:, 0], -width  / 2, width  / 2)
    pts[:, 1] = np.clip(pts[:, 1], -height / 2, height / 2)
    pts[:, 2] = np.clip(pts[:, 2], 0, length)

    tri = Delaunay(pts)
    return pts, tri.simplices.astype(np.int32)


def _filter_degenerate(nodes: np.ndarray, elements: np.ndarray,
                        min_vol_ratio: float = 5e-4) -> np.ndarray:
    """
    Vectorised removal of tiny slivered tetrahedra.

    Delaunay on a convex body always produces slivers near the boundary.
    Removing them before the solve:
      - Cuts 20-35 % of elements → proportionally faster FEM assembly
      - Improves global stiffness matrix conditioning → fewer solver warnings
    """
    p = nodes[elements]                   # (T, 4, 3)
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 2] - p[:, 0]
    v3 = p[:, 3] - p[:, 0]
    cross = np.cross(v2, v3)              # (T, 3)
    vols = np.abs(np.einsum("ij,ij->i", v1, cross)) / 6.0   # (T,)
    threshold = vols.max() * min_vol_ratio
    return elements[vols > threshold]


# ---------------------------------------------------------------------------
# Worker — must be module-level for multiprocessing spawn (Windows)
# ---------------------------------------------------------------------------
def _worker(args: tuple):
    """Generates and saves a single FEM sample. Designed for process pool use."""
    sample_id, output_dir, n_target, seed, mat_name, length, width, height, force = args

    # Re-seed numpy in this worker process
    np.random.seed(seed)

    mat = MATERIALS[mat_name]
    nodes, elements = _make_beam_mesh(length, width, height, n_target)
    elements = _filter_degenerate(nodes, elements)

    if len(elements) < 20:
        return None, "Too few elements after filtering"

    with tempfile.TemporaryDirectory() as tmpdir:
        vtk_path = os.path.join(tmpdir, "mesh.vtk")
        meshio.write(vtk_path, meshio.Mesh(points=nodes, cells=[("tetra", elements)]))

        engine = PhysicsEngine(mat["E"], mat["nu"], assume_E_units="pa")
        try:
            nodes_out, disp, stress_vm, elems_out, fixed_idx, _ = engine.solve(
                vtk_path, force, clamp_fraction=0.05, load_fraction=0.05,
            )
        except Exception as exc:
            return None, str(exc)

    if np.any(np.isnan(disp)) or np.any(np.isnan(stress_vm)):
        return None, "NaN in FEM output"
    max_disp = float(np.abs(disp).max())
    if max_disp > 10.0:
        return None, f"Implausible deflection {max_disp:.2f} m"

    os.makedirs(output_dir, exist_ok=True)
    engine.export_npz(
        os.path.join(output_dir, f"{sample_id}.npz"),
        nodes_out, elems_out, stress_vm, disp, force, fixed_idx,
    )

    return {
        "sample_id":      sample_id,
        "material":       mat_name,
        "length_m":       round(length, 4),
        "width_m":        round(width,  4),
        "height_m":       round(height, 4),
        "force_kN":       round(float(np.linalg.norm(force)) / 1000, 3),
        "nodes":          len(nodes_out),
        "elements":       len(elems_out),
        "max_stress_MPa": round(float(stress_vm.max() / 1e6), 2),
        "max_disp_mm":    round(max_disp * 1000, 4),
    }, None


def _optimal_workers() -> int:
    """
    Leave 2 logical cores free for the OS and scipy's BLAS threads.
    Cap at 8 to avoid memory pressure from many simultaneous meshio + scipy loads.
    """
    return max(1, min(os.cpu_count() - 2, 8))


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_fem_dataset(num_samples: int = 100,
                         output_dir: str = "fem_dataset",
                         train_split: float = 0.8,
                         seed: int = 42,
                         n_workers: int = None):
    """
    Generate a complete FEM-solved dataset with train/val split.
    Drop-in replacement for generate_advanced_dataset() in adv_generator.py.
    """
    if n_workers is None:
        n_workers = _optimal_workers()

    rng = np.random.default_rng(seed)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    n_train   = int(num_samples * train_split)
    train_dir = os.path.join(output_dir, "train")
    val_dir   = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    # Pre-generate all random parameters so the main process controls reproducibility
    worker_seeds  = rng.integers(0, 2**31, size=num_samples * 2)
    mat_choices   = rng.choice(_MAT_KEYS,             size=num_samples * 2)
    lengths       = rng.uniform(0.5, 2.0,             size=num_samples * 2)
    width_fracs   = rng.uniform(0.04, 0.18,           size=num_samples * 2)
    height_fracs  = rng.uniform(0.04, 0.18,           size=num_samples * 2)
    f_mags        = rng.uniform(5_000, 50_000,        size=num_samples * 2)
    axes_x        = rng.uniform(-20, 20,              size=num_samples * 2) * np.pi / 180
    axes_y        = rng.uniform(-20, 20,              size=num_samples * 2) * np.pi / 180
    n_targets     = rng.integers(120, 180,            size=num_samples * 2)

    def _make_args(idx, sample_id, split_dir):
        L = lengths[idx]
        w = width_fracs[idx]  * L
        h = height_fracs[idx] * L
        F = f_mags[idx]
        ax, ay = axes_x[idx], axes_y[idx]
        force = np.array([F * np.sin(ay), F * np.sin(ax) * np.cos(ay), 0.0])
        return (sample_id, split_dir, int(n_targets[idx]),
                int(worker_seeds[idx]), mat_choices[idx], L, w, h, force)

    print(f"Generating {num_samples} FEM samples using {n_workers} parallel workers ...")
    print(f"  Mesh size: 120-180 nodes per sample  |  degenerate-element filtering ON")

    metadata   = []
    failed     = 0
    completed  = 0
    param_idx  = 0
    t_start    = time.time()

    # Submit initial batch
    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        # Fill pool
        submitted = 0
        sample_counter = 0
        while submitted < min(num_samples * 2, num_samples + n_workers * 2) and param_idx < num_samples * 2:
            if sample_counter < num_samples:
                split_dir = train_dir if sample_counter < n_train else val_dir
                sid = f"sample_{sample_counter:04d}"
                args = _make_args(param_idx, sid, split_dir)
                fut = pool.submit(_worker, args)
                futures[fut] = (sample_counter, param_idx)
                sample_counter += 1
                param_idx += 1
                submitted += 1

        for fut in as_completed(futures):
            s_idx, p_idx = futures[fut]
            try:
                info, err = fut.result()
            except Exception as exc:
                info, err = None, str(exc)

            if info is None:
                failed += 1
                # Re-submit with next param slot if we still need samples
                if completed + len([f for f in futures if not f.done()]) < num_samples and param_idx < num_samples * 2:
                    new_sid   = f"sample_{sample_counter:04d}"
                    split_dir = train_dir if sample_counter < n_train else val_dir
                    new_args  = _make_args(param_idx, new_sid, split_dir)
                    new_fut   = pool.submit(_worker, new_args)
                    futures[new_fut] = (sample_counter, param_idx)
                    sample_counter += 1
                    param_idx += 1
            else:
                metadata.append(info)
                completed += 1
                if completed % 10 == 0:
                    elapsed = time.time() - t_start
                    per_s   = elapsed / completed
                    remain  = per_s * (num_samples - completed)
                    print(f"  {completed}/{num_samples}  "
                          f"({per_s:.1f}s/sample  ~{remain/60:.1f} min remaining"
                          f"{f'  {failed} failed' if failed else ''})")

            if completed >= num_samples:
                # Cancel any still-pending futures
                for f in futures:
                    f.cancel()
                break

    if failed:
        print(f"  Note: {failed} samples skipped (solver errors / bad geometry).")

    if not metadata:
        print("No samples generated successfully.")
        return None

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    n_val = len(metadata) - n_train
    elapsed = time.time() - t_start
    print(f"\nFEM dataset complete in {elapsed/60:.1f} min  "
          f"({elapsed/len(metadata):.1f}s/sample effective with {n_workers} workers)")
    print(f"  Train: {n_train}  Val: {max(n_val, 0)}")
    print(f"  Materials: {df['material'].nunique()} unique")
    print(f"  Stress: {df['max_stress_MPa'].min():.1f} – {df['max_stress_MPa'].max():.1f} MPa")
    print(f"  Disp:   {df['max_disp_mm'].min():.3f} – {df['max_disp_mm'].max():.3f} mm")

    return df


if __name__ == "__main__":
    generate_fem_dataset(num_samples=10, output_dir="fem_dataset")
