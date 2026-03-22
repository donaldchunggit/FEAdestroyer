"""
Inference utilities for FEA Destroyer.
Handles model loading, mesh generation, and PyG data construction
without needing a pre-generated NPZ file.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from models.engineering_gnn import EngineeringGNN  # noqa: E402

# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
MATERIALS = {
    "Structural Steel":    {"E": 200e9,   "nu": 0.30, "yield_stress": 250e6,  "density": 7850},
    "High-Strength Steel": {"E": 210e9,   "nu": 0.28, "yield_stress": 550e6,  "density": 7850},
    "Aluminum 6061-T6":    {"E":  68.9e9, "nu": 0.33, "yield_stress": 276e6,  "density": 2700},
    "Aluminum 7075-T6":    {"E":  71.7e9, "nu": 0.33, "yield_stress": 503e6,  "density": 2810},
    "Titanium Grade 5":    {"E": 114e9,   "nu": 0.34, "yield_stress": 880e6,  "density": 4430},
    "Stainless Steel 304": {"E": 193e9,   "nu": 0.29, "yield_stress": 215e6,  "density": 8000},
}

# Default model search paths
_DEFAULT_CHECKPOINTS = [
    "best_model_correlation.pt",
    "best_model.pt",
]


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------
def make_mesh(length: float, width: float, height: float, n: int = 200):
    """
    Perturbed structured grid → Delaunay tetrahedral mesh.
    Returns (nodes [N,3], elements [T,4]).
    """
    ratio_x = width  / length
    ratio_y = height / length
    nz = max(6, round((n / max(ratio_x * ratio_y, 1e-4)) ** (1 / 3)))
    nx = max(3, round(nz * ratio_x))
    ny = max(3, round(nz * ratio_y))

    xs = np.linspace(-width  / 2,  width  / 2, nx)
    ys = np.linspace(-height / 2,  height / 2, ny)
    zs = np.linspace(0, length, nz)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)

    jitter = np.array([width / (nx * 6), height / (ny * 6), length / (nz * 6)])
    pts += np.random.default_rng(0).standard_normal(pts.shape) * jitter
    pts[:, 0] = np.clip(pts[:, 0], -width  / 2,  width  / 2)
    pts[:, 1] = np.clip(pts[:, 1], -height / 2,  height / 2)
    pts[:, 2] = np.clip(pts[:, 2], 0, length)

    tri = Delaunay(pts)
    return pts.astype(np.float32), tri.simplices.astype(np.int32)


# ---------------------------------------------------------------------------
# PyG data construction — mirrors data_loader.py exactly
# ---------------------------------------------------------------------------
def build_data(nodes: np.ndarray, elements: np.ndarray,
               force_vec: np.ndarray, E: float, nu: float) -> Data:
    """
    Build a PyG Data object from raw mesh + boundary conditions.
    Mirrors data_loader.load_single_npz() so the model sees
    the same feature format it was trained on.
    """
    nodes      = np.asarray(nodes,      dtype=np.float32)
    elements   = np.asarray(elements,   dtype=np.int64)
    force_vec  = np.asarray(force_vec,  dtype=np.float32).reshape(3)
    N          = len(nodes)

    z     = nodes[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())

    # Masks — identical thresholds to data_loader.py
    fixed      = (z < z_min + 0.1).astype(np.float32)
    fixed_mask = fixed.reshape(-1, 1)
    free_mask  = 1.0 - fixed_mask
    tip_flag   = (z > z_max - 0.1).astype(np.float32).reshape(-1, 1)
    dist_root  = ((z - z_min) / (z_max - z_min + 1e-8)).astype(np.float32).reshape(-1, 1)

    # Global conditioning [6]: force_dir(3), log10(|F|+1), log10(E), nu
    F_mag  = float(np.linalg.norm(force_vec) + 1e-8)
    f_dir  = force_vec / F_mag
    logF   = np.float32(np.log10(F_mag + 1.0))
    logE   = np.float32(np.log10(E + 1e-12))
    g      = np.array([f_dir[0], f_dir[1], f_dir[2], logF, logE, nu], dtype=np.float32)
    g_node = np.tile(g[None, :], (N, 1))

    # Node features [N, 12]
    x = np.concatenate([nodes, fixed_mask, dist_root, tip_flag, g_node], axis=1)

    # Edges from tetrahedra (vectorised, matching data_loader._build_edges_from_tets)
    pair_idx = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=np.int64)
    src = elements[:, pair_idx[:, 0]].ravel()
    dst = elements[:, pair_idx[:, 1]].ravel()
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    edge_index = np.unique(np.stack([all_src, all_dst], axis=1), axis=0).T.astype(np.int64)

    pos_t        = torch.from_numpy(nodes).float()
    edge_index_t = torch.from_numpy(edge_index).long()

    src_pos  = pos_t[edge_index_t[0]]
    dst_pos  = pos_t[edge_index_t[1]]
    edge_vec = dst_pos - src_pos
    edge_len = torch.norm(edge_vec, dim=1, keepdim=True)
    edge_dir = edge_vec / (edge_len + 1e-8)
    E_col    = torch.full_like(edge_len, float(E))
    nu_col   = torch.full_like(edge_len, float(nu))
    edge_attr = torch.cat([edge_len, edge_dir, E_col, nu_col], dim=1)  # [E, 6]

    mat = np.array([E, nu], dtype=np.float32)

    return Data(
        x            = torch.from_numpy(x).float(),
        pos          = pos_t,
        edge_index   = edge_index_t,
        edge_attr    = edge_attr,
        force_vector = torch.from_numpy(force_vec).float(),
        material_params = torch.from_numpy(mat).float(),
        fixed_mask   = torch.from_numpy(fixed_mask).float(),
        free_mask    = torch.from_numpy(free_mask).float(),
        num_nodes    = N,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def find_checkpoint() -> str | None:
    for path in _DEFAULT_CHECKPOINTS:
        if os.path.isfile(path):
            return path
    return None


def load_model(checkpoint_path: str = None):
    """
    Load EngineeringGNN from a checkpoint file.
    Returns (model, info_dict) where info_dict contains training metadata.
    """
    if checkpoint_path is None:
        checkpoint_path = find_checkpoint()
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        return None, {"error": "No trained model found. Run training first."}

    ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    model = EngineeringGNN(
        node_dim   = 12,
        edge_dim   = 6,
        hidden_dim = config.get("HIDDEN_DIM", 256),
        num_layers = config.get("NUM_LAYERS", 5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    info = {
        "epoch":       ckpt.get("epoch",       "?"),
        "correlation": ckpt.get("correlation", None),
        "mae_mm":      ckpt.get("mae_mm",      None),
        "path":        checkpoint_path,
    }
    return model, info


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model: EngineeringGNN, data: Data) -> dict:
    """
    Run model forward pass and return results in engineering units.
    Returns dict with numpy arrays in mm / MPa / dimensionless.
    """
    out = model(data)

    disp_m   = out["displacement"].cpu().numpy()          # [N, 3]  metres
    stress_pa = out["stress"].cpu().numpy().reshape(-1)   # [N]     Pa
    sf        = out["safety_factor"].cpu().numpy().reshape(-1)  # [N]

    disp_mm        = disp_m * 1000.0
    disp_mag_mm    = np.linalg.norm(disp_mm, axis=1)
    stress_mpa     = stress_pa / 1e6

    return {
        "disp_mm":      disp_mm,          # [N, 3]
        "disp_mag_mm":  disp_mag_mm,      # [N]
        "stress_mpa":   stress_mpa,       # [N]
        "safety_factor": sf,              # [N]
        "max_disp_mm":  float(disp_mag_mm.max()),
        "max_stress_mpa": float(stress_mpa.max()),
        "min_sf":       float(np.clip(sf, 0, 100).min()),
    }
