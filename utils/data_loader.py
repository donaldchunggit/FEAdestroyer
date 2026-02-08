"""
Data loading utilities for NPZ -> PyTorch Geometric Data.

Fixes & upgrades:
- Ensures stress_true is always [N,1] and u_true is always [N,3] (batch-safe)
- Creates explicit fixed_mask/free_mask (so training masking is correct)
- Injects global conditioning into node features:
    force_dir(3), log10(|F|+1), log10(E), nu  -> broadcast to all nodes
- Keeps edge_attr at 6 dims for compatibility with your EngineeringGNN:
    [edge_length, dx_norm, dy_norm, dz_norm, E, nu]
"""

import os
import glob
from typing import List, Optional
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


# -----------------------------
# Utility helpers
# -----------------------------
def _as_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _as_long(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


def _ensure_2d_col(x: np.ndarray) -> np.ndarray:
    """Ensure shape [N,1] for per-node scalar arrays."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    # If weird shape, try flatten then col
    return x.reshape(-1, 1)


def _global_features(force_vec: np.ndarray, material_params: np.ndarray) -> np.ndarray:
    """
    Build global conditioning vector g of shape (6,):
      [Fx_dir, Fy_dir, Fz_dir, log10(|F|+1), log10(E), nu]
    """
    force_vec = _as_float32(force_vec).reshape(3,)
    E = float(_as_float32(material_params)[0])
    nu = float(_as_float32(material_params)[1])

    Fmag = float(np.linalg.norm(force_vec) + 1e-8)
    f_dir = force_vec / Fmag  # (3,)

    logF = np.log10(Fmag + 1.0).astype(np.float32)
    logE = np.log10(E + 1e-12).astype(np.float32)

    g = np.array([f_dir[0], f_dir[1], f_dir[2], logF, logE, nu], dtype=np.float32)
    return g


def _build_edges_from_tets(tets: np.ndarray) -> np.ndarray:
    """
    tets: [T,4] int
    returns directed edge_index as np array [2, E]
    """
    # Each tet has 6 undirected edges; we add both directions.
    # Use python set to dedupe.
    edges = set()
    for a, b, c, d in tets:
        pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        for i, j in pairs:
            edges.add((int(i), int(j)))
            edges.add((int(j), int(i)))

    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    edge_index = np.array(list(edges), dtype=np.int64).T  # [2, E]
    return edge_index


# -----------------------------
# Main loader
# -----------------------------
def load_single_npz(npz_path: str) -> Data:
    """
    Load a single NPZ file and convert to PyG Data object.

    Expected keys:
        node_coords:      [N, 3]
        connectivity:     [T, 4]  tetrahedra
        input_force:      [3,]
        material_params:  [2,]    [E (Pa), nu]
        node_stresses:    [N] or [N,1]
        node_disp:        [N,3]
    """
    data = np.load(npz_path, allow_pickle=True)

    # --- Core arrays
    coords = _as_float32(data["node_coords"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{npz_path}: node_coords must be [N,3], got {coords.shape}")
    N = coords.shape[0]

    tets = _as_long(data["connectivity"])
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"{npz_path}: connectivity must be [T,4], got {tets.shape}")

    force_vec = _as_float32(data["input_force"]).reshape(3,)
    mat = _as_float32(data["material_params"]).reshape(2,)
    E = float(mat[0])
    nu = float(mat[1])

    u_true = _as_float32(data["node_disp"])
    if u_true.shape != (N, 3):
        raise ValueError(f"{npz_path}: node_disp must be [N,3], got {u_true.shape}")

    stress_true = _as_float32(data["node_stresses"])
    stress_true = _ensure_2d_col(stress_true)  # [N,1]
    if stress_true.shape[0] != N:
        raise ValueError(f"{npz_path}: node_stresses must have N rows, got {stress_true.shape}")

    # --- Fixed boundary mask (cantilever clamp at z_min)
    z = coords[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())

    # This threshold is in your mesh units; keep consistent with your solver generation.
    fixed = (z < (z_min + 0.1)).astype(np.float32)  # [N]
    fixed_mask = _ensure_2d_col(fixed)              # [N,1]
    free_mask = 1.0 - fixed_mask                    # [N,1]

    # --- Tip mask / nodal forces (distributed)
    tip = (z > (z_max - 0.1)).astype(np.float32)    # [N]
    tip_count = float(tip.sum())
    f_nodes = np.zeros((N, 3), dtype=np.float32)
    if tip_count > 0:
        f_nodes[tip.astype(bool)] = force_vec / tip_count

    # --- Boundary condition tensor (3 dof)
    bc = np.zeros((N, 3), dtype=np.float32)
    bc[fixed.astype(bool), :] = 1.0

    # --- Global conditioning features (broadcast to nodes)
    g = _global_features(force_vec, mat)            # (6,)
    g_node = np.tile(g[None, :], (N, 1)).astype(np.float32)  # [N,6]

    # --- Distance-to-root feature (useful)
    dist_to_root = ((z - z_min) / (z_max - z_min + 1e-8)).astype(np.float32)
    dist_to_root = _ensure_2d_col(dist_to_root)     # [N,1]

    # --- Tip flag (free end)
    tip_flag = (z > (z_max - 0.1)).astype(np.float32)
    tip_flag = _ensure_2d_col(tip_flag)  # [N,1]

    # --- Node features
    # Keep your original useful parts AND add conditioning:
    # [x,y,z, fixed_mask, dist_to_root, force_dir(3), logF, logE, nu]
    x = np.concatenate(
        [
            coords,                 # 3
            fixed_mask,             # 1
            dist_to_root,           # 1
            tip_flag,              # 1
            g_node,                 # 6
        ],
        axis=1,
    ).astype(np.float32)            # total 3+1+1+1+6 = 12 dims

    # --- Edges
    edge_index = _build_edges_from_tets(tets)       # [2,E]
    edge_index_t = torch.from_numpy(edge_index).long()

    pos_t = torch.from_numpy(coords).float()

    # Edge attr: [length, dx_norm, dy_norm, dz_norm, E, nu]
    if edge_index.shape[1] == 0:
        edge_attr = torch.zeros((0, 6), dtype=torch.float32)
    else:
        src = pos_t[edge_index_t[0]]
        dst = pos_t[edge_index_t[1]]
        edge_vec = dst - src
        edge_len = torch.norm(edge_vec, dim=1, keepdim=True)  # [E,1]
        edge_dir = edge_vec / (edge_len + 1e-8)               # [E,3]
        E_col = torch.full_like(edge_len, float(E))
        nu_col = torch.full_like(edge_len, float(nu))
        edge_attr = torch.cat([edge_len, edge_dir, E_col, nu_col], dim=1)  # [E,6]

    # --- Build Data
    pyg = Data(
        x=torch.from_numpy(x).float(),                    # [N,11]
        pos=pos_t,                                        # [N,3]
        edge_index=edge_index_t,                           # [2,E]
        edge_attr=edge_attr,                               # [E,6]

        # Extra fields used by training / debugging:
        u_true=torch.from_numpy(u_true).float(),           # [N,3]
        stress_true=torch.from_numpy(stress_true).float(), # [N,1]

        fixed_mask=torch.from_numpy(fixed_mask).float(),   # [N,1]
        free_mask=torch.from_numpy(free_mask).float(),     # [N,1]
        node_tip=torch.from_numpy(tip_flag).float(),      # [N,1]

        f=torch.from_numpy(f_nodes).float(),               # [N,3]
        bc=torch.from_numpy(bc).float(),                   # [N,3]

        tetra_connectivity=torch.from_numpy(tets).long(),  # [T,4]
        material_params=torch.from_numpy(mat).float(),     # [2]
        force_vector=torch.from_numpy(force_vec).float(),  # [3]
        num_nodes=int(N),
    )

    return pyg


class NPZDataset(Dataset):
    """PyG Dataset for NPZ files."""
    def __init__(self, npz_dir: str, max_samples: Optional[int] = None):
        self.npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if max_samples is not None:
            self.npz_files = self.npz_files[:max_samples]
        super().__init__()

    def len(self):
        return len(self.npz_files)

    def get(self, idx):
        return load_single_npz(self.npz_files[idx])


def load_npz_dataset(npz_dir: str, max_samples: Optional[int] = None) -> List[Data]:
    """Load all NPZ files in a directory into a list of Data objects."""
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if max_samples is not None:
        npz_files = npz_files[:max_samples]

    print(f"Loading {len(npz_files)} samples from {npz_dir}")

    out: List[Data] = []
    for p in tqdm(npz_files, desc="Loading data"):
        try:
            out.append(load_single_npz(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")

    return out
