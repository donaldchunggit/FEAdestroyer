"""
Data loading utilities for NPZ to PyG conversion.
"""

import os
import glob
from typing import List, Optional
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

def load_single_npz(npz_path: str) -> Data:
    """
    Load a single NPZ file and convert to PyG Data object.
    
    Expected keys (from your physics_engine.py):
        node_coords: [N, 3] - nodal coordinates
        connectivity: [E, 4] - tetrahedron connectivity
        input_force: [3,] - applied force vector
        material_params: [2,] - [E, nu]
        node_stresses: [N, 1] - von Mises stress
        node_disp: [N, 3] - nodal displacements
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # Node coordinates
        pos = torch.tensor(data['node_coords'], dtype=torch.float32)
        num_nodes = pos.shape[0]
        
        # Node features: [x, y, z, is_boundary, distance_to_root]
        z_min = pos[:, 2].min()
        z_max = pos[:, 2].max()
        
        # Boundary flag (clamped at z_min)
        boundary = (pos[:, 2] < z_min + 0.1).float().unsqueeze(1)
        
        # Normalized distance from clamped end
        dist_to_root = (pos[:, 2] - z_min) / (z_max - z_min + 1e-8)
        dist_to_root = dist_to_root.unsqueeze(1)
        
        # Node features
        x = torch.cat([pos, boundary, dist_to_root], dim=1)  # [N, 5]
        
        # Create edges from tetrahedral connectivity
        connectivity = torch.tensor(data['connectivity'], dtype=torch.long)
        
        # Build edge list (undirected, both directions)
        edges = []
        for tet in connectivity:
            nodes = tet.tolist()
            # Create all 6 edges per tetrahedron
            edges.extend([(nodes[0], nodes[1]), (nodes[1], nodes[0])])
            edges.extend([(nodes[0], nodes[2]), (nodes[2], nodes[0])])
            edges.extend([(nodes[0], nodes[3]), (nodes[3], nodes[0])])
            edges.extend([(nodes[1], nodes[2]), (nodes[2], nodes[1])])
            edges.extend([(nodes[1], nodes[3]), (nodes[3], nodes[1])])
            edges.extend([(nodes[2], nodes[3]), (nodes[3], nodes[2])])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Remove duplicate edges (optional, for efficiency)
        edge_index = torch.unique(edge_index, dim=1)
        
        # Edge features: [distance, dx, dy, dz, material_E, material_nu]
        src_pos = pos[edge_index[0]]
        dst_pos = pos[edge_index[1]]
        edge_vec = dst_pos - src_pos
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
        
        # Material properties
        E = torch.tensor(data['material_params'][0], dtype=torch.float32)
        nu = torch.tensor(data['material_params'][1], dtype=torch.float32)
        
        edge_attr = torch.cat([
            edge_length,  # [E, 1]
            edge_vec / (edge_length + 1e-8),  # [E, 3] normalized direction
            torch.ones_like(edge_length) * E,  # [E, 1]
            torch.ones_like(edge_length) * nu  # [E, 1]
        ], dim=1)  # [E, 6]
        
        # External forces (distributed to tip nodes)
        force_vector = torch.tensor(data['input_force'], dtype=torch.float32)
        tip_mask = (pos[:, 2] > z_max - 0.1).float()
        num_tip_nodes = tip_mask.sum().item()
        
        if num_tip_nodes > 0:
            force_per_node = force_vector / num_tip_nodes
            f = torch.zeros_like(pos)
            f[tip_mask.bool()] = force_per_node
        else:
            f = torch.zeros_like(pos)
        
        # Boundary conditions (3 DOF per node)
        bc = torch.zeros(num_nodes, 3, dtype=torch.float32)
        bc[boundary.squeeze().bool(), :] = 1.0  # Fixed at clamped end
        
        # Ground truth labels
        u_true = torch.tensor(data['node_disp'], dtype=torch.float32)
        stress_true = torch.tensor(data['node_stresses'], dtype=torch.float32)
        
        # Additional info
        material_params = torch.tensor([E, nu], dtype=torch.float32)
        force_vector_tensor = torch.tensor(force_vector, dtype=torch.float32)
        
        return Data(
            x=x,  # Node features [N, 5]
            pos=pos,  # Coordinates [N, 3]
            edge_index=edge_index,  # [2, E]
            edge_attr=edge_attr,  # [E, 6]
            f=f,  # Nodal forces [N, 3]
            bc=bc,  # Boundary conditions [N, 3]
            u_true=u_true,  # Displacement labels [N, 3]
            stress_true=stress_true,  # Stress labels [N, 1]
            tetra_connectivity=connectivity,  # [E_tet, 4]
            material_params=material_params,  # [2]
            force_vector=force_vector_tensor,  # [3]
            num_nodes=num_nodes
        )
        
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        raise

class NPZDataset(Dataset):
    """PyG Dataset for NPZ files"""
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
    """Load all NPZ files in directory"""
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    
    if max_samples is not None:
        npz_files = npz_files[:max_samples]
    
    print(f"Loading {len(npz_files)} samples from {npz_dir}")
    
    data_list = []
    for npz_file in tqdm(npz_files, desc="Loading data"):
        try:
            data = load_single_npz(npz_file)
            data_list.append(data)
        except Exception as e:
            print(f"Skipping {npz_file}: {e}")
    
    return data_list

def split_dataset(data_list, train_ratio=0.8, val_ratio=0.2):
    """Split dataset into train and validation"""
    assert train_ratio + val_ratio == 1.0, "Ratios must sum to 1.0"
    
    num_samples = len(data_list)
    num_train = int(num_samples * train_ratio)
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    
    return train_data, val_data

def create_data_loaders(train_data, val_data, batch_size=4):
    """Create data loaders"""
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader