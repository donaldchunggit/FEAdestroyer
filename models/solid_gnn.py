"""
3D Solid GNN model for continuum mechanics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class SolidPINN_GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Message passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            conv = GINEConv(nn=mlp, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output heads
        self.displacement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3D displacement
        )
        
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Von Mises stress
            nn.Softplus()  # Stress is positive
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, data):
        # Encode nodes and edges
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        
        # Message passing with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, data.edge_index, edge_attr)
            x = norm(x)
            x = F.gelu(x)
            x = x_res + x  # Residual connection
        
        # Predict outputs
        displacement = self.displacement_head(x)
        stress = self.stress_head(x)
        
        return {
            'displacement': displacement,
            'stress': stress,
            'node_features': x
        }

class SimpleSolidGNN(nn.Module):
    """Simplified version for CPU training"""
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        
        # Node encoder
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Edge encoder
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim))
        
        # Output heads
        self.head_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # 3D displacement
        )
        
        self.head_stress = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Stress
        )
    
    def forward(self, data):
        # Encode
        x = self.node_enc(data.x)
        edge_attr = self.edge_enc(data.edge_attr)
        
        # Message passing
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr)
            x = F.relu(x)
        
        # Predict
        displacement = self.head_u(x)
        stress = self.head_stress(x)
        
        return {
            'displacement': displacement,
            'stress': stress
        }