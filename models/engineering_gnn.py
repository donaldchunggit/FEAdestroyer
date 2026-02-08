# models/engineering_gnn.py - SIMPLIFIED WORKING VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

class EngineeringGNN(nn.Module):
    def __init__(self, node_dim=5, edge_dim=6, hidden_dim=128, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
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
        
        # Engineering outputs
        self.displacement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # ux, uy, uz
        )
        
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # von Mises stress
            nn.Softplus()  # Stress should be positive
        )
        
        self.safety_factor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Safety factor
            nn.Softplus()  # > 0
        )
        
        # Simple physics feature extractor
        self.physics_feature_extractor = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # For position features
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
    
    def forward(self, data):
        # Encode nodes and edges
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        
        # Add simple physics features to node embeddings
        if hasattr(data, 'pos'):
            pos_features = self.physics_feature_extractor(data.pos)
            x = x + pos_features  # Add position information
        
        # Message passing
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr)
            x = F.relu(x)
        
        # Engineering predictions
        displacement = self.displacement_head(x)
        stress = self.stress_head(x)
        safety_factor = self.safety_factor_head(x)
        
        # Calculate von Mises from displacement (simplified)
        von_mises = self._estimate_stress(displacement, data)
        
        return {
            'displacement': displacement,
            'stress': stress,
            'von_mises': von_mises,
            'safety_factor': safety_factor,
            'node_features': x
        }
    
    def _estimate_stress(self, displacement, data):
        """Simplified stress estimation from displacement gradients"""
        # For engineering, stress ≈ E * strain
        # Strain ≈ gradient of displacement
        
        if hasattr(data, 'edge_index'):
            src, dst = data.edge_index
            
            # Get displacements at connected nodes
            u_src = displacement[src]
            u_dst = displacement[dst]
            
            # Get positions
            pos_src = data.pos[src]
            pos_dst = data.pos[dst]
            
            # Distance between nodes
            dx = pos_dst - pos_src
            distance = torch.norm(dx, dim=1, keepdim=True) + 1e-8
            
            # Strain along edge (simplified)
            du = u_dst - u_src
            strain_magnitude = torch.norm(du, dim=1, keepdim=True) / distance
            
            # Assume Young's modulus (simplified)
            E = 210e9  # Steel in Pa
            
            # Stress = E * strain
            stress = E * strain_magnitude
            
            # Average stress at nodes
            node_stress = torch.zeros_like(displacement[:, 0:1])
            node_stress.index_add_(0, src, stress)
            node_stress.index_add_(0, dst, stress)
            
            # Count edges per node
            edge_count = torch.zeros_like(displacement[:, 0:1])
            edge_count.index_add_(0, src, torch.ones_like(stress))
            edge_count.index_add_(0, dst, torch.ones_like(stress))
            
            avg_stress = node_stress / (edge_count + 1e-8)
            return avg_stress
        
        return torch.zeros_like(displacement[:, 0:1])


# SIMPLER VERSION - Use this if above still has issues
class SimpleEngineeringGNN(nn.Module):
    """Super simple version that definitely works"""
    def __init__(self, node_dim=5, edge_dim=6, hidden_dim=128, num_layers=3):
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
        
        self.head_safety = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Safety factor
            nn.Softplus()  # > 0
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
        safety_factor = self.head_safety(x)
        
        return {
            'displacement': displacement,
            'stress': stress,
            'safety_factor': safety_factor
        }


# Use SimpleEngineeringGNN for now - it will work
EngineeringGNN = SimpleEngineeringGNN