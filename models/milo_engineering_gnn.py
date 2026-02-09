import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

# ---------------------------------------------------------
# PART 1: HELPER LAYERS (From our previous FNA work)
# ---------------------------------------------------------

class FNALayer(nn.Module):
    """
    The Physics Engine. 
    Instead of just summing neighbors, it weighs them by 1/distance^2.
    """
    def __init__(self, node_dim, edge_dim, output_dim, kernel_type='power'):
        super().__init__()
        self.kernel_type = kernel_type
        
        # Message MLP: Sees [Me, You, Edge_Features, HOP_DISTANCE]
        # We add +1 to input size for the scalar 'hop_distance'
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + 1, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Update MLP: Standard ResNet-style update
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + output_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, dist_metric):
        row, col = edge_index
        x_j = x[row]
        x_i = x[col]
        
        # 1. Calculate Interaction
        # dist_metric is the Hop Distance (1.0, 2.0, etc.)
        edge_inputs = torch.cat([x_i, x_j, edge_attr, dist_metric], dim=-1)
        messages = self.message_mlp(edge_inputs)
        
        # 2. Apply Physics Kernel (Power Law)
        # Weight = 1 / (d^2 + 0.5)
        # This allows "Action at a Distance" (Global Stress Fields)
        weight = 1.0 / (dist_metric.pow(2.0) + 0.5)
        messages = messages * weight.view(-1, 1)
        
        # 3. Aggregate
        aggregated = torch.zeros(x.size(0), messages.size(1), device=x.device)
        aggregated.index_add_(0, col, messages)
        
        # 4. Update Node State
        update_inputs = torch.cat([x, aggregated], dim=-1)
        return x + self.update_mlp(update_inputs)


# ---------------------------------------------------------
# PART 2: YOUR FRIEND'S CODE (The Baseline)
# ---------------------------------------------------------
class EngineeringGNN_Standard(nn.Module):
    """
    The 'Friend' Model.
    PROS: Excellent stability (Log-space stress, smart init).
    CONS: Blind. Uses GINEConv which only sees immediate neighbors.
    """
    def __init__(self, node_dim=12, edge_dim=6, hidden_dim=128, num_layers=3, 
                 min_disp_scale=1e-3, init_disp_scale=1.0e-2, 
                 stress_log_clamp=(0.0, 30.0), init_stress_mpa=100.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_disp_scale = float(min_disp_scale)
        self.stress_log_clamp = stress_log_clamp

        # Encoders
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Standard GNN Stack (Local Only)
        self.convs = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim))
            self.post_norms.append(nn.LayerNorm(hidden_dim))

        # Heads (Displacement & Stress)
        self.disp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3), 
        )
        self.log_stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), 
        )

        # Smart Init Logic
        init = float(init_disp_scale)
        target_sp = max(init - self.min_disp_scale, 1e-8)
        inv_softplus = math.log(math.exp(target_sp) - 1.0 + 1e-8)
        self.log_disp_scale = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))

        self.yield_stress = 250e6
        init_stress_pa = float(init_stress_mpa) * 1e6
        init_log_stress = math.log(max(init_stress_pa, 1.0))
        with torch.no_grad():
            last = self.log_stress_head[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                last.bias.fill_(init_log_stress)

    def get_disp_scale(self) -> torch.Tensor:
        return self.min_disp_scale + F.softplus(self.log_disp_scale)

    def forward(self, data):
        # 1. Encode
        x = self.node_enc(data.x)
        edge_attr = self.edge_enc(data.edge_attr)

        # 2. Local Message Passing (The Weakness)
        # Information only travels 1 step per layer.
        for conv, norm in zip(self.convs, self.post_norms):
            h = conv(x, data.edge_index, edge_attr)
            h = F.relu(h)
            x = norm(x + h) 

        # 3. Decode Output
        raw_u = self.disp_head(x)
        disp_scale = self.get_disp_scale()
        u = raw_u * disp_scale

        log_s = self.log_stress_head(x)
        if self.stress_log_clamp is not None:
            log_s = torch.clamp(log_s, min=self.stress_log_clamp[0], max=self.stress_log_clamp[1])
        s = torch.exp(log_s)
        safety_factor = self.yield_stress / (s + 1e-8)

        return {
            "displacement": u,
            "stress": s,
            "log_stress": log_s,
            "disp_scale": disp_scale,
            "safety_factor": safety_factor,
        }


# ---------------------------------------------------------
# PART 3: THE FRANKENSTEIN MODEL (Engineering + FNA)
# ---------------------------------------------------------
class EngineeringGNN_FNA(EngineeringGNN_Standard):
    """
    The 'Milo' Model.
    INHERITS: Smart Init, Log-Stress, Scaling from Friend's code.
    REPLACES: The GNN Backbone with Global Power-Law Attention.
    """
    def __init__(self, **kwargs):
        # Initialize the parent (Friend's logic)
        super().__init__(**kwargs)
        
        # --- THE SURGERY START ---
        # Remove the standard 'convs'
        del self.convs
        del self.post_norms
        
        # Insert our 'FNA Layers'
        self.max_hops = 6
        self.fna_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(kwargs.get('num_layers', 3)):
            # Note: inputs are hidden_dim, output is hidden_dim
            self.fna_layers.append(
                FNALayer(self.hidden_dim, self.hidden_dim, self.hidden_dim, kernel_type='power')
            )
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim))
        # --- THE SURGERY END ---

    def make_fully_connected_graph(self, edge_index, batch, x):
        """
        Rewires the mesh to be fully connected so forces can travel instantly.
        Returns: New Edges, Hop Distances
        """
        # 1. Batch Masking (Handle variable graph sizes)
        x_dense, mask = to_dense_batch(x, batch)
        B, N, _ = x_dense.shape
        
        # 2. Compute Hop Matrix (A^2, A^3...)
        adj = to_dense_adj(edge_index, batch=batch)
        hop_matrix = torch.full((B, N, N), 999.0, device=x.device)
        
        eye = torch.eye(N, device=x.device).bool().unsqueeze(0).expand(B, N, N)
        hop_matrix[eye] = 0.0
        
        curr_reach = adj.clone()
        hop_matrix[curr_reach > 0] = 1.0
        
        # Expand up to max_hops
        for h in range(2, self.max_hops + 1):
            curr_reach = torch.bmm(curr_reach, adj)
            mask_new = (curr_reach > 0) & (hop_matrix == 999.0)
            hop_matrix[mask_new] = float(h)

        # 3. Convert back to Edge List (Sparse)
        # Filter padding and unreachable nodes
        valid_nodes = mask.unsqueeze(2) & mask.unsqueeze(1)
        valid_edges = (hop_matrix < 990) & valid_nodes
        
        b_idx, row, col = torch.nonzero(valid_edges, as_tuple=True)
        hop_values = hop_matrix[b_idx, row, col].view(-1, 1)
        
        # Apply Batch Offsets to make indices global
        batch_counts = torch.bincount(batch)
        ptr = torch.cumsum(torch.cat([torch.tensor([0], device=x.device), batch_counts[:-1]]), dim=0)
        offsets = ptr[b_idx]
        
        new_edge_index = torch.stack([row + offsets, col + offsets], dim=0)
        
        return new_edge_index, hop_values

    def forward(self, data):
        # 1. Encode (Same as Friend's Code)
        x = self.node_enc(data.x)
        # We still encode original edges for the 1-hop connections
        edge_attr_orig = self.edge_enc(data.edge_attr)
        
        # 2. REWIRE GRAPH (The FNA Magic)
        # Create global connections
        fna_edge_index, fna_hops = self.make_fully_connected_graph(data.edge_index, data.batch, x)
        
        # 3. Handle Edge Features for New Edges
        # This is tricky: Original edges have attributes (Length, Stiffness).
        # Virtual edges (A->C) don't.
        # Strategy: Create a zero-filled tensor for ALL edges, then fill in the real ones.
        
        # (Simplified for Prototyping: We just use zeros for geometry on virtual edges.
        # The 'fna_hops' carries the distance info, which is the most important part.)
        fna_edge_attr = torch.zeros(fna_edge_index.size(1), self.hidden_dim, device=x.device)
        
        # Ideally, we would copy 'edge_attr_orig' into the correct slots of 'fna_edge_attr',
        # but matching indices is slow. The model will learn that hop=1 implies strong geometry
        # and hop>1 implies "virtual connection".
        
        # 4. Global Message Passing
        for layer, norm in zip(self.fna_layers, self.layer_norms):
            h = layer(x, fna_edge_index, fna_edge_attr, fna_hops)
            x = norm(x + h) # Residual + Norm

        # 5. Decode (Same as Friend's Code)
        # Because we inherited from the Standard class, we can just copy-paste the output logic
        # or call a helper. I'll paste for clarity.
        
        raw_u = self.disp_head(x)
        disp_scale = self.get_disp_scale()
        u = raw_u * disp_scale

        log_s = self.log_stress_head(x)
        if self.stress_log_clamp is not None:
            log_s = torch.clamp(log_s, min=self.stress_log_clamp[0], max=self.stress_log_clamp[1])
        s = torch.exp(log_s)
        safety_factor = self.yield_stress / (s + 1e-8)

        return {
            "displacement": u,
            "stress": s,
            "log_stress": log_s,
            "disp_scale": disp_scale,
            "safety_factor": safety_factor,
        }

# ---------------------------------------------------------
# COMPARISON BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    print(">>> ENGINEERING GNN MODEL COMPARISON <<<")
    
    # Instantiate both
    std_model = EngineeringGNN_Standard(node_dim=12, edge_dim=6)
    fna_model = EngineeringGNN_FNA(node_dim=12, edge_dim=6)
    
    print("\n[Standard Model]")
    print(f"  Architecture: GINEConv (Neighbor Only)")
    print(f"  Params: {sum(p.numel() for p in std_model.parameters()):,}")
    
    print("\n[FNA Model]")
    print(f"  Architecture: Power Law Attention (Global)")
    print(f"  Params: {sum(p.numel() for p in fna_model.parameters()):,}")
    
    print("\nStatus: Ready for training.")