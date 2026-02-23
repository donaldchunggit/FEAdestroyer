# models/engineering_gnn.py
# Displacement fixes:
# 1) Per-graph RMS normalization of raw_u so scale controls magnitude
# 2) Replace BatchNorm with LayerNorm in scale MLP
# 3) Increased log_mult_bound to 4.0 to prevent scale-locking
# 4) Stronger physics-informed global features
# 5) More robust geometry extraction and batching

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class EngineeringGNN(nn.Module):
    """
    Engineering GNN with stable displacement scaling.
    """

    def __init__(
        self,
        node_dim: int = 12,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        min_disp_scale: float = 1e-6,
        init_disp_scale: float = 1.0e-3,
        stress_log_clamp: tuple[float, float] = (0.0, 30.0),
        init_stress_mpa: float = 100.0,
        # LOOSENED: Increased from 2.0 to 4.0 to allow for larger scale adjustments
        log_mult_bound: float = 4.0, 
        section_I_index: int = 3,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.min_disp_scale = float(min_disp_scale)
        self.stress_log_clamp = stress_log_clamp
        self.log_mult_bound = float(log_mult_bound)
        self.section_I_index = int(section_I_index)

        # ---------------- Encoders ----------------
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

        # ---------------- GNN trunk ----------------
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

        # ---------------- Heads ----------------
        self.disp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        scale_in_dim = 6
        self.scale_mlp = nn.Sequential(
            nn.Linear(scale_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        target_sp = max(float(init_disp_scale) - self.min_disp_scale, 1e-10)
        inv_softplus = math.log(math.exp(target_sp) - 1.0 + 1e-8)
        self.log_base_disp_scale = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))

        self.log_stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.yield_stress = 250e6
        init_stress_pa = float(init_stress_mpa) * 1e6
        init_log_stress = math.log(max(init_stress_pa, 1.0))
        with torch.no_grad():
            last = self.log_stress_head[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                last.bias.fill_(init_log_stress)
            last_s = self.scale_mlp[-1]
            if isinstance(last_s, nn.Linear) and last_s.bias is not None:
                last_s.bias.fill_(0.0)

    def get_base_disp_scale(self) -> torch.Tensor:
        return self.min_disp_scale + F.softplus(self.log_base_disp_scale)

    def _get_batch(self, data, n_nodes: int, device: torch.device) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(n_nodes, dtype=torch.long, device=device)
        return batch.squeeze() if batch.dim() > 1 else batch

    def _graph_features(self, data, x_emb: torch.Tensor):
        device = x_emb.device
        batch = self._get_batch(data, x_emb.size(0), device)
        B = int(batch.max().item() + 1) if batch.numel() > 0 else 1

        f = getattr(data, "force_vector", getattr(data, "input_force", None))
        if f is None:
            force_mag = torch.ones((B, 1), device=device)
        else:
            if f.dim() == 1: f = f.unsqueeze(0)
            force_mag = torch.norm(f, dim=-1, keepdim=True)
            if force_mag.size(0) == 1 and B > 1: force_mag = force_mag.repeat(B, 1)

        if hasattr(data, "material_params") and data.material_params is not None:
            mp = data.material_params
            if mp.dim() == 1: mp = mp.unsqueeze(0)
            E = mp[:, 0:1].clamp(min=1.0)
            nu = mp[:, 1:2].clamp(0.0, 0.49)
            if E.size(0) == 1 and B > 1:
                E, nu = E.repeat(B, 1), nu.repeat(B, 1)
        else:
            E, nu = torch.full((B, 1), 210e9, device=device), torch.full((B, 1), 0.3, device=device)

        pos = getattr(data, "pos", getattr(data, "node_coords", None))
        if pos is not None:
            z = pos[:, 2]
            zmin = torch.full((B,), float("inf"), device=device)
            zmax = torch.full((B,), float("-inf"), device=device)
            zmin = zmin.scatter_reduce(0, batch, z, reduce="amin", include_self=True)
            zmax = zmax.scatter_reduce(0, batch, z, reduce="amax", include_self=True)
            L = (zmax - zmin).clamp(min=1e-6).unsqueeze(-1)
        else:
            L = torch.ones((B, 1), device=device)

        if hasattr(data, "section_props") and data.section_props is not None:
            sp = data.section_props
            if sp.dim() == 1: sp = sp.unsqueeze(0)
            idx = self.section_I_index
            I = sp[:, idx:idx+1].clamp(min=1e-18) if sp.size(-1) > idx else torch.ones((sp.size(0), 1), device=device)
            if I.size(0) == 1 and B > 1: I = I.repeat(B, 1)
        else:
            I = torch.ones((B, 1), device=device)

        logF, logE, logL, logI = torch.log(force_mag + 1.0), torch.log(E + 1e-12), torch.log(L + 1e-6), torch.log(I + 1e-18)
        phys = logF + 3.0 * logL - logE - logI
        return torch.cat([logF, logE, nu, logL, logI, phys], dim=-1), batch

    def forward(self, data):
        x = self.node_enc(data.x)
        edge_attr = self.edge_enc(data.edge_attr)
        for conv, norm in zip(self.convs, self.post_norms):
            h = conv(x, data.edge_index, edge_attr)
            x = norm(x + F.relu(h))

        raw_u = self.disp_head(x)
        batch = self._get_batch(data, raw_u.size(0), raw_u.device)
        raw_mag2 = (raw_u ** 2).sum(dim=-1, keepdim=True)
        raw_rms = global_mean_pool(raw_mag2, batch).sqrt().clamp(min=1e-8)
        raw_u = raw_u / raw_rms[batch]

        feats, _ = self._graph_features(data, x)
        base = self.get_base_disp_scale()
        log_mult = self.scale_mlp(feats)
        if self.log_mult_bound > 0:
            log_mult = self.log_mult_bound * torch.tanh(log_mult / self.log_mult_bound)
        
        disp_scale_graph = (base * torch.exp(log_mult)).clamp(min=self.min_disp_scale)
        u = raw_u * disp_scale_graph[batch]

        log_s = self.log_stress_head(x)
        if self.stress_log_clamp:
            log_s = torch.clamp(log_s, *self.stress_log_clamp)
        s = torch.exp(log_s)
        return {
            "displacement": u,
            "raw_displacement": raw_u,
            "stress": s,
            "log_stress": log_s,
            "disp_scale": disp_scale_graph.mean(),
            "disp_scale_graph": disp_scale_graph,
            "safety_factor": self.yield_stress / (s + 1e-8),
        }