# models/engineering_gnn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class EngineeringGNN(nn.Module):
    """
    - Node-wise raw displacement head
    - Per-graph displacement scale from (force, material, geometry)
    - Stress in log-space
    """

    def __init__(
        self,
        node_dim: int = 12,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        min_disp_scale: float = 1e-5,
        init_disp_scale: float = 1.0e-2,
        stress_log_clamp: tuple[float, float] = (0.0, 30.0),
        init_stress_mpa: float = 100.0,
        log_mult_clamp: tuple[float, float] = (-8.0, 8.0),
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.min_disp_scale = float(min_disp_scale)
        self.stress_log_clamp = stress_log_clamp
        self.log_mult_clamp = log_mult_clamp

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

        self.disp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # [log(|F|+1), log(E), nu, log(L)] -> log_mult
        self.scale_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learned base disp scale (positive)
        target_sp = max(float(init_disp_scale) - self.min_disp_scale, 1e-10)
        inv_softplus = math.log(math.exp(target_sp) - 1.0 + 1e-8)
        self.log_base_disp_scale = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))

        self.log_stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.yield_stress = 250e6

        # init stress bias ~100MPa
        init_stress_pa = float(init_stress_mpa) * 1e6
        init_log_stress = math.log(max(init_stress_pa, 1.0))
        with torch.no_grad():
            last = self.log_stress_head[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                last.bias.fill_(init_log_stress)

        # init log_mult ~ 0 => exp(0)=1
        with torch.no_grad():
            last_s = self.scale_mlp[-1]
            if isinstance(last_s, nn.Linear) and last_s.bias is not None:
                last_s.bias.fill_(0.0)

    def get_base_disp_scale(self) -> torch.Tensor:
        return self.min_disp_scale + F.softplus(self.log_base_disp_scale)

    def _graph_features(self, data, x_emb: torch.Tensor):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x_emb.size(0), dtype=torch.long, device=x_emb.device)

        B = int(batch.max().item() + 1)

        # force vector (your loader uses force_vector)
        f = None
        if hasattr(data, "force_vector") and data.force_vector is not None:
            f = data.force_vector
        elif hasattr(data, "input_force") and data.input_force is not None:
            f = data.input_force

        if f is None:
            force_mag = torch.ones((B, 1), device=x_emb.device)
        else:
            if f.dim() == 1:
                f = f.unsqueeze(0)
            force_mag = torch.norm(f, dim=-1, keepdim=True)
            if force_mag.size(0) == 1 and B > 1:
                force_mag = force_mag.repeat(B, 1)

        # material params (E, nu)
        if hasattr(data, "material_params") and data.material_params is not None:
            mp = data.material_params
            if mp.dim() == 1:
                mp = mp.unsqueeze(0)
            E = mp[:, 0:1].clamp(min=1.0)
            nu = mp[:, 1:2].clamp(0.0, 0.49)
            if E.size(0) == 1 and B > 1:
                E = E.repeat(B, 1)
                nu = nu.repeat(B, 1)
        else:
            E = torch.full((B, 1), 210e9, device=x_emb.device)
            nu = torch.full((B, 1), 0.3, device=x_emb.device)

        # geometry z-range
        pos = getattr(data, "pos", None)
        if pos is None and hasattr(data, "node_coords"):
            pos = data.node_coords

        if pos is not None:
            z = pos[:, 2]
            zmin = torch.full((B,), float("inf"), device=x_emb.device)
            zmax = torch.full((B,), float("-inf"), device=x_emb.device)

            if hasattr(zmin, "scatter_reduce"):
                zmin = zmin.scatter_reduce(0, batch, z, reduce="amin", include_self=True)
                zmax = zmax.scatter_reduce(0, batch, z, reduce="amax", include_self=True)
                geom_L = (zmax - zmin).clamp(min=1e-6).unsqueeze(-1)
            else:
                # safe fallback
                geom_L = torch.ones((B, 1), device=x_emb.device)
        else:
            geom_L = torch.ones((B, 1), device=x_emb.device)

        f_in = torch.log(force_mag + 1.0)
        logE = torch.log(E + 1e-12)
        L_in = torch.log(geom_L + 1e-6)

        feats = torch.cat([f_in, logE, nu, L_in], dim=-1)  # [B,4]
        return feats, batch

    def forward(self, data):
        x = self.node_enc(data.x)
        edge_attr = self.edge_enc(data.edge_attr)

        for conv, norm in zip(self.convs, self.post_norms):
            h = conv(x, data.edge_index, edge_attr)
            h = F.relu(h)
            x = norm(x + h)

        raw_u = self.disp_head(x)  # [N,3]

        feats, batch = self._graph_features(data, x)
        base = self.get_base_disp_scale()

        log_mult = self.scale_mlp(feats)
        if self.log_mult_clamp is not None:
            log_mult = torch.clamp(log_mult, min=self.log_mult_clamp[0], max=self.log_mult_clamp[1])

        mult = torch.exp(log_mult)
        disp_scale_graph = (base * mult).clamp(min=self.min_disp_scale)  # [B,1]

        u = raw_u * disp_scale_graph[batch]

        log_s = self.log_stress_head(x)
        if self.stress_log_clamp is not None:
            log_s = torch.clamp(log_s, min=self.stress_log_clamp[0], max=self.stress_log_clamp[1])
        s = torch.exp(log_s)

        safety_factor = self.yield_stress / (s + 1e-8)

        return {
            "displacement": u,
            "raw_displacement": raw_u,              # <<< added
            "stress": s,
            "log_stress": log_s,
            "disp_scale": disp_scale_graph.mean(),
            "disp_scale_graph": disp_scale_graph,
            "safety_factor": safety_factor,
        }


EngineeringGNN = EngineeringGNN
