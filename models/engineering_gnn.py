# models/engineering_gnn.py - FIXED VERSION (realistic stress scale + proper init)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class EngineeringGNN(nn.Module):
    """
    Key fixes:
    - Stress is predicted in LOG-SPACE (log(Pa)) and exponentiated => positive + stable.
    - Stress head final bias is initialized to log(1e8) (~100 MPa) so it doesn't start at ~1 Pa.
    - Removed the tiny stress clamp that forced stresses <= ~1 kPa.
      Instead, use a very wide clamp ONLY to avoid exp overflow.
    - Residual GINEConv stack retained.
    - Displacement scaling stays positive and bounded away from zero.
    """

    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        min_disp_scale: float = 1e-3,       # meters
        init_disp_scale: float = 1.0e-2,    # meters
        # Wide log-stress clamp (in log(Pa)) just to prevent exp overflow:
        # log(1 Pa)=0, log(1e12 Pa)=~27.6
        # Typical range for you: 10–400 MPa => 1e7–4e8 Pa => log ~ 16.1–19.8
        stress_log_clamp: tuple[float, float] = (0.0, 30.0),
        init_stress_mpa: float = 100.0,     # initialize around 100 MPa
    ):
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

        # Residual GINEConv stack
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

        # Heads
        self.disp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # ux, uy, uz (raw)
        )

        # Predict log-stress (log(Pa))
        self.log_stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # log_sigma(Pa)
        )

        # Displacement scale parameterization:
        # disp_scale = min_disp_scale + softplus(log_disp_scale)
        init = float(init_disp_scale)
        # Choose log_disp_scale such that softplus(log_disp_scale) ~= (init - min)
        target_sp = max(init - self.min_disp_scale, 1e-8)
        # inverse softplus approx: log(exp(x)-1)
        inv_softplus = math.log(math.exp(target_sp) - 1.0 + 1e-8)
        self.log_disp_scale = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))

        # Yield stress for safety factor output (loss handled in training)
        self.yield_stress = 250e6

        # ---- IMPORTANT: stress head init to realistic scale ----
        # Initialize final bias so exp(log_s) ~ init_stress_mpa * 1e6 Pa
        init_stress_pa = float(init_stress_mpa) * 1e6
        init_log_stress = math.log(max(init_stress_pa, 1.0))
        with torch.no_grad():
            # last layer is the Linear(hidden_dim//2, 1)
            last = self.log_stress_head[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                last.bias.fill_(init_log_stress)

    def get_disp_scale(self) -> torch.Tensor:
        return self.min_disp_scale + F.softplus(self.log_disp_scale)

    def forward(self, data):
        x = self.node_enc(data.x)
        edge_attr = self.edge_enc(data.edge_attr)

        for conv, norm in zip(self.convs, self.post_norms):
            h = conv(x, data.edge_index, edge_attr)
            h = F.relu(h)
            x = norm(x + h)  # residual + norm

        # Displacement
        raw_u = self.disp_head(x)
        disp_scale = self.get_disp_scale()
        u = raw_u * disp_scale

        # Stress
        log_s = self.log_stress_head(x)  # log(Pa)
        if self.stress_log_clamp is not None:
            log_s = torch.clamp(log_s, min=self.stress_log_clamp[0], max=self.stress_log_clamp[1])
        s = torch.exp(log_s)  # Pa

        safety_factor = self.yield_stress / (s + 1e-8)

        return {
            "displacement": u,
            "stress": s,
            "log_stress": log_s,
            "disp_scale": disp_scale,
            "safety_factor": safety_factor,
        }


# Backwards-compatible alias
EngineeringGNN = EngineeringGNN
