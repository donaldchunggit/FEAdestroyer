# train_advanced_simple.py
import os, sys, torch, warnings, math
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import deque
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models.engineering_gnn import EngineeringGNN
    print("✓ Engineering GNN imported")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

from utils.data_loader import load_npz_dataset

class Config:
    EXPERIMENT_NAME = "advanced_professor_v1"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_DIR = "generator_adv/advanced_dataset/train"
    VAL_DIR = "generator_adv/advanced_dataset/val"
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    BATCH_SIZE = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    WARMUP_EPOCHS = 5
    YIELD_STRESS = 250e6
    MIN_SAFETY_FACTOR = 1.5
    HUBER_DELTA = 1e-3
    EMA_MOMENTUM = 0.9
    TARGET_DISP_ERROR = 10.0
    TARGET_STRESS_ERROR = 15.0

    # ADJUSTED: Increased scale_weight in early stage to fix magnitude errors
    CURRICULUM_STAGES = [
        {"name": "scale", "epochs": 20, "disp_weight": 1.0, "stress_weight": 0.0, "scale_weight": 40.0},
        {"name": "displacement", "epochs": 30, "disp_weight": 1.0, "stress_weight": 0.3, "scale_weight": 15.0},
        {"name": "stress", "epochs": 30, "disp_weight": 1.0, "stress_weight": 0.7, "scale_weight": 5.0},
        {"name": "physics", "epochs": 20, "disp_weight": 1.0, "stress_weight": 1.0, "scale_weight": 2.0},
    ]

config = Config()

def safe_log(x, eps=1e-12):
    return torch.log(torch.clamp(x, min=eps))

def get_free_mask(batch):
    if hasattr(batch, "free_mask") and batch.free_mask is not None:
        return batch.free_mask.float()
    if hasattr(batch, "fixed_mask") and batch.fixed_mask is not None:
        return (1.0 - batch.fixed_mask.float())
    return None

class PhysicsInformedLoss:
    @staticmethod
    def beam_deflection_loss(displacements, positions, force_vector, E, I, length):
        force_magnitude = torch.norm(force_vector, dim=-1, keepdim=True)
        force_dir = force_vector / (force_magnitude + 1e-8)
        z = positions[:, 2:3]
        theoretical_magnitude = (force_magnitude / (6 * E * I + 1e-12)) * (3 * length * z**2 - z**3)
        theoretical = force_dir * theoretical_magnitude
        return (displacements - theoretical).norm(dim=-1).mean()

    @staticmethod
    def compatibility_loss(displacements, edge_index):
        src, dst = edge_index
        return (displacements[src] - displacements[dst]).norm(dim=-1).mean()

class AdaptiveLossWeights:
    def __init__(self, momentum=0.9, target_disp_error=10.0, target_stress_error=15.0):
        self.momentum, self.target_disp_error, self.target_stress_error = momentum, target_disp_error, target_stress_error
        self.disp_weight, self.stress_weight, self.scale_weight = 1.0, 0.5, 5.0
        self.corr_weight, self.physics_weight = 0.2, 0.1
        self.disp_error_history, self.stress_error_history = deque(maxlen=5), deque(maxlen=5)

    def update(self, disp_error, stress_error, phase_name):
        self.disp_error_history.append(disp_error)
        self.stress_error_history.append(stress_error)
        if len(self.disp_error_history) < 2: return None
        
        if phase_name == "scale" and disp_error > self.target_disp_error:
            self.scale_weight = min(self.scale_weight * 1.05, 50.0)
        elif phase_name == "displacement" and disp_error > self.target_disp_error:
            self.disp_weight = min(self.disp_weight * 1.05, 5.0)
        
        return {"scale_weight": self.scale_weight, "disp_weight": self.disp_weight}

class AdvancedTrainer:
    def __init__(self, model, config, device):
        self.model, self.config, self.device = model, config, device
        self.adaptive_weights = AdaptiveLossWeights(momentum=config.EMA_MOMENTUM)
        self.physics_loss = PhysicsInformedLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.history = {"train_loss": [], "val_loss": [], "disp_error": []}
        self.best_val_loss = float("inf")

    def train_epoch(self, loader, phase):
        self.model.train()
        total_loss, num_batches = 0.0, 0
        pbar = tqdm(loader, desc=f"Phase: {phase['name']}", leave=False)

        for batch in pbar:
            batch = batch.to(self.device)
            out = self.model(batch)
            u_pred, u_true = out["displacement"], batch.u_true
            free_mask = get_free_mask(batch)
            bidx = getattr(batch, "batch", torch.zeros(u_true.size(0), dtype=torch.long, device=self.device))
            mask_mult = free_mask if free_mask is not None else 1.0

            loss_u = F.smooth_l1_loss(u_pred * mask_mult, u_true * mask_mult, beta=self.config.HUBER_DELTA)
            
            # Scale supervision
            u_true_use, raw_use, bidx_use = u_true, out["raw_displacement"], bidx
            if free_mask is not None:
                m = free_mask.squeeze(-1) > 0.5
                u_true_use, raw_use, bidx_use = u_true[m], raw_use[m], bidx[m]
            
            if u_true_use.numel() > 0:
                gt_rms = global_mean_pool((u_true_use**2).sum(-1, keepdim=True), bidx_use).sqrt().clamp(min=1e-12)
                raw_rms = global_mean_pool((raw_use**2).sum(-1, keepdim=True), bidx_use).sqrt().clamp(min=1e-12)
                scale_target = (gt_rms / raw_rms).clamp(min=1e-6, max=1e6)
                loss_scale = F.mse_loss(safe_log(out["disp_scale_graph"]), safe_log(scale_target))
            else:
                loss_scale = torch.tensor(0.0, device=self.device)

            loss_stress = F.mse_loss(safe_log(out["stress"] + 1.0), safe_log(batch.stress_true + 1.0)) if phase["stress_weight"] > 0 else torch.tensor(0.0, device=self.device)

            total = phase["disp_weight"] * loss_u + phase["scale_weight"] * loss_scale + phase["stress_weight"] * loss_stress
            
            self.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()
            total_loss += total.item()
            num_batches += 1
            
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        errs = []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            u_pred, u_true = out["displacement"], batch.u_true
            bidx = getattr(batch, "batch", torch.zeros(u_true.size(0), dtype=torch.long, device=self.device))
            
            err2 = ((u_pred - u_true)**2).sum(-1, keepdim=True)
            tru2 = (u_true**2).sum(-1, keepdim=True)
            rel = (global_mean_pool(err2, bidx).sqrt() / global_mean_pool(tru2, bidx).sqrt().clamp(min=1e-12)).mean().item()
            errs.append(rel * 100.0)
        return {"rel_disp_error": np.mean(errs), "val_loss": np.mean(errs)}

    def train(self, train_loader, val_loader):
        for epoch in range(1, self.config.EPOCHS + 1):
            phase = next(p for p in reversed(self.config.CURRICULUM_STAGES) if epoch > sum(s['epochs'] for s in self.config.CURRICULUM_STAGES if self.config.CURRICULUM_STAGES.index(s) < self.config.CURRICULUM_STAGES.index(p)))
            train_loss = self.train_epoch(train_loader, phase)
            val_summary = self.validate(val_loader)
            print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Rel Err: {val_summary['rel_disp_error']:.2f}%")

def main():
    device = torch.device(config.DEVICE)
    train_data = load_npz_dataset(config.TRAIN_DIR)
    val_data = load_npz_dataset(config.VAL_DIR)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)
    
    sample = train_data[0]
    model = EngineeringGNN(node_dim=sample.x.shape[1], edge_dim=sample.edge_attr.shape[1]).to(device)
    trainer = AdvancedTrainer(model, config, device)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()