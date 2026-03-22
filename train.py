#!/usr/bin/env python3
"""Main FEA Destroyer training script.

Uses EngineeringGNN with curriculum learning and correlation-based model selection.
Run: python train.py
"""
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
    HIDDEN_DIM = 256
    NUM_LAYERS = 5
    BATCH_SIZE = 4
    EPOCHS = 200  # More epochs
    LEARNING_RATE = 3e-4  # Slightly lower
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    WARMUP_EPOCHS = 15  # Longer warmup
    YIELD_STRESS = 250e6
    MIN_SAFETY_FACTOR = 1.5
    HUBER_DELTA = 1e-3
    EMA_MOMENTUM = 0.9
    TARGET_DISP_ERROR = 10.0
    TARGET_STRESS_ERROR = 15.0

    # Adjusted curriculum
    CURRICULUM_STAGES = [
        {"name": "warmup", "epochs": 15, "disp_weight": 0.1, "stress_weight": 0.0, "scale_weight": 1000.0, "corr_weight": 0.0},
        {"name": "scale", "epochs": 25, "disp_weight": 0.5, "stress_weight": 0.0, "scale_weight": 500.0, "corr_weight": 0.0},
        {"name": "correlation", "epochs": 80, "disp_weight": 1.0, "stress_weight": 0.1, "scale_weight": 100.0, "corr_weight": 0.5},
        {"name": "stress", "epochs": 80, "disp_weight": 1.0, "stress_weight": 0.5, "scale_weight": 20.0, "corr_weight": 0.2},
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

def get_curriculum_phase(epoch, stages):
    cumulative_epochs = 0
    for stage in stages:
        if epoch <= cumulative_epochs + stage["epochs"]:
            return stage
        cumulative_epochs += stage["epochs"]
    return stages[-1]

def correlation_loss(u_pred, u_true, mask=None):
    """Improved correlation loss that's bounded and stable"""
    if mask is not None:
        u_pred = u_pred[mask]
        u_true = u_true[mask]
    
    if u_pred.numel() == 0:
        return torch.tensor(0.0, device=u_pred.device)
    
    # Normalize predictions and targets to have zero mean and unit variance
    u_pred_norm = (u_pred - u_pred.mean(dim=0, keepdim=True)) / (u_pred.std(dim=0, keepdim=True) + 1e-8)
    u_true_norm = (u_true - u_true.mean(dim=0, keepdim=True)) / (u_true.std(dim=0, keepdim=True) + 1e-8)
    
    # FIXED: Better correlation loss - use MSE on normalized values
    # This encourages correlation without the numerical issues of 1-corr
    loss = F.mse_loss(u_pred_norm, u_true_norm)
    
    return loss

class AdvancedTrainer:
    def __init__(self, model, config, device):
        self.model, self.config, self.device = model, config, device
        
        # Create separate optimizer with higher LR for scale parameters
        scale_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'log_base_disp_scale' in name or 'scale' in name.lower():
                scale_params.append(param)
            else:
                other_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': config.LEARNING_RATE},
            {'params': scale_params, 'lr': config.LEARNING_RATE * 3.0},  # Reduced from 5x
        ], weight_decay=config.WEIGHT_DECAY)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.history = {"train_loss": [], "val_loss": [], "correlation": []}
        self.best_correlation = -float("inf")
        self.best_model_path = "best_model_correlation.pt"

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
            
            # Apply mask for displacement
            if free_mask is not None:
                m = free_mask.squeeze(-1) > 0.5
                u_pred_masked = u_pred[m]
                u_true_masked = u_true[m]
                bidx_masked = bidx[m]
                raw_masked = out["raw_displacement"][m]
            else:
                u_pred_masked = u_pred
                u_true_masked = u_true
                bidx_masked = bidx
                raw_masked = out["raw_displacement"]
            
            # 1. Displacement loss (Huber for robustness)
            loss_u = F.huber_loss(u_pred_masked, u_true_masked, delta=self.config.HUBER_DELTA)
            
            # 2. Scale supervision
            if u_true_masked.numel() > 0:
                gt_rms = global_mean_pool((u_true_masked**2).sum(-1, keepdim=True), bidx_masked).sqrt()
                raw_rms = global_mean_pool((raw_masked**2).sum(-1, keepdim=True), bidx_masked).sqrt().clamp(min=1e-8)
                scale_target = (gt_rms / raw_rms).squeeze()
                
                # Handle shape mismatches
                if scale_target.dim() == 0:
                    scale_target = scale_target.unsqueeze(0)
                if out["disp_scale_graph"].dim() == 0:
                    disp_scale = out["disp_scale_graph"].unsqueeze(0)
                else:
                    disp_scale = out["disp_scale_graph"]
                
                # Use log scale for better gradient flow
                loss_scale = F.mse_loss(safe_log(disp_scale), safe_log(scale_target))
            else:
                loss_scale = torch.tensor(0.0, device=self.device)
            
            # 3. FIXED: Better correlation loss
            loss_corr = correlation_loss(u_pred_masked, u_true_masked)
            
            # 4. Stress loss (if applicable)
            loss_stress = torch.tensor(0.0, device=self.device)
            if phase["stress_weight"] > 0 and hasattr(batch, "stress_true"):
                if free_mask is not None:
                    stress_pred = out["stress"][m]
                    stress_true = batch.stress_true[m]
                else:
                    stress_pred = out["stress"]
                    stress_true = batch.stress_true
                
                if stress_pred.numel() > 0:
                    loss_stress = F.mse_loss(safe_log(stress_pred + 1.0), safe_log(stress_true + 1.0))
            
            # Combined loss with phase weights
            total = (phase["disp_weight"] * loss_u + 
                    phase["scale_weight"] * loss_scale + 
                    phase.get("corr_weight", 0.0) * loss_corr +
                    phase["stress_weight"] * loss_stress)
            
            self.optimizer.zero_grad()
            total.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()
            
            total_loss += total.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total.item():.4f}',
                'corr_loss': f'{loss_corr.item():.3f}',
                'scale': f'{out["disp_scale"].item():.6f}'
            })
            
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, loader):
        """Validation focused on correlation"""
        self.model.eval()
        
        all_u_true = []
        all_u_pred = []
        val_losses = []
        
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            
            u_true = batch.u_true
            u_pred = out["displacement"]
            free_mask = get_free_mask(batch)
            
            # Apply mask
            if free_mask is not None:
                m = free_mask.squeeze(-1) > 0.5
                u_true_masked = u_true[m]
                u_pred_masked = u_pred[m]
            else:
                u_true_masked = u_true
                u_pred_masked = u_pred
            
            if u_pred_masked.numel() > 0:
                val_losses.append(F.mse_loss(u_pred_masked, u_true_masked).item())
                
                # Store for correlation calculation
                all_u_true.append(u_true_masked.cpu().numpy())
                all_u_pred.append(u_pred_masked.cpu().numpy())
        
        if not all_u_true:
            return {"mse": 0.0, "correlation": 0.0, "mae_mm": 0.0}
        
        # Concatenate all data
        u_true = np.concatenate(all_u_true, axis=0)
        u_pred = np.concatenate(all_u_pred, axis=0)
        
        # Compute metrics
        mse = np.mean((u_pred - u_true)**2)
        mae_mm = np.mean(np.abs(u_pred - u_true)) * 1000.0
        
        # FIXED: Better correlation calculation
        # Reshape to combine all dimensions for a single correlation metric
        u_true_flat = u_true.flatten()
        u_pred_flat = u_pred.flatten()
        
        # Compute Pearson correlation
        corr_matrix = np.corrcoef(u_true_flat, u_pred_flat)
        correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        
        return {
            "mse": float(np.mean(val_losses)) if val_losses else 0.0,
            "correlation": correlation,
            "mae_mm": mae_mm,
            "true_mean_mm": np.mean(np.abs(u_true)) * 1000.0,
            "pred_mean_mm": np.mean(np.abs(u_pred)) * 1000.0,
        }

    def train(self, train_loader, val_loader):
        print("\n" + "="*80)
        print("FEA DESTROYER — TRAINING")
        print("="*80)
        
        for epoch in range(1, self.config.EPOCHS + 1):
            phase = get_curriculum_phase(epoch, self.config.CURRICULUM_STAGES)
            
            # Train
            train_loss = self.train_epoch(train_loader, phase)
            
            # Validate
            stats = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(stats["mse"])
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            print(f"\nEpoch {epoch:3d} [{phase['name']}] LR: {current_lr:.2e}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  True mean:  {stats['true_mean_mm']:.3f}mm")
            print(f"  Pred mean:  {stats['pred_mean_mm']:.3f}mm")
            print(f"  MAE:        {stats['mae_mm']:.3f}mm")
            print(f"  Correlation: {stats['correlation']:.3f}")
            
            # Track best model by correlation
            if stats['correlation'] > self.best_correlation:
                self.best_correlation = stats['correlation']
                print(f"  ✓ NEW BEST CORRELATION: {stats['correlation']:.3f}")
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'correlation': stats['correlation'],
                    'mae_mm': stats['mae_mm'],
                    'config': {k: v for k, v in vars(self.config).items() if not k.startswith('_')}
                }, self.best_model_path)
            
            # Early stopping if correlation is good enough
            if stats['correlation'] > 0.7:  # Slightly lower threshold
                print(f"\n🎉 Achieved good correlation! Stopping early.")
                break
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print(f"Best correlation achieved: {self.best_correlation:.3f}")
        print(f"Best model saved to: {self.best_model_path}")
        print("="*80)

def main():
    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    device = torch.device(config.DEVICE)
    
    print("Loading datasets...")
    train_data = load_npz_dataset(config.TRAIN_DIR, max_samples=config.MAX_TRAIN_SAMPLES)
    val_data = load_npz_dataset(config.VAL_DIR, max_samples=config.MAX_VAL_SAMPLES)
    
    print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")
    
    _pin = (config.DEVICE == "cuda")
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=_pin)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, pin_memory=_pin)
    
    sample = train_data[0]
    model = EngineeringGNN(
        node_dim=sample.x.shape[1], 
        edge_dim=sample.edge_attr.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS
    ).to(device)
    
    # Initialize scale based on data statistics
    with torch.no_grad():
        # Compute average scale from a few batches
        scales = []
        for i, batch in enumerate(train_loader):
            if i >= 5: break
            batch = batch.to(device)
            u_true = batch.u_true
            free_mask = get_free_mask(batch)
            if free_mask is not None:
                u_true = u_true[free_mask.squeeze(-1) > 0.5]
            if u_true.numel() > 0:
                scales.append(u_true.abs().mean().item())
        
        if scales:
            init_scale = np.mean(scales)
            model.log_base_disp_scale.fill_(np.log(max(init_scale, 1e-6)))
            print(f"Initialized scale from data: {init_scale:.6f}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial scale: {torch.exp(model.log_base_disp_scale).item():.6f}")
    
    trainer = AdvancedTrainer(model, config, device)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()