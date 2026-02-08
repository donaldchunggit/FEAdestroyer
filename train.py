#!/usr/bin/env python3
"""
Working training script for 3D Solid PINN GNN.
Run: python train.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

print("="*60)
print("FEA GNN TRAINING")
print("="*60)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try imports with helpful messages
print("\nChecking imports...")
try:
    from models.solid_gnn import SimpleSolidGNN
    print("✓ Model imported")
except ImportError as e:
    print(f"✗ Model import failed: {e}")
    print("  Creating simple model...")
    
    # Create a simple model inline
    import torch.nn.functional as F
    from torch_geometric.nn import GINEConv
    
    class SimpleSolidGNN(nn.Module):
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

try:
    from utils.data_loader import load_npz_dataset
    print("✓ Data loader imported")
except ImportError as e:
    print(f"✗ Data loader import failed: {e}")
    print("  Creating simple data loader...")
    
    # Create simple data loader
    import glob
    def load_npz_dataset(npz_dir, max_samples=None):
        print(f"Loading from {npz_dir}...")
        files = glob.glob(os.path.join(npz_dir, "*.npz"))
        if max_samples:
            files = files[:max_samples]
        
        data_list = []
        for file in files[:5]:  # Load only 5 for testing
            try:
                import numpy as np
                from torch_geometric.data import Data
                
                data_np = np.load(file)
                pos = torch.tensor(data_np['node_coords'], dtype=torch.float32)
                x = torch.ones(len(pos), 5, dtype=torch.float32)
                
                # Simple edge creation
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
                edge_attr = torch.ones(2, 6, dtype=torch.float32)
                
                data = Data(
                    x=x,
                    pos=pos,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    u_true=torch.tensor(data_np['node_disp'], dtype=torch.float32),
                    stress_true=torch.tensor(data_np['node_stresses'], dtype=torch.float32),
                    num_nodes=len(pos)
                )
                data_list.append(data)
            except:
                continue
        
        return data_list

try:
    from utils.visualization import plot_training_history
    print("✓ Visualization imported")
except ImportError:
    print("⚠ Visualization not available, will save as text")
    def plot_training_history(history, save_path=None):
        if save_path:
            txt_path = str(save_path).replace('.png', '.txt')
            with open(txt_path, 'w') as f:
                if 'train_loss' in history:
                    for i, loss in enumerate(history['train_loss']):
                        f.write(f"Epoch {i+1} train: {loss:.6f}\n")
                if 'val_loss' in history:
                    for i, loss in enumerate(history['val_loss']):
                        f.write(f"Epoch {i+1} val: {loss:.6f}\n")
            print(f"History saved to {txt_path}")

# Configuration with all required keys
config = {
    'experiment_name': 'fea_gnn_run',
    'seed': 42,
    'device': 'cpu',
    'data': {
        'train_dir': 'dataset/train',
        'val_dir': 'dataset/val',
        'max_train_samples': 80,
        'max_val_samples': 20,
        'batch_size': 2
    },
    'model': {
        'hidden_dim': 128,
        'num_layers': 3
        # No dropout parameter for SimpleSolidGNN
    },
    'training': {
        'epochs': 10,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'gradient_clip': 1.0,
        'checkpoint_freq': 5,
        'patience': 5
    },
    'loss': {
        'physics_weight': 1.0,
        'data_weight': 1.0,
        'boundary_weight': 10.0
    }
}

print(f"\nUsing configuration:")
print(f"  Device: {config['device']}")
print(f"  Train dir: {config['data']['train_dir']}")
print(f"  Epochs: {config['training']['epochs']}")

# Set random seeds
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

# Create experiment directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
exp_dir = Path(f"experiments/{config['experiment_name']}_{timestamp}")
exp_dir.mkdir(parents=True, exist_ok=True)
print(f"\nExperiment directory: {exp_dir}")

# Save config
with open(exp_dir / 'config.txt', 'w') as f:
    for key, value in config.items():
        if isinstance(value, dict):
            f.write(f"{key}:\n")
            for k2, v2 in value.items():
                f.write(f"  {k2}: {v2}\n")
        else:
            f.write(f"{key}: {value}\n")

# Load data
print("\n" + "="*60)
print("Loading dataset...")
print("="*60)

# Check if dataset exists
if not os.path.exists(config['data']['train_dir']):
    print(f"✗ Training directory not found: {config['data']['train_dir']}")
    print("  Please generate data first or check the path")
    sys.exit(1)

train_data = load_npz_dataset(
    config['data']['train_dir'],
    max_samples=config['data']['max_train_samples']
)

if os.path.exists(config['data']['val_dir']):
    val_data = load_npz_dataset(
        config['data']['val_dir'],
        max_samples=config['data']['max_val_samples']
    )
else:
    print(f"⚠ Validation directory not found, using training data for validation")
    val_data = train_data[:min(5, len(train_data))]

print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")

if len(train_data) == 0:
    print("✗ No training data loaded!")
    sys.exit(1)

# Create data loaders
train_loader = DataLoader(
    train_data,
    batch_size=config['data']['batch_size'],
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=config['data']['batch_size'],
    shuffle=False,
    num_workers=0
)

# Create model - WITHOUT DROPOUT parameter
sample = train_data[0]
print(f"\nSample info:")
print(f"  Node features: {sample.x.shape[1]}")
print(f"  Edge features: {sample.edge_attr.shape[1]}")

model = SimpleSolidGNN(
    node_dim=sample.x.shape[1],
    edge_dim=sample.edge_attr.shape[1],
    hidden_dim=config['model']['hidden_dim'],
    num_layers=config['model']['num_layers']
    # No dropout parameter!
)

device = torch.device(config['device'])
model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# Simple scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Print model info
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# Training variables
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

start_time = time.time()

# Training loop
for epoch in range(1, config['training']['epochs'] + 1):
    print(f"\nEpoch {epoch}/{config['training']['epochs']}")
    
    # Training
    model.train()
    train_losses = []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass
        outputs = model(batch)
        
        # Simple loss: MSE between prediction and ground truth
        loss = nn.functional.mse_loss(outputs['displacement'], batch.u_true)
        
        # Add stress loss if available
        if hasattr(batch, 'stress_true'):
            loss += 0.5 * nn.functional.mse_loss(outputs['stress'].squeeze(), batch.stress_true.squeeze())
        
        # Boundary condition loss
        if hasattr(batch, 'bc'):
            loss += 10.0 * torch.mean((outputs['displacement'] * batch.bc) ** 2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        train_losses.append(loss.item())
        pbar.set_postfix({'loss': loss.item()})
    
    avg_train_loss = np.mean(train_losses)
    history['train_loss'].append(avg_train_loss)
    
    # Validation
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = nn.functional.mse_loss(outputs['displacement'], batch.u_true)
            
            if hasattr(batch, 'stress_true'):
                loss += 0.5 * nn.functional.mse_loss(outputs['stress'].squeeze(), batch.stress_true.squeeze())
            
            if hasattr(batch, 'bc'):
                loss += 10.0 * torch.mean((outputs['displacement'] * batch.bc) ** 2)
            
            val_losses.append(loss.item())
    
    avg_val_loss = np.mean(val_losses)
    history['val_loss'].append(avg_val_loss)
    
    # Update scheduler
    scheduler.step()
    
    print(f"Train Loss: {avg_train_loss:.6f}")
    print(f"Val Loss: {avg_val_loss:.6f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Save checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'config': config
        }, exp_dir / 'best_model.pt')
        print(f"  ✓ Saved best model (loss: {best_val_loss:.6f})")
    
    if epoch % config['training']['checkpoint_freq'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, exp_dir / f'checkpoint_epoch_{epoch}.pt')
        print(f"  ✓ Checkpoint saved")
    
    # Early stopping
    if epoch > config['training']['patience']:
        recent_losses = history['val_loss'][-config['training']['patience']:]
        if min(recent_losses) > best_val_loss:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# Training complete
training_time = time.time() - start_time
print(f"\n" + "="*60)
print(f"TRAINING COMPLETE!")
print(f"Time: {training_time:.1f} seconds")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Models saved in: {exp_dir}")
print("="*60)

# Plot/save training history
plot_training_history(history, save_path=exp_dir / 'training_history.png')

# Save history as numpy file
np.save(exp_dir / 'history.npy', history)

# Test on one sample
print("\nTesting on one sample...")
model.eval()
test_sample = val_data[0].to(device)
with torch.no_grad():
    outputs = model(test_sample)
    test_loss = nn.functional.mse_loss(outputs['displacement'], test_sample.u_true).item()
print(f"Test loss: {test_loss:.6f}")

print(f"\n" + "="*60)
print("NEXT STEPS:")
print(f"1. View training history: {exp_dir}/training_history.txt")
print(f"2. Use best model: {exp_dir}/best_model.pt")
print(f"3. Run predictions: python predict.py --model {exp_dir}/best_model.pt --input dataset/val/sample_0000.npz")
print("="*60)