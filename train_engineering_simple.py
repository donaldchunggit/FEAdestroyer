# train_engineering_simple.py - FIXED VERSION
import os
import sys
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

print("="*80)
print("ENGINEERING FEA DESTROYER - SIMPLE VERSION")
print("="*80)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import
try:
    from models.engineering_gnn import EngineeringGNN
    print("✓ Engineering GNN imported")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

from utils.data_loader import load_npz_dataset
from torch_geometric.loader import DataLoader

# Configuration
config = {
    'experiment_name': 'engineering_v1',
    'seed': 42,
    'device': 'cpu',
    'train_dir': 'dataset/train',
    'val_dir': 'dataset/val',
    'batch_size': 2,
    'hidden_dim': 128,
    'num_layers': 3,
    'epochs': 20,
    'learning_rate': 0.001,
    'yield_stress': 250e6,  # Steel yield stress in Pa
    'min_safety_factor': 1.5
}

# Set seeds
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

# Create directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
exp_dir = Path(f"engineering_experiments/{config['experiment_name']}_{timestamp}")
exp_dir.mkdir(parents=True, exist_ok=True)

print(f"Experiment: {exp_dir}")
print(f"Device: {config['device']}")

# Load data
print("\nLoading data...")
train_data = load_npz_dataset(config['train_dir'], max_samples=80)
val_data = load_npz_dataset(config['val_dir'], max_samples=20)

print(f"Train: {len(train_data)} samples")
print(f"Val: {len(val_data)} samples")

if len(train_data) == 0:
    print("No training data!")
    sys.exit(1)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

# Create model
sample = train_data[0]
model = EngineeringGNN(
    node_dim=sample.x.shape[1],
    edge_dim=sample.edge_attr.shape[1],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers']
)

device = torch.device(config['device'])
model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Starting training for {config['epochs']} epochs...")

# Training loop
for epoch in range(1, config['epochs'] + 1):
    print(f"\nEpoch {epoch}/{config['epochs']}")
    
    # Training
    model.train()
    train_losses = []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        outputs = model(batch)
        
        # 1. Displacement loss (main objective)
        disp_loss = torch.nn.functional.mse_loss(outputs['displacement'], batch.u_true)
        
        # 2. Stress consistency (if stress_true available)
        stress_loss = torch.tensor(0.0, device=device)
        if hasattr(batch, 'stress_true'):
            stress_loss = torch.nn.functional.mse_loss(outputs['stress'], batch.stress_true)
        
        # 3. Engineering penalty: stress should not exceed yield
        stress_ratio = outputs['stress'] / config['yield_stress']
        yield_penalty = torch.mean(torch.relu(stress_ratio - 0.9)**2)  # Penalize >90% yield
        
        # 4. Safety factor penalty: should be > minimum
        safety_penalty = torch.mean(torch.relu(config['min_safety_factor'] - outputs['safety_factor'])**2)
        
        # Total loss with weights
        total_loss = (
            disp_loss + 
            0.5 * stress_loss + 
            5.0 * yield_penalty + 
            10.0 * safety_penalty
        )
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(total_loss.item())
        pbar.set_postfix({'loss': total_loss.item()})
    
    avg_train_loss = np.mean(train_losses)
    
    # Validation
    model.eval()
    val_losses = []
    engineering_stats = {
        'max_stress_ratio': [],
        'min_safety_factor': [],
        'displacement_error': []
    }
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            
            # Validation loss
            val_loss = torch.nn.functional.mse_loss(outputs['displacement'], batch.u_true)
            val_losses.append(val_loss.item())
            
            # Engineering statistics
            stress_ratio = outputs['stress'] / config['yield_stress']
            engineering_stats['max_stress_ratio'].append(stress_ratio.max().item())
            engineering_stats['min_safety_factor'].append(outputs['safety_factor'].min().item())
            
            # Displacement error in mm
            disp_error_mm = torch.mean(torch.abs(outputs['displacement'] - batch.u_true)) * 1000
            engineering_stats['displacement_error'].append(disp_error_mm.item())
    
    avg_val_loss = np.mean(val_losses)
    
    print(f"Train Loss: {avg_train_loss:.6f}")
    print(f"Val Loss: {avg_val_loss:.6f}")
    print(f"Max Stress/Yield: {np.mean(engineering_stats['max_stress_ratio']):.3f}")
    print(f"Min Safety Factor: {np.mean(engineering_stats['min_safety_factor']):.2f}")
    print(f"Avg Disp Error: {np.mean(engineering_stats['displacement_error']):.2f} mm")
    
    # Save checkpoint
    if epoch % 5 == 0 or epoch == config['epochs']:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'config': config
        }, exp_dir / f'model_epoch_{epoch}.pt')
        print(f"  ✓ Checkpoint saved")

print(f"\n" + "="*80)
print("TRAINING COMPLETE!")
print(f"Models saved in: {exp_dir}")
print("="*80)

# Final engineering validation
print("\nFinal engineering validation...")
model.eval()

with torch.no_grad():
    # Test on first validation sample
    test_sample = val_data[0].to(device)
    outputs = model(test_sample)
    
    print("\nEngineering Results:")
    print(f"Max displacement: {outputs['displacement'].abs().max().item()*1000:.2f} mm")
    print(f"Max stress: {outputs['stress'].max().item()/1e6:.1f} MPa")
    print(f"Yield stress: {config['yield_stress']/1e6:.1f} MPa")
    print(f"Stress/Yield ratio: {outputs['stress'].max().item()/config['yield_stress']:.3f}")
    print(f"Min safety factor: {outputs['safety_factor'].min().item():.2f}")
    
    # Engineering assessment
    max_stress_ratio = outputs['stress'].max().item() / config['yield_stress']
    min_safety = outputs['safety_factor'].min().item()
    
    if max_stress_ratio < 0.9 and min_safety > 1.5:
        print("\n✅ PASS: Meets engineering requirements!")
        print("  • Stress < 90% of yield ✓")
        print(f"  • Safety factor > 1.5 ({min_safety:.2f}) ✓")
    else:
        print("\n⚠️  WARNING: Check engineering requirements")
        if max_stress_ratio >= 0.9:
            print(f"  • Stress ratio: {max_stress_ratio:.3f} (should be < 0.9)")
        if min_safety <= 1.5:
            print(f"  • Safety factor: {min_safety:.2f} (should be > 1.5)")

print(f"\nTo use this model:")
print(f"python predict.py --model {exp_dir}/model_epoch_{config['epochs']}.pt --input dataset/val/sample_0000.npz")