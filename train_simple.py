# train_simple.py - No config file needed
import os
import sys
import torch
import numpy as np
from datetime import datetime

print("="*60)
print("FEA GNN Training - Simple Version")
print("="*60)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import
try:
    from models.solid_gnn import SimpleSolidGNN
    print("✓ Model imported")
except ImportError as e:
    print(f"✗ Could not import model: {e}")
    sys.exit(1)

try:
    from utils.data_loader import load_npz_dataset
    print("✓ Data loader imported")
except ImportError as e:
    print(f"✗ Could not import data loader: {e}")
    sys.exit(1)

try:
    from utils.visualization import plot_training_history
    print("✓ Visualization imported")
except ImportError:
    print("⚠ Visualization not available, continuing without it...")
    def plot_training_history(history, save_path=None):
        print("Training history (no plot available):")
        if 'train_loss' in history:
            print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        if 'val_loss' in history:
            print(f"Final val loss: {history['val_loss'][-1]:.6f}")

# Check for data
print("\nChecking dataset...")
if not os.path.exists("dataset/train"):
    print("✗ dataset/train folder not found!")
    print("Please generate data first with your main.py")
    sys.exit(1)

train_files = [f for f in os.listdir("dataset/train") if f.endswith('.npz')]
val_files = [f for f in os.listdir("dataset/val") if f.endswith('.npz')] if os.path.exists("dataset/val") else []

if not train_files:
    print("✗ No .npz files found in dataset/train/")
    sys.exit(1)

print(f"Found {len(train_files)} training files")
print(f"Found {len(val_files)} validation files")

# Load a small sample to get dimensions
print("\nLoading sample data...")
try:
    from utils.data_loader import load_single_npz
    sample_file = os.path.join("dataset/train", train_files[0])
    sample_data = load_single_npz(sample_file)
    print(f"Sample loaded: {sample_data.num_nodes} nodes")
except Exception as e:
    print(f"⚠ Could not load sample: {e}")
    # Use defaults
    sample_data = type('obj', (object,), {'x': torch.randn(10, 5), 'edge_attr': torch.randn(20, 6)})

# Configuration (hardcoded)
config = {
    'experiment_name': 'fea_gnn_simple',
    'train_dir': 'dataset/train',
    'val_dir': 'dataset/val' if val_files else 'dataset/train',
    'max_train_samples': min(100, len(train_files)),
    'max_val_samples': min(20, len(val_files)) if val_files else 5,
    'batch_size': 2,
    'hidden_dim': 128,
    'num_layers': 3,
    'epochs': 10,  # Small for testing
    'learning_rate': 0.001,
    'patience': 5
}

print("\nConfiguration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Ask user to continue
response = input("\nStart training? (y/n): ")
if response.lower() != 'y':
    print("Training cancelled")
    sys.exit(0)

print("\n" + "="*60)
print("Starting training...")
print("="*60)

# Now you would add your training loop here
# For now, just show that we can proceed
print("\n✅ All checks passed!")
print("The system is ready for training.")
print("\nTo actually train, you need to:")
print("1. Make sure your train.py has the training loop code")
print("2. Or use the complete training script from earlier")
print("\nFor now, create the missing config file and run:")
print("python train.py --config configs/config.yaml")

# Create the config file automatically
os.makedirs("configs", exist_ok=True)
config_content = f"""experiment_name: "{config['experiment_name']}"
device: "cpu"
data:
  train_dir: "{config['train_dir']}"
  val_dir: "{config['val_dir']}"
  max_train_samples: {config['max_train_samples']}
  max_val_samples: {config['max_val_samples']}
  batch_size: {config['batch_size']}
model:
  hidden_dim: {config['hidden_dim']}
  num_layers: {config['num_layers']}
  dropout: 0.1
training:
  epochs: {config['epochs']}
  learning_rate: {config['learning_rate']}
  weight_decay: 0.0001
  patience: {config['patience']}
loss:
  physics_weight: 1.0
  data_weight: 1.0
  boundary_weight: 10.0
wandb:
  enabled: false"""

with open("configs/config.yaml", 'w') as f:
    f.write(config_content)

print(f"\nCreated config file: configs/config.yaml")
print("\nNow run: python train.py")