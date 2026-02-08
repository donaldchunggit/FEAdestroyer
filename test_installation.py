#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly.
Run: python test_installation.py
"""

import sys
import os
import subprocess
import pkg_resources

REQUIRED_PACKAGES = [
    'torch',
    'torch_geometric',
    'numpy',
    'scipy',
    'pyvista',
    'matplotlib',
    'pandas',
    'pyyaml',
    'tqdm',
    'scikit-learn'
]

def check_package(package_name):
    """Check if a package is installed"""
    try:
        dist = pkg_resources.get_distribution(package_name)
        return True, dist.version
    except pkg_resources.DistributionNotFound:
        return False, None

def test_torch():
    """Test PyTorch installation"""
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    # Test basic tensor operations
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.mm(x, y)
    print(f"Tensor test: {x.shape} * {y.shape} = {z.shape} ✓")

def test_torch_geometric():
    """Test PyTorch Geometric installation"""
    from torch_geometric.data import Data
    import torch
    
    # Create a simple graph
    x = torch.tensor([[1], [2], [3]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    print(f"PyG Data object created: {data} ✓")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")

def test_numpy():
    """Test NumPy installation"""
    import numpy as np
    arr = np.random.randn(10, 3)
    print(f"NumPy test: Array shape {arr.shape} ✓")

def check_memory():
    """Check available memory"""
    import psutil
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / 1024**3:.1f} GB")
    print(f"Total memory: {memory.total / 1024**3:.1f} GB")
    
    if memory.available < 2 * 1024**3:  # Less than 2GB
        print("⚠️  Warning: Low memory available!")

def main():
    print("="*60)
    print("FEA-GNN Project Installation Test")
    print("="*60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check packages
    print("\nChecking required packages:")
    all_installed = True
    
    for package in REQUIRED_PACKAGES:
        installed, version = check_package(package)
        status = "✓" if installed else "✗"
        version_str = f" ({version})" if version else ""
        print(f"  {status} {package}{version_str}")
        
        if not installed:
            all_installed = False
    
    if not all_installed:
        print("\n❌ Some packages are missing!")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages are installed!")
    
    # Run individual tests
    print("\nRunning detailed tests:")
    
    try:
        test_torch()
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False
    
    try:
        test_torch_geometric()
    except Exception as e:
        print(f"❌ PyTorch Geometric test failed: {e}")
        return False
    
    try:
        test_numpy()
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        return False
    
    # Check memory
    try:
        check_memory()
    except:
        print("⚠️  Could not check memory (psutil not installed)")
    
    # Check project structure
    print("\nChecking project structure:")
    required_dirs = ['dataset', 'models', 'utils', 'configs']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
    
    # Check for data
    print("\nChecking for dataset:")
    if os.path.exists('dataset') and len(os.listdir('dataset')) > 0:
        print("  ✓ Dataset directory exists")
        
        # Check for NPZ files
        npz_files = [f for f in os.listdir('dataset') if f.endswith('.npz')]
        if npz_files:
            print(f"  ✓ Found {len(npz_files)} NPZ files")
        else:
            print("  ⚠️  No NPZ files found in dataset/")
    else:
        print("  ⚠️  Dataset directory not found or empty")
        print("     Generate data with: python main.py")
    
    print("\n" + "="*60)
    print("✅ Installation test completed successfully!")
    print("\nNext steps:")
    print("1. Generate dataset: python main.py")
    print("2. Split data: mkdir dataset/train dataset/val")
    print("3. Move 80% of .npz files to train, 20% to val")
    print("4. Train model: python train.py")
    print("="*60)
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)