# test_model.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.loader import DataLoader
from models.engineering_gnn import EngineeringGNN
from utils.data_loader import load_npz_dataset
import sys

# Load your config (or copy the relevant parts)
class TestConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1  # Test one sample at a time
    VAL_DIR = "generator_adv/advanced_dataset/val"

def get_free_mask(batch):
    if hasattr(batch, "free_mask") and batch.free_mask is not None:
        return batch.free_mask.float()
    if hasattr(batch, "fixed_mask") and batch.fixed_mask is not None:
        return (1.0 - batch.fixed_mask.float())
    return None

def test_model():
    config = TestConfig()
    device = torch.device(config.DEVICE)
    
    # 1. Load the validation data
    print("Loading validation data...")
    val_data = load_npz_dataset(config.VAL_DIR, max_samples=None)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Loaded {len(val_data)} validation samples")
    
    # 2. Recreate the model architecture (must match training)
    sample = val_data[0]
    model = EngineeringGNN(
        node_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        hidden_dim=256,  # Must match your training config
        num_layers=5      # Must match your training config
    ).to(device)
    
    # 3. Load the trained weights
    checkpoint_path = "best_model_correlation.pt"
    if not Path(checkpoint_path).exists():
        print(f"Error: {checkpoint_path} not found!")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best correlation achieved: {checkpoint.get('correlation', 0):.3f}")
    
    # 4. Test on all validation samples
    model.eval()
    
    all_metrics = {
        'mae': [],
        'correlation': [],
        'max_error': [],
        'scale': [],
        'sample_idx': []  # Store sample indices
    }
    
    print("\n" + "="*60)
    print("TESTING ON VALIDATION SAMPLES")
    print("="*60)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = batch.to(device)
            out = model(batch)
            
            # Get predictions and ground truth
            u_pred = out["displacement"].cpu().numpy()
            u_true = batch.u_true.cpu().numpy()
            
            # Apply free mask if available
            free_mask = get_free_mask(batch)
            if free_mask is not None:
                mask = free_mask.cpu().numpy().squeeze() > 0.5
                u_pred = u_pred[mask]
                u_true = u_true[mask]
            
            if len(u_pred) == 0:
                continue
            
            # Calculate metrics
            error = u_pred - u_true
            mae = np.mean(np.abs(error)) * 1000  # Convert to mm
            max_error = np.max(np.abs(error)) * 1000  # mm
            
            # Correlation (flatten all dimensions)
            u_true_flat = u_true.flatten()
            u_pred_flat = u_pred.flatten()
            corr = np.corrcoef(u_true_flat, u_pred_flat)[0, 1]
            
            all_metrics['mae'].append(mae)
            all_metrics['correlation'].append(corr)
            all_metrics['max_error'].append(max_error)
            all_metrics['scale'].append(out["disp_scale"].item())
            all_metrics['sample_idx'].append(i)
            
            # Print first 5 samples in detail
            if i < 5:
                print(f"\nSample {i}:")
                print(f"  MAE: {mae:.3f} mm")
                print(f"  Max Error: {max_error:.3f} mm")
                print(f"  Correlation: {corr:.3f}")
                print(f"  Scale: {out['disp_scale'].item():.6f}")
                print(f"  True mean: {np.mean(np.abs(u_true))*1000:.3f} mm")
                print(f"  Pred mean: {np.mean(np.abs(u_pred))*1000:.3f} mm")
    
    # 5. Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Average MAE: {np.mean(all_metrics['mae']):.3f} ± {np.std(all_metrics['mae']):.3f} mm")
    print(f"Average Max Error: {np.mean(all_metrics['max_error']):.3f} ± {np.std(all_metrics['max_error']):.3f} mm")
    print(f"Average Correlation: {np.mean(all_metrics['correlation']):.3f} ± {np.std(all_metrics['correlation']):.3f}")
    print(f"Average Scale: {np.mean(all_metrics['scale']):.6f}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    
    # Call analyze_worst_case with the loaded data to avoid reloading
    analyze_worst_case(model, val_data, device)

def analyze_worst_case(model=None, val_data=None, device=None):
    """Find and analyze the worst-performing sample"""
    config = TestConfig()
    if device is None:
        device = torch.device(config.DEVICE)
    
    # Load data if not provided
    if val_data is None:
        print("Loading validation data for worst-case analysis...")
        val_data = load_npz_dataset(config.VAL_DIR, max_samples=None)
    
    # Load model if not provided
    if model is None:
        model = EngineeringGNN(
            node_dim=val_data[0].x.shape[1],
            edge_dim=val_data[0].edge_attr.shape[1],
            hidden_dim=256,
            num_layers=5
        ).to(device)
        
        checkpoint = torch.load("best_model_correlation.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    # Create a DataLoader with batch_size=1
    from torch_geometric.loader import DataLoader
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    # Find worst sample
    worst_mae = -1
    worst_idx = -1
    worst_pred = None
    worst_true = None
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = batch.to(device)
            out = model(batch)
            
            u_pred = out["displacement"].cpu().numpy()
            u_true = batch.u_true.cpu().numpy()
            
            free_mask = get_free_mask(batch)
            if free_mask is not None:
                mask = free_mask.cpu().numpy().squeeze() > 0.5
                u_pred = u_pred[mask]
                u_true = u_true[mask]
            
            if len(u_pred) > 0:
                mae = np.mean(np.abs(u_pred - u_true)) * 1000
                if mae > worst_mae:
                    worst_mae = mae
                    worst_idx = i
                    worst_pred = u_pred
                    worst_true = u_true
    
    if worst_idx >= 0:
        print(f"\n{'='*60}")
        print("WORST-CASE ANALYSIS")
        print(f"{'='*60}")
        print(f"Worst sample (index {worst_idx}):")
        print(f"  MAE: {worst_mae:.3f} mm")
        print(f"  Error statistics:")
        error = worst_pred - worst_true
        for dim in range(3):
            print(f"    {['X','Y','Z'][dim]}: mean={np.mean(error[:,dim])*1000:.3f}mm, "
                  f"std={np.std(error[:,dim])*1000:.3f}mm")
        
        # Calculate correlation for worst sample
        u_true_flat = worst_true.flatten()
        u_pred_flat = worst_pred.flatten()
        corr = np.corrcoef(u_true_flat, u_pred_flat)[0, 1]
        print(f"  Correlation: {corr:.3f}")
    else:
        print("\nNo valid samples found for worst-case analysis")

if __name__ == "__main__":
    test_model()