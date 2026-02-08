#!/usr/bin/env python3
"""
Inference script for trained model.
Run: python predict.py --model path/to/model.pt --input path/to/sample.npz
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.solid_gnn import SolidPINN_GNN
from utils.data_loader import load_single_npz
from utils.visualization import visualize_predictions, plot_mesh_deformation

def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model
    config = checkpoint['config']
    model = SolidPINN_GNN(
        node_dim=config['model']['node_dim'] if 'node_dim' in config['model'] else 5,
        edge_dim=config['model']['edge_dim'] if 'edge_dim' in config['model'] else 6,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def predict_single(model, npz_path, device='cpu'):
    """Make predictions for a single sample"""
    # Load data
    data = load_single_npz(npz_path)
    data = data.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(data)
    
    # Convert to numpy for analysis
    predictions = {
        'displacement': outputs['displacement'].cpu().numpy(),
        'stress': outputs['stress'].cpu().numpy(),
    }
    
    # Add ground truth if available
    if hasattr(data, 'u_true'):
        predictions['displacement_true'] = data.u_true.cpu().numpy()
    if hasattr(data, 'stress_true'):
        predictions['stress_true'] = data.stress_true.cpu().numpy()
    
    return predictions, data

def evaluate_predictions(predictions, data):
    """Calculate evaluation metrics"""
    metrics = {}
    
    if 'displacement_true' in predictions:
        u_pred = predictions['displacement']
        u_true = predictions['displacement_true']
        
        # MSE
        mse = np.mean((u_pred - u_true) ** 2)
        metrics['displacement_mse'] = mse
        
        # Relative error
        norm_true = np.sqrt(np.sum(u_true ** 2)) + 1e-8
        norm_diff = np.sqrt(np.sum((u_pred - u_true) ** 2))
        metrics['displacement_rel_error'] = norm_diff / norm_true
        
        # Max error
        metrics['displacement_max_error'] = np.max(np.abs(u_pred - u_true))
    
    if 'stress_true' in predictions:
        s_pred = predictions['stress']
        s_true = predictions['stress_true']
        
        metrics['stress_mse'] = np.mean((s_pred - s_true) ** 2)
        metrics['stress_rel_error'] = np.mean(np.abs(s_pred - s_true)) / (np.mean(np.abs(s_true)) + 1e-8)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input NPZ file')
    parser.add_argument('--output', type=str, default='predictions', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading model from: {args.model}")
    print(f"Processing input: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"Using device: {args.device}")
    
    # Load model
    model, checkpoint = load_model(args.model, args.device)
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Make predictions
    predictions, data = predict_single(model, args.input, args.device)
    
    # Calculate metrics
    metrics = evaluate_predictions(predictions, data)
    
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Plot displacement comparison
    if 'displacement_true' in predictions:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (title, field) in enumerate([
            ('Predicted Displacement (Z)', predictions['displacement'][:, 2]),
            ('Ground Truth (Z)', predictions['displacement_true'][:, 2]),
            ('Error', predictions['displacement'][:, 2] - predictions['displacement_true'][:, 2])
        ]):
            ax = axes[i]
            scatter = ax.scatter(data.pos[:, 0].cpu(), data.pos[:, 1].cpu(), 
                                c=field, cmap='viridis', s=10)
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'displacement_comparison.png', dpi=150)
        plt.show()
    
    # Plot stress
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Predicted stress
    ax1 = axes[0]
    scatter1 = ax1.scatter(data.pos[:, 0].cpu(), data.pos[:, 2].cpu(),
                          c=predictions['stress'].flatten(), cmap='hot', s=10)
    ax1.set_title('Predicted Stress')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    plt.colorbar(scatter1, ax=ax1, shrink=0.8, label='Stress')
    
    # True stress if available
    if 'stress_true' in predictions:
        ax2 = axes[1]
        scatter2 = ax2.scatter(data.pos[:, 0].cpu(), data.pos[:, 2].cpu(),
                              c=predictions['stress_true'].flatten(), cmap='hot', s=10)
        ax2.set_title('Ground Truth Stress')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        plt.colorbar(scatter2, ax=ax2, shrink=0.8, label='Stress')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stress_comparison.png', dpi=150)
    plt.show()
    
    # Save predictions to file
    np.savez(output_dir / 'predictions.npz',
             displacement_pred=predictions['displacement'],
             stress_pred=predictions['stress'],
             **{k: v for k, v in predictions.items() if 'true' in k})
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("Prediction Metrics\n")
        f.write("="*50 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.6f}\n")
    
    print(f"\nResults saved to: {output_dir}")
    print("Done!")

if __name__ == '__main__':
    main()