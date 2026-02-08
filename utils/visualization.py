# utils/visualization.py - Fixed version
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, save_path=None):
    """Plot training and validation loss curves - FIXED VERSION"""
    try:
        plt.figure(figsize=(10, 4))
        
        # Plot train loss if available
        if 'train_loss' in history and history['train_loss']:
            plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        
        # Plot validation loss if available  
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Could not plot training history: {e}")
        # Save as text file instead
        if save_path:
            txt_path = str(save_path).replace('.png', '.txt')
            with open(txt_path, 'w') as f:
                f.write("Training History\n")
                f.write("="*50 + "\n")
                if 'train_loss' in history:
                    f.write("Train Loss:\n")
                    for i, loss in enumerate(history['train_loss']):
                        f.write(f"Epoch {i+1}: {loss:.6f}\n")
                if 'val_loss' in history:
                    f.write("\nValidation Loss:\n")
                    for i, loss in enumerate(history['val_loss']):
                        f.write(f"Epoch {i+1}: {loss:.6f}\n")
            print(f"History saved as text to: {txt_path}")