import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
from dataset_engineering import load_npz_dataset
# Import the NEW Spatial FNA model we just discussed
from milo_engineering_gnn import EngineeringGNN_SpatialFNA 

# --- CONFIG ---
CONFIG = {
    'npz_folder': 'data/my_meshes', 
    'batch_size': 4,
    'hidden_dim': 128,
    'epochs': 50,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # CRITICAL PARAMETER: Interaction Radius (in meters)
    # Start small (0.1m or 10cm). If your part is 1m long, try 0.2m.
    'radius': 0.2 
}

def train():
    print(f"Running on: {CONFIG['device']}")
    
    # 1. Load Data
    dataset = load_npz_dataset(CONFIG['npz_folder'], max_samples=100)
    train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 2. Initialize Spatial FNA Model
    model = EngineeringGNN_SpatialFNA(
        node_dim=12,   
        edge_dim=6,    
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=3,
        interaction_radius=CONFIG['radius'] # Pass the radius here
    ).to(CONFIG['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    print("\n>>> STARTING SPATIAL FNA TRAINING <<<")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        total_stress_error = 0
        
        # We use 'enumerate' so we can trigger the check on the first batch only
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(CONFIG['device'])
            
            # --- SANITY CHECK (First batch of every epoch) ---
            if batch_idx == 0:
                num_nodes = batch.x.size(0)
                # Calculate edges created by the radius graph
                # We can't see inside the forward() easily, so we just calculate it here to show you
                with torch.no_grad():
                    # This is just for printing, doesn't affect training
                    from torch_geometric.nn import radius_graph
                    check_edges = radius_graph(batch.pos, r=CONFIG['radius'], batch=batch.batch, loop=False)
                    num_edges = check_edges.size(1)
                    avg_neighbors = num_edges / num_nodes
                
                print(f"\n[Check] Batch Nodes: {num_nodes} | Radius Edges: {num_edges}")
                print(f"[Check] Avg Neighbors: {avg_neighbors:.1f} (Target: 20-50 is healthy)")
                
                if avg_neighbors > 200:
                    print("WARNING: Radius is too large! Memory might explode.")
                if avg_neighbors < 5:
                    print("WARNING: Radius is too small! Information won't flow.")
            # -------------------------------------------------

            optimizer.zero_grad()
            
            # Forward Pass
            out = model(batch)
            
            # Loss Calculation
            stress_true_pa = batch.stress_true
            # Clamp to avoid log(0)
            log_stress_true = torch.log(torch.max(stress_true_pa, torch.tensor(1.0, device=CONFIG['device'])))
            
            loss_stress = nn.MSELoss()(out['log_stress'], log_stress_true)
            loss_disp = nn.MSELoss()(out['displacement'], batch.u_true)
            
            # Weighted Loss
            loss = loss_stress + (100.0 * loss_disp)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                # Mean Absolute Error in MPa
                error_mpa = torch.abs(out['stress'] - stress_true_pa).mean() / 1e6
                total_stress_error += error_mpa.item()

        avg_loss = total_loss / len(train_loader)
        avg_stress_err = total_stress_error / len(train_loader)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | Stress Error: {avg_stress_err:.2f} MPa")

if __name__ == "__main__":
    train()