import numpy as np
import os
import pandas as pd # For metadata concatenation
import cantilever_nodes 
from physics_engine import PhysicsEngine

# --- CONFIGURATION ---
NODE_THRESHOLD = 1200  
TARGET_SAMPLES = 1000
VISUALIZE_EVERY = 1000
MATERIAL_DATA = {
    "Steel": {"E": 210000, "nu": 0.3},
    "Aluminum": {"E": 68900, "nu": 0.33},
    "Titanium": {"E": 114000, "nu": 0.34}
}

def generate_random_force(min_mag=100, max_mag=2000):
    """Generates a random 3D force vector."""
    magnitude = np.random.uniform(min_mag, max_mag)
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi)
    return [
        magnitude * np.sin(theta) * np.cos(phi),
        magnitude * np.sin(theta) * np.sin(phi),
        magnitude * np.cos(theta)
    ]

# Setup Output
os.makedirs("dataset", exist_ok=True)
metadata_list = [] # To store summary data for your friend

# --- DATA GENERATION LOOP ---
samples_generated = 0
attempts = 0

print(f"Starting batch generation of {TARGET_SAMPLES} samples...")

while samples_generated < TARGET_SAMPLES:
    attempts += 1
    
    # 1. Constrained Geometry (Clamped to stay under 1500 nodes)
    width = np.random.uniform(8, 15)
    height = np.random.uniform(8, 15)
    length = np.random.uniform(max(width, height) * 10, 150)
    
    factory = cantilever_nodes.BeamFactory(mesh_size=4.0)
    
    # 2. Shape Selection (Cycling through all types)
    shape_choice = np.random.choice(["I", "T", "Tube", "Box"])
    
    if shape_choice == "I":
        factory.create_i_beam(L=length, w=width, h=height, t_f=width/5, t_w=width/7)
    elif shape_choice == "T":
        factory.create_t_beam(L=length, w=width, h=height, t_f=width/5, t_w=width/7)
    elif shape_choice == "Box":
        factory.create_hollow_box(L=length, w=width, h=height, t=2.0)
    elif shape_choice == "Tube":
        factory.create_circular_tube(L=length, r_out=width/2, r_in=(width/2)-2.0)

    num_nodes = factory.get_node_count() 
    
    if num_nodes > NODE_THRESHOLD:
        continue # Discard and retry silently

    # 3. Physics Solve
    mat_name = np.random.choice(list(MATERIAL_DATA.keys()))
    mat = MATERIAL_DATA[mat_name]
    engine = PhysicsEngine(youngs_modulus=mat["E"], poisson_ratio=mat["nu"])
    force = generate_random_force()
    
    nodes, disp, stress, elements = engine.solve("current_beam.vtk", force)
    
    # 4. Conditional Visualization
    if samples_generated % VISUALIZE_EVERY == 0:
        print(f"Visualizing Sample {samples_generated}...")
        engine.visualize_results("current_beam.vtk", nodes, disp, stress, force)
    
    # 5. Export and Logging
    sample_id = f"sample_{samples_generated:04d}"
    save_path = f"dataset/{sample_id}.npz"
    engine.export_npz(save_path, nodes, elements, stress, disp, force)
    
    # Keep track of metadata for the summary file
    metadata_list.append({
        "sample_id": sample_id,
        "shape": shape_choice,
        "material": mat_name,
        "nodes": num_nodes,
        "force_magnitude": np.linalg.norm(force),
        "length": length
    })
    
    samples_generated += 1
    if samples_generated % 50 == 0:
        print(f"Progress: {samples_generated}/{TARGET_SAMPLES} samples complete.")

# 6. Final Concatenation (The "Nice" Part for your friend)
df = pd.DataFrame(metadata_list)
df.to_csv("dataset/master_metadata.csv", index=False)

print(f"\nSuccess! 1000 samples generated in 'dataset/' folder.")
print("Master metadata saved as 'dataset/master_metadata.csv'.")