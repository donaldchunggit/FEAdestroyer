import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
NODE_THRESHOLD = 1200  
TARGET_SAMPLES = 100  # Reduced from 1000 for testing
VISUALIZE_EVERY = 1000  # Won't visualize during testing
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

def create_dummy_mesh_data(length, width, height, shape_choice):
    """Create dummy mesh data without gmsh/pyvista."""
    # Generate random number of nodes
    base_nodes = 200
    if shape_choice == "I":
        num_nodes = base_nodes + np.random.randint(-50, 100)
    elif shape_choice == "T":
        num_nodes = base_nodes + np.random.randint(-50, 50)
    elif shape_choice == "Box":
        num_nodes = base_nodes + np.random.randint(-30, 70)
    elif shape_choice == "Tube":
        num_nodes = base_nodes + np.random.randint(-40, 60)
    
    # Ensure node count is reasonable
    num_nodes = min(num_nodes, NODE_THRESHOLD - 100)
    num_nodes = max(num_nodes, 50)
    
    # Create node coordinates (simulating a cantilever beam)
    nodes = np.zeros((num_nodes, 3), dtype=np.float32)
    
    # Distribute nodes along the beam
    for i in range(num_nodes):
        # Random position in beam volume
        x = np.random.uniform(-width/2, width/2)
        y = np.random.uniform(-height/2, height/2)
        z = np.random.uniform(0, length)
        nodes[i] = [x, y, z]
    
    # Sort by z for more realistic distribution
    nodes = nodes[nodes[:, 2].argsort()]
    
    # Create tetrahedral connectivity (simplified)
    num_elements = max(100, num_nodes // 2)
    elements = np.zeros((num_elements, 4), dtype=np.int32)
    
    for i in range(num_elements):
        # Random tetrahedron connecting nearby nodes
        base = np.random.randint(0, num_nodes - 10)
        elements[i] = [
            base,
            base + np.random.randint(1, 4),
            base + np.random.randint(4, 7),
            base + np.random.randint(7, 10)
        ]
    
    return nodes, elements, num_nodes

def solve_fake_fea(nodes, elements, E, nu, force_vector):
    """Generate realistic-looking FEA results without actual solver."""
    num_nodes = len(nodes)
    
    # Generate displacements that look realistic for a cantilever
    displacements = np.zeros((num_nodes, 3), dtype=np.float32)
    
    # Tip displacement (largest at free end)
    tip_z = nodes[:, 2].max()
    clamped_z = nodes[:, 2].min()
    
    for i in range(num_nodes):
        z = nodes[i, 2]
        # Normalized position along beam (0 at clamp, 1 at tip)
        t = (z - clamped_z) / (tip_z - clamped_z + 1e-8)
        
        # Cubic beam deflection formula (simplified)
        # Real cantilever: displacement ∝ (3Lz² - z³) for point load at tip
        displacement_magnitude = (3 * t**2 - t**3) * 0.01
        
        # Direction based on force
        force_dir = force_vector / (np.linalg.norm(force_vector) + 1e-8)
        displacements[i] = force_dir * displacement_magnitude
        
        # Add some random variation
        displacements[i] += np.random.randn(3) * 0.001
    
    # Generate stresses (higher near clamp, tip)
    stresses = np.zeros((num_nodes, 1), dtype=np.float32)
    
    for i in range(num_nodes):
        z = nodes[i, 2]
        t = (z - clamped_z) / (tip_z - clamped_z + 1e-8)
        
        # Stress is higher near clamp (t=0) for cantilever
        base_stress = 100 * E / 210000  # Scale with Young's modulus
        
        # Stress distribution: higher near clamp, lower at tip
        stress_value = base_stress * (1 - 0.8 * t)
        
        # Add element-based variation
        element_stress = 0
        element_count = 0
        for elem in elements:
            if i in elem:
                # Random stress contribution from each element
                element_stress += np.random.uniform(0.8, 1.2)
                element_count += 1
        
        if element_count > 0:
            stress_value *= element_stress / element_count
        
        # Add some noise
        stress_value *= np.random.uniform(0.9, 1.1)
        stresses[i] = max(0.1, stress_value)  # Ensure positive
    
    return displacements, stresses

class SimplePhysicsEngine:
    """Simplified physics engine without meshio/pyvista."""
    def __init__(self, youngs_modulus, poisson_ratio):
        self.E = youngs_modulus
        self.nu = poisson_ratio
    
    def solve(self, dummy_vtk_file, force_vector):
        """Dummy solve that returns realistic data."""
        # We'll create the mesh data directly
        return None, None, None, None  # Placeholders
    
    def export_npz(self, filename, nodes, elements, stresses, displacements, force_vector):
        """Export data in the same format as before."""
        np.savez_compressed(
            filename,
            node_coords=nodes.astype(np.float32),
            connectivity=elements.astype(np.int32),
            input_force=np.array(force_vector, dtype=np.float32),
            material_params=np.array([self.E, self.nu], dtype=np.float32),
            node_stresses=stresses.astype(np.float32),
            node_disp=displacements.astype(np.float32)
        )

# Setup Output
os.makedirs("dataset", exist_ok=True)
metadata_list = []

# --- DATA GENERATION LOOP ---
samples_generated = 0
attempts = 0

print(f"Starting batch generation of {TARGET_SAMPLES} samples...")

while samples_generated < TARGET_SAMPLES:
    attempts += 1
    
    # 1. Constrained Geometry
    width = np.random.uniform(8, 15)
    height = np.random.uniform(8, 15)
    length = np.random.uniform(max(width, height) * 10, 150)
    
    # 2. Shape Selection
    shape_choice = np.random.choice(["I", "T", "Tube", "Box"])
    
    # Create mesh data directly (no gmsh)
    nodes, elements, num_nodes = create_dummy_mesh_data(length, width, height, shape_choice)
    
    if num_nodes > NODE_THRESHOLD:
        continue  # Discard and retry

    # 3. Physics Solve (fake)
    mat_name = np.random.choice(list(MATERIAL_DATA.keys()))
    mat = MATERIAL_DATA[mat_name]
    engine = SimplePhysicsEngine(youngs_modulus=mat["E"], poisson_ratio=mat["nu"])
    force = generate_random_force()
    
    # Generate displacements and stresses
    disp, stress = solve_fake_fea(nodes, elements, mat["E"], mat["nu"], force)
    
    # 4. Skip visualization for now
    # if samples_generated % VISUALIZE_EVERY == 0:
    #     print(f"Visualizing Sample {samples_generated}...")
    #     # Skip visualization
    
    # 5. Export and Logging
    sample_id = f"sample_{samples_generated:04d}"
    save_path = f"dataset/{sample_id}.npz"
    engine.export_npz(save_path, nodes, elements, stress, disp, force)
    
    # Keep track of metadata
    metadata_list.append({
        "sample_id": sample_id,
        "shape": shape_choice,
        "material": mat_name,
        "nodes": num_nodes,
        "force_magnitude": np.linalg.norm(force),
        "length": length,
        "width": width,
        "height": height
    })
    
    samples_generated += 1
    if samples_generated % 10 == 0:
        print(f"Progress: {samples_generated}/{TARGET_SAMPLES} samples complete.")
    
    # Safety check
    if attempts > TARGET_SAMPLES * 3:
        print(f"Warning: Too many attempts ({attempts}), stopping early.")
        break

# 6. Save metadata
if metadata_list:
    df = pd.DataFrame(metadata_list)
    df.to_csv("dataset/master_metadata.csv", index=False)
    
    print(f"\nSuccess! {samples_generated} samples generated in 'dataset/' folder.")
    print(f"Master metadata saved as 'dataset/master_metadata.csv'.")
    
    # Create train/val split for GNN
    os.makedirs("dataset/train", exist_ok=True)
    os.makedirs("dataset/val", exist_ok=True)
    
    files = [f for f in os.listdir("dataset") if f.endswith('.npz')]
    np.random.shuffle(files)
    
    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    for f in train_files:
        os.rename(f"dataset/{f}", f"dataset/train/{f}")
    
    for f in val_files:
        os.rename(f"dataset/{f}", f"dataset/val/{f}")
    
    print(f"Split into {len(train_files)} training and {len(val_files)} validation samples.")
else:
    print("\nNo samples were generated. Check the code for issues.")