# generator_adv/adv_generator_fixed.py - FIXED VERSION
"""
Advanced Physics-Based FEA Data Generator
Creates realistic cantilever beam simulations with actual physics calculations
"""

import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.spatial import Delaunay
import warnings
warnings.filterwarnings('ignore')

class AdvancedBeamGenerator:
    """
    Generates realistic cantilever beam FEA data using analytical solutions
    and proper mesh generation with physics-based stress calculations
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        
        # Material database (realistic values)
        self.materials = {
            "Structural Steel": {
                "E": 200e9,
                "nu": 0.3,
                "yield_stress": 250e6,
                "density": 7850,
                "name": "Steel"
            },
            "High-Strength Steel": {
                "E": 210e9,
                "nu": 0.28,
                "yield_stress": 550e6,
                "density": 7850,
                "name": "HSS"
            },
            "Aluminum 6061-T6": {
                "E": 68.9e9,
                "nu": 0.33,
                "yield_stress": 276e6,
                "density": 2700,
                "name": "Al6061"
            },
            "Aluminum 7075-T6": {
                "E": 71.7e9,
                "nu": 0.33,
                "yield_stress": 503e6,
                "density": 2810,
                "name": "Al7075"
            },
            "Titanium Grade 5": {
                "E": 114e9,
                "nu": 0.34,
                "yield_stress": 880e6,
                "density": 4430,
                "name": "Ti6Al4V"
            },
            "Stainless Steel 304": {
                "E": 193e9,
                "nu": 0.29,
                "yield_stress": 215e6,
                "density": 8000,
                "name": "SS304"
            }
        }
        
        # Cross-section types
        self.sections = {
            "rectangle": self._create_rectangle_section,
            "i_beam": self._create_i_beam_section,
            "c_channel": self._create_c_channel_section,
            "tube": self._create_tube_section,
            "t_beam": self._create_t_beam_section
        }
    
    def _create_rectangle_section(self, width, height):
        width = max(width, 0.01)
        height = max(height, 0.01)
        area = width * height
        I_xx = (width * height**3) / 12
        I_yy = (height * width**3) / 12
        return {
            "type": "rectangle",
            "width": width,
            "height": height,
            "area": max(area, 1e-6),
            "I_xx": max(I_xx, 1e-12),
            "I_yy": max(I_yy, 1e-12),
            "centroid_y": height/2,
            "centroid_z": width/2
        }
    
    def _create_i_beam_section(self, width, height, flange_thickness=None, web_thickness=None):
        width = max(width, 0.01)
        height = max(height, 0.01)
        if flange_thickness is None:
            flange_thickness = max(height * 0.1, 0.005)
        if web_thickness is None:
            web_thickness = max(width * 0.1, 0.005)
        tf = flange_thickness
        tw = web_thickness
        h = height
        b = width
        area = 2 * (b * tf) + (h - 2*tf) * tw
        I_xx = (b * h**3 - (b - tw) * (h - 2*tf)**3) / 12
        I_yy = (2 * tf * b**3 + (h - 2*tf) * tw**3) / 12
        return {
            "type": "i_beam",
            "width": b,
            "height": h,
            "tf": tf,
            "tw": tw,
            "area": max(area, 1e-6),
            "I_xx": max(I_xx, 1e-12),
            "I_yy": max(I_yy, 1e-12),
            "centroid_y": h/2,
            "centroid_z": b/2
        }
    
    def _create_c_channel_section(self, width, height, flange_thickness=None, web_thickness=None):
        width = max(width, 0.01)
        height = max(height, 0.01)
        if flange_thickness is None:
            flange_thickness = max(height * 0.1, 0.005)
        if web_thickness is None:
            web_thickness = max(width * 0.1, 0.005)
        tf = flange_thickness
        tw = web_thickness
        h = height
        b = width
        area = b*tf + (h - tf)*tw + b*tf
        I_xx = (b * h**3 - (b - tw) * (h - 2*tf)**3) / 12
        return {
            "type": "c_channel",
            "width": b,
            "height": h,
            "tf": tf,
            "tw": tw,
            "area": max(area, 1e-6),
            "I_xx": max(I_xx, 1e-12),
            "centroid_y": h/2,
            "centroid_z": b/2
        }
    
    def _create_tube_section(self, width, height, wall_thickness=None):
        width = max(width, 0.01)
        height = max(height, 0.01)
        if wall_thickness is None:
            wall_thickness = max(min(width, height) * 0.1, 0.005)
        t = min(wall_thickness, min(width, height) * 0.4)
        b = width
        h = height
        b_inner = max(b - 2*t, 0.001)
        h_inner = max(h - 2*t, 0.001)
        area = b * h - b_inner * h_inner
        I_xx = (b * h**3 - b_inner * h_inner**3) / 12
        I_yy = (h * b**3 - h_inner * b_inner**3) / 12
        return {
            "type": "tube",
            "width": b,
            "height": h,
            "wall_thickness": t,
            "area": max(area, 1e-6),
            "I_xx": max(I_xx, 1e-12),
            "I_yy": max(I_yy, 1e-12),
            "centroid_y": h/2,
            "centroid_z": b/2
        }
    
    def _create_t_beam_section(self, width, height, flange_thickness=None, web_thickness=None):
        width = max(width, 0.01)
        height = max(height, 0.01)
        if flange_thickness is None:
            flange_thickness = max(height * 0.15, 0.005)
        if web_thickness is None:
            web_thickness = max(width * 0.15, 0.005)
        tf = flange_thickness
        tw = web_thickness
        h = height
        b = width
        A_flange = b * tf
        A_web = (h - tf) * tw
        area = A_flange + A_web
        y_flange = h - tf/2
        y_web = (h - tf)/2
        centroid_y = (A_flange * y_flange + A_web * y_web) / max(area, 1e-6)
        return {
            "type": "t_beam",
            "width": b,
            "height": h,
            "tf": tf,
            "tw": tw,
            "area": max(area, 1e-6),
            "centroid_y": centroid_y,
            "centroid_z": b/2
        }
    
    def generate_mesh(self, length, section, num_nodes=500):
        """Generate a 3D tetrahedral mesh for the cantilever beam"""
        nodes = []
        
        # Distribute nodes along length
        z_positions = np.random.uniform(0, length, num_nodes)
        cluster_mask = np.random.random(num_nodes) < 0.3
        if cluster_mask.sum() > 0:
            z_positions[cluster_mask] = np.random.uniform(0, length * 0.2, cluster_mask.sum())
        z_positions.sort()
        
        for z in z_positions:
            if section["type"] == "rectangle":
                x = np.random.uniform(-section["width"]/2, section["width"]/2)
                y = np.random.uniform(-section["height"]/2, section["height"]/2)
            elif section["type"] == "i_beam":
                attempts = 0
                while attempts < 100:
                    x = np.random.uniform(-section["width"]/2, section["width"]/2)
                    y = np.random.uniform(-section["height"]/2, section["height"]/2)
                    abs_y = abs(y)
                    if abs_y <= section["height"]/2:
                        if abs_y > section["height"]/2 - section["tf"]:
                            break
                        else:
                            if abs(x) <= section["tw"]/2:
                                break
                    attempts += 1
                else:
                    x = 0
                    y = 0
            elif section["type"] == "tube":
                attempts = 0
                while attempts < 100:
                    x = np.random.uniform(-section["width"]/2, section["width"]/2)
                    y = np.random.uniform(-section["height"]/2, section["height"]/2)
                    if abs(x) > section["width"]/2 - section["wall_thickness"] or \
                       abs(y) > section["height"]/2 - section["wall_thickness"]:
                        break
                    attempts += 1
                else:
                    x = section["width"]/2 - section["wall_thickness"]/2
                    y = 0
            else:
                x = np.random.uniform(-section["width"]/2, section["width"]/2)
                y = np.random.uniform(-section["height"]/2, section["height"]/2)
            nodes.append([x, y, z])
        
        nodes = np.array(nodes[:num_nodes])
        
        # Generate TETRAHEDRAL elements (4 nodes per element)
        elements = []
        
        # Simple approach: create tetrahedra by grouping nearby nodes
        # Sort nodes by z-coordinate for better connectivity
        z_indices = np.argsort(nodes[:, 2])
        
        # Create layers
        num_layers = min(20, len(nodes) // 10)
        layer_size = len(nodes) // num_layers
        
        for layer in range(num_layers - 1):
            start_idx = layer * layer_size
            end_idx = min((layer + 2) * layer_size, len(nodes))
            
            # Get nodes in this layer and next layer
            layer_nodes = z_indices[start_idx:end_idx]
            
            if len(layer_nodes) >= 8:
                # Create tetrahedra
                for i in range(0, len(layer_nodes) - 4, 4):
                    if i + 7 < len(layer_nodes):
                        # Create a hexahedron and split into 5 tetrahedra
                        n1, n2, n3, n4 = layer_nodes[i:i+4]
                        n5, n6, n7, n8 = layer_nodes[i+4:i+8]
                        
                        # Split into tetrahedra
                        elements.append([n1, n2, n3, n5])
                        elements.append([n2, n3, n5, n6])
                        elements.append([n3, n5, n6, n7])
                        elements.append([n3, n4, n7, n8])
                        elements.append([n3, n5, n7, n8])
        
        # Ensure we have at least some elements
        if len(elements) < 10:
            # Fallback: random tetrahedra
            for _ in range(100):
                tetra = np.random.choice(len(nodes), 4, replace=False)
                elements.append(tetra)
        
        elements = np.array(elements[:min(500, len(elements))])
        
        return nodes, elements
    
    def calculate_beam_theory(self, length, section, force_vector, material):
        """Calculate theoretical beam values"""
        E = material["E"]
        force_magnitude = max(np.linalg.norm(force_vector), 1.0)
        force_dir = force_vector / force_magnitude
        
        I_xx = section.get("I_xx", section["width"] * section["height"]**3 / 12)
        I_yy = section.get("I_yy", section["height"] * section["width"]**3 / 12)
        
        if abs(force_dir[1]) > abs(force_dir[0]):
            I = max(I_xx, 1e-12)
            c = section["height"] / 2
        else:
            I = max(I_yy, 1e-12)
            c = section["width"] / 2
        
        max_moment = force_magnitude * length
        max_stress = max_moment * c / I if I > 1e-12 else 1e6
        max_deflection = force_magnitude * length**3 / (3 * E * I) if I > 1e-12 else 0.01
        
        max_stress = np.clip(max_stress, 1e6, 1e9)
        max_deflection = np.clip(max_deflection, 1e-6, 0.1)
        
        return {
            "max_stress": max_stress,
            "max_deflection": max_deflection,
            "moment": max_moment,
            "I": I,
            "c": c
        }
    
    def compute_node_stresses(self, nodes, elements, force_vector, material, section, beam_theory):
        """Compute realistic stress distribution"""
        num_nodes = len(nodes)
        stresses = np.zeros((num_nodes, 1))
        
        length = nodes[:, 2].max() - nodes[:, 2].min()
        if length < 1e-6:
            length = 1.0
        
        clamped_z = nodes[:, 2].min()
        tip_z = nodes[:, 2].max()
        
        force_magnitude = max(np.linalg.norm(force_vector), 1.0)
        force_dir = force_vector / force_magnitude
        
        I = beam_theory["I"]
        c = beam_theory["c"]
        area = section["area"]
        
        for i in range(num_nodes):
            z = nodes[i, 2]
            x = nodes[i, 0]
            y = nodes[i, 1]
            
            t = (z - clamped_z) / (tip_z - clamped_z + 1e-8)
            t = np.clip(t, 0, 1)
            
            moment = force_magnitude * (tip_z - z)
            moment = max(moment, 0)
            
            if abs(force_dir[1]) > abs(force_dir[0]):
                pos_factor = y / (section["height"]/2 + 1e-8) if section["height"] > 0 else 0
            else:
                pos_factor = x / (section["width"]/2 + 1e-8) if section["width"] > 0 else 0
            
            pos_factor = np.clip(pos_factor, -1, 1)
            
            if I > 1e-12:
                bending_stress = moment * pos_factor * c / I
            else:
                bending_stress = 1e6 * (1 - t)
            
            if area > 1e-12:
                shear_stress = force_magnitude * 1.5 / area * (1 - pos_factor**2)
            else:
                shear_stress = bending_stress * 0.1
            
            stress_magnitude = np.sqrt(bending_stress**2 + 3 * shear_stress**2)
            stress_magnitude = abs(stress_magnitude)
            stress_magnitude *= np.random.uniform(0.9, 1.1)
            
            min_stress = 10e6
            max_stress = material["yield_stress"] * 1.2
            stress_magnitude = np.clip(stress_magnitude, min_stress, max_stress)
            
            stresses[i] = stress_magnitude
        
        return stresses
    
    def compute_node_displacements(self, nodes, force_vector, material, section, beam_theory):
        """Compute realistic displacement field"""
        num_nodes = len(nodes)
        displacements = np.zeros((num_nodes, 3))
        
        length = nodes[:, 2].max() - nodes[:, 2].min()
        if length < 1e-6:
            length = 1.0
        
        clamped_z = nodes[:, 2].min()
        tip_z = nodes[:, 2].max()
        
        force_magnitude = max(np.linalg.norm(force_vector), 1.0)
        force_dir = force_vector / force_magnitude
        
        I = beam_theory["I"]
        E = material["E"]
        
        for i in range(num_nodes):
            z = nodes[i, 2]
            t = (z - clamped_z) / (tip_z - clamped_z + 1e-8)
            t = np.clip(t, 0, 1)
            
            if I > 1e-12 and E > 1e-12:
                deflection_magnitude = (force_magnitude / (6 * E * I)) * (3 * length * z**2 - z**3)
            else:
                deflection_magnitude = 0.01 * (3 * t**2 - t**3)
            
            deflection_magnitude = max(abs(deflection_magnitude), 1e-8)
            displacements[i] = force_dir * deflection_magnitude
        
        return displacements
    
    def generate_sample(self, sample_id, output_dir="advanced_dataset"):
        """Generate a single complete FEA sample"""
        length = np.random.uniform(0.5, 2.0)
        width = np.random.uniform(0.05, 0.15) * length
        height = np.random.uniform(0.1, 0.2) * length
        
        section_type = np.random.choice(list(self.sections.keys()))
        
        if section_type == "rectangle":
            section = self._create_rectangle_section(width, height)
        elif section_type == "i_beam":
            section = self._create_i_beam_section(width, height)
        elif section_type == "c_channel":
            section = self._create_c_channel_section(width, height)
        elif section_type == "tube":
            section = self._create_tube_section(width, height)
        elif section_type == "t_beam":
            section = self._create_t_beam_section(width, height)
        
        material_name = np.random.choice(list(self.materials.keys()))
        material = self.materials[material_name]
        
        force_magnitude = np.random.uniform(5000, 50000)
        force_angle_x = np.random.uniform(-15, 15) * np.pi / 180
        force_angle_y = np.random.uniform(-15, 15) * np.pi / 180
        
        force_vector = np.array([
            force_magnitude * np.sin(force_angle_y) * np.cos(force_angle_x),
            force_magnitude * np.sin(force_angle_y) * np.sin(force_angle_x),
            0
        ])
        
        beam_theory = self.calculate_beam_theory(length, section, force_vector, material)
        
        num_nodes = np.random.randint(200, 500)
        nodes, elements = self.generate_mesh(length, section, num_nodes)
        
        stresses = self.compute_node_stresses(nodes, elements, force_vector, material, section, beam_theory)
        displacements = self.compute_node_displacements(nodes, force_vector, material, section, beam_theory)
        
        fixed_nodes = nodes[:, 2] < length * 0.05
        
        # Create node features
        node_features = np.column_stack([
            nodes,
            fixed_nodes.astype(float),
            (nodes[:, 2] > length * 0.9).astype(float),
            np.abs(nodes[:, 0]) / (width/2 + 1e-8),
            np.abs(nodes[:, 1]) / (height/2 + 1e-8),
        ])
        
        # Create edge indices and features
        edge_indices = []
        edge_features = []
        
        for elem in elements[:min(100, len(elements))]:
            for i in range(4):
                for j in range(i+1, 4):
                    if elem[i] < len(nodes) and elem[j] < len(nodes):
                        edge_indices.append([elem[i], elem[j]])
                        vec = nodes[elem[j]] - nodes[elem[i]]
                        edge_features.append(np.concatenate([vec, [np.linalg.norm(vec)]]))
        
        if len(edge_indices) == 0:
            edge_indices = np.array([[0, 1]])
            edge_features = np.array([[0, 0, 0, 1e-6]])
        else:
            edge_indices = np.array(edge_indices[:1000])
            edge_features = np.array(edge_features[:1000])
        
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{sample_id}.npz")
        
        np.savez_compressed(
            save_path,
            node_coords=nodes.astype(np.float32),
            connectivity=elements.astype(np.int32),  # Now [T, 4] as required
            edge_index=edge_indices.T.astype(np.int64),
            edge_attr=edge_features.astype(np.float32),
            input_force=force_vector.astype(np.float32),
            force_magnitude=np.array([force_magnitude], dtype=np.float32),
            force_direction=(force_vector / (force_magnitude + 1e-8)).astype(np.float32),
            material_params=np.array([material["E"], material["nu"]], dtype=np.float32),
            material_name=material["name"],
            yield_stress=np.array([material["yield_stress"]], dtype=np.float32),
            node_stresses=stresses.astype(np.float32),
            node_disp=displacements.astype(np.float32),
            fixed_nodes=fixed_nodes,
            section_type=section_type,
            section_props=np.array([
                section["width"],
                section["height"],
                section["area"],
                beam_theory["I"],
                beam_theory["c"]
            ], dtype=np.float32),
            length=np.array([length], dtype=np.float32),
            theoretical_max_stress=np.array([beam_theory["max_stress"]], dtype=np.float32),
            theoretical_max_disp=np.array([beam_theory["max_deflection"]], dtype=np.float32),
            node_features=node_features.astype(np.float32)
        )
        
        return {
            "sample_id": sample_id,
            "material": material_name,
            "section": section_type,
            "nodes": len(nodes),
            "elements": len(elements),
            "force_kN": force_magnitude/1000,
            "length_m": length,
            "max_stress_MPa": stresses.max()/1e6,
            "max_disp_mm": displacements.max() * 1000,
            "theoretical_stress_MPa": beam_theory["max_stress"]/1e6,
            "theoretical_disp_mm": beam_theory["max_deflection"] * 1000
        }


def generate_advanced_dataset(num_samples=100, output_dir="advanced_dataset", train_split=0.8):
    """Generate a complete dataset with train/val split"""
    generator = AdvancedBeamGenerator(seed=42)
    
    # Clean old directory if exists
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    metadata = []
    
    print(f"Generating {num_samples} advanced physics-based samples...")
    
    for i in range(num_samples):
        sample_id = f"sample_{i:04d}"
        
        if i < num_samples * train_split:
            sample_dir = train_dir
        else:
            sample_dir = val_dir
        
        try:
            sample_info = generator.generate_sample(sample_id, sample_dir)
            metadata.append(sample_info)
        except Exception as e:
            print(f"  Error generating sample {i}: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{num_samples} samples")
    
    if metadata:
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
        
        print(f"\nDataset generation complete!")
        print(f"  Train: {len(df[df.index < num_samples * train_split])} samples")
        print(f"  Val: {len(df[df.index >= num_samples * train_split])} samples")
        print(f"  Materials: {df['material'].nunique()}")
        print(f"  Sections: {df['section'].nunique()}")
        print(f"\nStress range: {df['max_stress_MPa'].min():.1f} - {df['max_stress_MPa'].max():.1f} MPa")
        print(f"Disp range: {df['max_disp_mm'].min():.2f} - {df['max_disp_mm'].max():.2f} mm")
        
        return df
    else:
        print("No samples generated successfully!")
        return None


if __name__ == "__main__":
    # Generate 100 samples
    df = generate_advanced_dataset(num_samples=100, output_dir="advanced_dataset")
    
    if df is not None:
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(df.groupby('material')['max_stress_MPa'].agg(['mean', 'std', 'count']))
        print("\n" + "="*60)
        print(df.groupby('section')['max_disp_mm'].agg(['mean', 'std', 'count']))