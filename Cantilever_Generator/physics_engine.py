import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import pyvista as pv
import meshio

class PhysicsEngine:
    def __init__(self, youngs_modulus, poisson_ratio):
        """Initializes with material properties. No hardcoded values."""
        self.E = youngs_modulus
        self.nu = poisson_ratio
        # Lame parameters for linear elasticity
        self.lam = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))

    def _get_tetra_stiffness(self, coords):
        """Calculates the 12x12 stiffness matrix for one tetrahedron."""
        C = np.array([
            [self.lam + 2*self.mu, self.lam, self.lam, 0, 0, 0],
            [self.lam, self.lam + 2*self.mu, self.lam, 0, 0, 0],
            [self.lam, self.lam, self.lam + 2*self.mu, 0, 0, 0],
            [0, 0, 0, self.mu, 0, 0],
            [0, 0, 0, 0, self.mu, 0],
            [0, 0, 0, 0, 0, self.mu]
        ])
        M = np.ones((4, 4))
        M[:, 1:] = coords
        V = np.abs(np.linalg.det(M)) / 6.0
        inv_M = np.linalg.inv(M)
        B_sub = inv_M[1:, :].T 
        B = np.zeros((6, 12))
        for i in range(4):
            B[0, 3*i] = B_sub[i, 0]; B[1, 3*i+1] = B_sub[i, 1]; B[2, 3*i+2] = B_sub[i, 2]
            B[3, 3*i] = B_sub[i, 1]; B[3, 3*i+1] = B_sub[i, 0]; B[4, 3*i+1] = B_sub[i, 2]
            B[4, 3*i+2] = B_sub[i, 1]; B[5, 3*i] = B_sub[i, 2]; B[5, 3*i+2] = B_sub[i, 0]
        return B.T @ C @ B * V

    def solve(self, vtk_file, force_vector):
        """Performs Finite Element Analysis (FEA) and returns simulation data."""
        mesh = meshio.read(vtk_file)
        nodes, elements = mesh.points, mesh.cells_dict["tetra"]
        num_nodes = len(nodes)
        K, F = lil_matrix((3 * num_nodes, 3 * num_nodes)), np.zeros(3 * num_nodes)

        # 1. Global Stiffness Assembly
        for cell in elements:
            Ke = self._get_tetra_stiffness(nodes[cell])
            dofs = np.array([[3*n, 3*n+1, 3*n+2] for n in cell]).flatten()
            for i in range(12):
                for j in range(12): K[dofs[i], dofs[j]] += Ke[i, j]

        # 2. Boundary Conditions (Z-min is wall, Z-max is tip)
        z_min, z_max = nodes[:, 2].min(), nodes[:, 2].max()
        fixed_indices = np.where(nodes[:, 2] < z_min + 0.1)[0]
        tip_indices = np.where(nodes[:, 2] > z_max - 0.1)[0]

        load_per_node = np.array(force_vector) / len(tip_indices)
        for idx in tip_indices:
            F[3*idx:3*idx+3] = load_per_node

        for idx in fixed_indices:
            for d in range(3):
                rc = 3 * idx + d
                K[rc, :], K[:, rc], K[rc, rc], F[rc] = 0, 0, 1, 0

        # 3. Sparse Solver
        u_vec = spsolve(K.tocsr(), F)
        disp = u_vec.reshape((num_nodes, 3))
        
        # 4. Stress Recovery (Tensor-based Von Mises)
        stresses, counts = np.zeros(num_nodes), np.zeros(num_nodes)
        for cell in elements:
            M = np.ones((4, 4)); M[:, 1:] = nodes[cell]
            grad_phi = np.linalg.inv(M)[1:, :].T
            Du = disp[cell].T @ grad_phi
            eps = 0.5 * (Du + Du.T)
            sigma = self.lam * np.trace(eps) * np.eye(3) + 2 * self.mu * eps
            vm = np.sqrt(0.5 * ((sigma[0,0]-sigma[1,1])**2 + (sigma[1,1]-sigma[2,2])**2 + 
                                (sigma[2,2]-sigma[0,0])**2 + 6*(sigma[0,1]**2 + sigma[1,2]**2 + sigma[0,2]**2)))
            stresses[cell] += vm; counts[cell] += 1

        return nodes, disp, stresses / np.maximum(counts, 1), elements

    def visualize_results(self, vtk_file, nodes, disp, stress, force_vector, warp_factor=5.0):
        """Visual verification of stress and loading."""
        mesh = pv.read(vtk_file)
        mesh.point_data["Stress_MPa"], mesh.point_data["Disp"] = stress, disp
        warped = mesh.warp_by_vector("Disp", factor=warp_factor)
        
        # Calculate force arrow origin
        tip_indices = np.where(nodes[:, 2] > nodes[:, 2].max() - 0.1)[0]
        tip_center = nodes[tip_indices].mean(axis=0)
        arrow_dir = np.array(force_vector) / np.linalg.norm(force_vector) * 20 

        p = pv.Plotter()
        p.add_mesh_clip_plane(warped, scalars="Stress_MPa", cmap="jet", show_edges=True)
        p.add_arrows(cent=tip_center, direction=arrow_dir, mag=1.0, color='white')
        p.add_point_labels([nodes[0]], ["FIXED WALL"], font_size=15, text_color="white")
        p.add_text(f"Load: {np.linalg.norm(force_vector):.1f} N", position='upper_right')
        p.show()

    def export_npz(self, filename, nodes, elements, stresses, displacements, force_vector):
        """
        Finalized Export for AI Training.
        Contains the full spatial graph and physics results.
        """
        np.savez_compressed(
            filename,
            # --- INPUT FEATURES ---
            node_coords=nodes.astype(np.float32),      # [N, 3] The (x,y,z) Positions
            connectivity=elements.astype(np.int32),    # [E, 4] How nodes form tetrahedrons
            input_force=np.array(force_vector, dtype=np.float32), # [3,] The Load
            material_params=np.array([self.E, self.nu], dtype=np.float32), # [2,] E and nu
            
            # --- TARGET LABELS (What the AI predicts) ---
            node_stresses=stresses.astype(np.float32), # [N, 1] Von Mises Stress
            node_disp=displacements.astype(np.float32) # [N, 3] Nodal movement
        )