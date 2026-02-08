# physics_engine.py - FIXED & IMPROVED VERSION
# Linear-elastic tetra FEM with correct units handling, robust BC selection,
# consistent stress computation (Von Mises in Pa), and NPZ export that includes
# a node_fixed mask for proper ML masking.

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Optional deps (only needed for visualization / reading VTK)
import meshio
try:
    import pyvista as pv
except Exception:
    pv = None


class PhysicsEngine:
    """
    Finite Element Physics Engine for 3D linear elasticity (tetrahedral mesh).

    Key fixes vs your version:
    - Unit-safe: treats E in Pa; can auto-convert if user passes MPa-scale values.
    - Robust boundary selection: uses % of beam length, not a hardcoded 0.1.
    - Stress is computed in Pa; visualization converts to MPa correctly.
    - Exports node_stresses as shape [N,1] and includes node_fixed mask [N,1].
    - Returns fixed_indices and tip_indices for debugging/metadata.
    """

    def __init__(self, youngs_modulus, poisson_ratio, *, assume_E_units="auto"):
        """
        Args:
            youngs_modulus: Young's modulus (preferably in Pa).
            poisson_ratio: Poisson's ratio (unitless).
            assume_E_units:
                - "auto": if E looks like MPa (e.g. 210000), convert to Pa
                - "pa": treat as Pa
                - "mpa": treat as MPa and convert to Pa
        """
        self.nu = float(poisson_ratio)

        E_in = float(youngs_modulus)
        if assume_E_units == "mpa":
            self.E = E_in * 1e6
        elif assume_E_units == "pa":
            self.E = E_in
        else:
            # Heuristic: typical structural E in Pa is ~1e9-1e12
            # If E is ~1e4-1e6, user likely passed MPa
            self.E = E_in * 1e6 if E_in < 1e8 else E_in

        if not (0.0 < self.nu < 0.49):
            raise ValueError(f"Poisson ratio looks invalid: nu={self.nu}")

        # Lame parameters
        self.lam = (self.E * self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu = self.E / (2.0 * (1.0 + self.nu))

    # -----------------------------
    # FEM core
    # -----------------------------
    def _get_tetra_stiffness(self, coords):
        """
        Computes the 12x12 element stiffness matrix for a linear tetrahedron.

        coords: (4,3) node coordinates for the element.
        """
        C = np.array(
            [
                [self.lam + 2 * self.mu, self.lam, self.lam, 0, 0, 0],
                [self.lam, self.lam + 2 * self.mu, self.lam, 0, 0, 0],
                [self.lam, self.lam, self.lam + 2 * self.mu, 0, 0, 0],
                [0, 0, 0, self.mu, 0, 0],
                [0, 0, 0, 0, self.mu, 0],
                [0, 0, 0, 0, 0, self.mu],
            ],
            dtype=np.float64,
        )

        M = np.ones((4, 4), dtype=np.float64)
        M[:, 1:] = coords.astype(np.float64)

        detM = np.linalg.det(M)
        V = np.abs(detM) / 6.0
        if V < 1e-18:
            # Degenerate element; return zeros to avoid blowing up assembly
            return np.zeros((12, 12), dtype=np.float64)

        inv_M = np.linalg.inv(M)
        B_sub = inv_M[1:, :].T  # (4,3)

        B = np.zeros((6, 12), dtype=np.float64)
        for i in range(4):
            # normal strains
            B[0, 3 * i] = B_sub[i, 0]
            B[1, 3 * i + 1] = B_sub[i, 1]
            B[2, 3 * i + 2] = B_sub[i, 2]
            # shear strains (engineering shear)
            B[3, 3 * i] = B_sub[i, 1]
            B[3, 3 * i + 1] = B_sub[i, 0]
            B[4, 3 * i + 1] = B_sub[i, 2]
            B[4, 3 * i + 2] = B_sub[i, 1]
            B[5, 3 * i] = B_sub[i, 2]
            B[5, 3 * i + 2] = B_sub[i, 0]

        Ke = (B.T @ C @ B) * V
        return Ke

    @staticmethod
    def _von_mises_from_sigma(sigma):
        """
        sigma: (3,3) Cauchy stress tensor.
        returns von Mises scalar.
        """
        sxx, syy, szz = sigma[0, 0], sigma[1, 1], sigma[2, 2]
        sxy, syz, sxz = sigma[0, 1], sigma[1, 2], sigma[0, 2]
        return np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + sxz**2)
        )

    def solve(
        self,
        vtk_file,
        force_vector,
        *,
        clamp_fraction=0.02,
        load_fraction=0.02,
        solver_eps=1e-12,
    ):
        """
        Performs FEA and returns:
            nodes: (N,3)
            disp: (N,3) in same length units as nodes (typically meters)
            stress_vm: (N,) von Mises stress in Pa
            elements: (E,4) tetra connectivity
            fixed_indices: (K,) indices of fixed nodes
            tip_indices: (T,) indices of load nodes

        Args:
            clamp_fraction: fraction of beam length used for fixed region at z_min
            load_fraction: fraction of beam length used for loaded region at z_max
        """
        mesh = meshio.read(vtk_file)
        if "tetra" not in mesh.cells_dict:
            raise ValueError("Mesh must contain tetra cells (mesh.cells_dict['tetra']).")

        nodes = np.asarray(mesh.points, dtype=np.float64)
        elements = np.asarray(mesh.cells_dict["tetra"], dtype=np.int32)
        num_nodes = nodes.shape[0]

        # Build global K and F
        K = lil_matrix((3 * num_nodes, 3 * num_nodes), dtype=np.float64)
        F = np.zeros(3 * num_nodes, dtype=np.float64)

        # 1) Assemble
        for cell in elements:
            Ke = self._get_tetra_stiffness(nodes[cell])
            if not np.any(Ke):
                continue
            dofs = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in cell], dtype=np.int64).flatten()
            # Add Ke into K
            for i in range(12):
                Ki = dofs[i]
                row = K.rows[Ki]
                data = K.data[Ki]
                for j in range(12):
                    Kj = dofs[j]
                    # lil_matrix supports direct add but this avoids overhead of slicing
                    # (still fine for your size; clarity > micro-optim)
                    K[Ki, Kj] += Ke[i, j]

        # 2) Boundary conditions and loads
        z_min = float(nodes[:, 2].min())
        z_max = float(nodes[:, 2].max())
        L = max(z_max - z_min, 1e-9)

        clamp_eps = max(clamp_fraction * L, 1e-9)
        load_eps = max(load_fraction * L, 1e-9)

        fixed_indices = np.where(nodes[:, 2] <= z_min + clamp_eps)[0]
        tip_indices = np.where(nodes[:, 2] >= z_max - load_eps)[0]

        if fixed_indices.size == 0:
            raise RuntimeError("No fixed nodes selected. Check clamp_fraction and mesh scale.")
        if tip_indices.size == 0:
            raise RuntimeError("No tip/load nodes selected. Check load_fraction and mesh scale.")

        force_vector = np.asarray(force_vector, dtype=np.float64).reshape(3)
        load_per_node = force_vector / float(len(tip_indices))

        for idx in tip_indices:
            F[3 * idx : 3 * idx + 3] += load_per_node

        # Apply Dirichlet BCs by row/col elimination
        for idx in fixed_indices:
            for d in range(3):
                rc = 3 * idx + d
                # Zero row & column, set diagonal to 1, RHS to 0
                K[rc, :] = 0.0
                K[:, rc] = 0.0
                K[rc, rc] = 1.0
                F[rc] = 0.0

        # 3) Solve
        K_csr = K.tocsr()
        # Small diagonal jitter can help if matrix ends up near-singular from mesh issues
        if solver_eps and solver_eps > 0:
            K_csr = K_csr + (solver_eps * lil_matrix(np.eye(K_csr.shape[0], dtype=np.float64))).tocsr()

        u_vec = spsolve(K_csr, F)
        disp = u_vec.reshape((num_nodes, 3))

        # 4) Stress recovery: compute per-element stress (constant strain), then nodal average
        stresses = np.zeros(num_nodes, dtype=np.float64)
        counts = np.zeros(num_nodes, dtype=np.float64)

        for cell in elements:
            coords = nodes[cell].astype(np.float64)

            M = np.ones((4, 4), dtype=np.float64)
            M[:, 1:] = coords
            detM = np.linalg.det(M)
            if abs(detM) < 1e-18:
                continue

            # grad_phi: (4,3) gradient of shape functions
            grad_phi = np.linalg.inv(M)[1:, :].T

            # Displacement gradient Du = sum_i u_i ⊗ ∇phi_i
            Du = disp[cell].T @ grad_phi  # (3,4)@(4,3)->(3,3)
            eps = 0.5 * (Du + Du.T)

            sigma = self.lam * np.trace(eps) * np.eye(3) + 2.0 * self.mu * eps
            vm = self._von_mises_from_sigma(sigma)

            stresses[cell] += vm
            counts[cell] += 1.0

        stress_vm = stresses / np.maximum(counts, 1.0)  # Pa

        return nodes, disp, stress_vm, elements, fixed_indices, tip_indices

    # -----------------------------
    # Visualization
    # -----------------------------
    def visualize_results(
        self,
        vtk_file,
        nodes,
        disp,
        stress_vm,
        force_vector,
        *,
        warp_factor=5.0,
        clamp_fraction=0.02,
        load_fraction=0.02,
    ):
        """Visual verification of stress and loading."""
        if pv is None:
            raise ImportError("pyvista is not available. Install pyvista to use visualization.")

        mesh = pv.read(vtk_file)

        # Stress is Pa internally; convert to MPa for visualization
        mesh.point_data["Stress_MPa"] = (np.asarray(stress_vm) / 1e6).astype(np.float32)
        mesh.point_data["Disp"] = np.asarray(disp, dtype=np.float32)

        warped = mesh.warp_by_vector("Disp", factor=float(warp_factor))

        z_min = float(nodes[:, 2].min())
        z_max = float(nodes[:, 2].max())
        L = max(z_max - z_min, 1e-9)
        clamp_eps = max(clamp_fraction * L, 1e-9)
        load_eps = max(load_fraction * L, 1e-9)

        tip_indices = np.where(nodes[:, 2] >= z_max - load_eps)[0]
        tip_center = nodes[tip_indices].mean(axis=0) if tip_indices.size else nodes.mean(axis=0)

        fv = np.asarray(force_vector, dtype=np.float64)
        fv_norm = np.linalg.norm(fv) + 1e-12
        arrow_dir = fv / fv_norm * (0.2 * L)  # scale arrow relative to geometry

        p = pv.Plotter()
        p.add_mesh_clip_plane(warped, scalars="Stress_MPa", cmap="jet", show_edges=True)
        p.add_arrows(cent=tip_center, direction=arrow_dir, mag=1.0, color="white")
        p.add_text(f"Load: {fv_norm:.1f} N", position="upper_right")
        p.show()

    # -----------------------------
    # Export
    # -----------------------------
    def export_npz(self, filename, nodes, elements, stress_vm, displacements, force_vector, fixed_indices=None):
        """
        Export for AI training with:
          - node_coords: [N,3]
          - connectivity: [E,4]
          - input_force: [3]
          - material_params: [2] (E, nu) in SI (Pa)
          - node_fixed: [N,1] (1 fixed, 0 free)
          - node_stresses: [N,1] von Mises stress in Pa
          - node_disp: [N,3]
        """
        nodes = np.asarray(nodes, dtype=np.float32)
        elements = np.asarray(elements, dtype=np.int32)
        force_vector = np.asarray(force_vector, dtype=np.float32).reshape(3)

        stress_vm = np.asarray(stress_vm, dtype=np.float32).reshape(-1, 1)
        displacements = np.asarray(displacements, dtype=np.float32)

        num_nodes = nodes.shape[0]
        node_fixed = np.zeros((num_nodes, 1), dtype=np.float32)
        if fixed_indices is not None:
            node_fixed[np.asarray(fixed_indices, dtype=np.int64)] = 1.0

        np.savez_compressed(
            filename,
            node_coords=nodes,
            connectivity=elements,
            input_force=force_vector,
            material_params=np.array([self.E, self.nu], dtype=np.float32),

            node_fixed=node_fixed,
            node_stresses=stress_vm,
            node_disp=displacements,
        )
