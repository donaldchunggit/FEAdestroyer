# solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Node:
    x: float
    y: float
    fx: float = 0.0
    fy: float = 0.0
    fix_x: int = 0
    fix_y: int = 0


@dataclass
class Member:
    i: int
    j: int
    A: float  # cross-sectional area (m^2)


@dataclass
class TrussResult:
    stable: bool
    u: Optional[np.ndarray]                # (2N,) displacements [ux0, uy0, ux1, uy1, ...]
    member_forces: Optional[np.ndarray]    # (M,) axial forces (+tension, -compression)
    max_deflection: Optional[float]
    max_utilisation: Optional[float]
    governing_member: Optional[int]
    governing_mode: str                    # "stress" | "deflection" | "unstable"


def solve_truss_2d(
    nodes: List[Node],
    members: List[Member],
    E: float,
    sigma_allow: float,
    defl_limit: float,
    eps_sing: float = 1e-12,
) -> TrussResult:
    """
    Linear 2D truss solver using global stiffness matrix.
    Returns nodal displacements and member axial forces.
    """
    N = len(nodes)
    M = len(members)

    if N < 2 or M < 1:
        return TrussResult(False, None, None, None, None, None, "unstable")

    # DOF mapping: node k -> dof 2k (x), 2k+1 (y)
    ndof = 2 * N
    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros((ndof,), dtype=float)

    # Build load vector
    for k, n in enumerate(nodes):
        F[2 * k] = n.fx
        F[2 * k + 1] = n.fy

    # Assemble stiffness
    for m_idx, m in enumerate(members):
        i, j = m.i, m.j
        if not (0 <= i < N and 0 <= j < N) or i == j:
            return TrussResult(False, None, None, None, None, None, "unstable")

        xi, yi = nodes[i].x, nodes[i].y
        xj, yj = nodes[j].x, nodes[j].y
        dx, dy = (xj - xi), (yj - yi)
        L = float(np.hypot(dx, dy))
        if L < 1e-9:
            return TrussResult(False, None, None, None, None, None, "unstable")

        c = dx / L
        s = dy / L

        k_local = (E * m.A / L) * np.array(
            [
                [ c*c,  c*s, -c*c, -c*s],
                [ c*s,  s*s, -c*s, -s*s],
                [-c*c, -c*s,  c*c,  c*s],
                [-c*s, -s*s,  c*s,  s*s],
            ],
            dtype=float,
        )

        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += k_local[a, b]

    # Apply boundary conditions by eliminating fixed DOFs
    fixed = []
    for k, n in enumerate(nodes):
        if n.fix_x:
            fixed.append(2 * k)
        if n.fix_y:
            fixed.append(2 * k + 1)
    fixed = np.array(sorted(set(fixed)), dtype=int)

    if fixed.size == 0:
        # No supports => mechanism
        return TrussResult(False, None, None, None, None, None, "unstable")

    free = np.array([d for d in range(ndof) if d not in set(fixed.tolist())], dtype=int)
    if free.size == 0:
        return TrussResult(False, None, None, None, None, None, "unstable")

    Kff = K[np.ix_(free, free)]
    Ff = F[free]

    # Check singularity (mechanism / unstable)
    # Use condition number heuristic
    try:
        cond = np.linalg.cond(Kff)
        if not np.isfinite(cond) or cond > 1e14:
            return TrussResult(False, None, None, None, None, None, "unstable")
        uf = np.linalg.solve(Kff, Ff)
    except np.linalg.LinAlgError:
        return TrussResult(False, None, None, None, None, None, "unstable")

    u = np.zeros((ndof,), dtype=float)
    u[free] = uf
    u[fixed] = 0.0

    # Member axial forces
    member_forces = np.zeros((M,), dtype=float)
    utilisations = np.zeros((M,), dtype=float)

    for m_idx, m in enumerate(members):
        i, j = m.i, m.j
        xi, yi = nodes[i].x, nodes[i].y
        xj, yj = nodes[j].x, nodes[j].y
        dx, dy = (xj - xi), (yj - yi)
        L = float(np.hypot(dx, dy))
        c = dx / L
        s = dy / L

        # axial deformation = [-c -s c s] * [uix uiy ujx ujy]^T
        ue = np.array([u[2*i], u[2*i+1], u[2*j], u[2*j+1]], dtype=float)
        axial_strain = (1.0 / L) * np.dot(np.array([-c, -s, c, s], dtype=float), ue)
        axial_force = E * m.A * axial_strain  # N
        member_forces[m_idx] = axial_force

        # Stress utilisation (axial stress = N/A)
        stress = axial_force / m.A
        utilisations[m_idx] = abs(stress) / sigma_allow

    # Max deflection
    disp = u.reshape(N, 2)
    defl = np.sqrt(disp[:, 0] ** 2 + disp[:, 1] ** 2)
    max_deflection = float(defl.max())

    max_utilisation = float(utilisations.max())
    governing_member = int(utilisations.argmax())

    # Governing mode
    if max_utilisation > 1.0:
        mode = "stress"
    elif max_deflection > defl_limit:
        mode = "deflection"
    else:
        mode = "stress" if max_utilisation >= (max_deflection / max(defl_limit, eps_sing)) else "deflection"

    return TrussResult(
        stable=True,
        u=u,
        member_forces=member_forces,
        max_deflection=max_deflection,
        max_utilisation=max_utilisation,
        governing_member=governing_member,
        governing_mode=mode,
    )
