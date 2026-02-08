"""
Physics-informed loss functions for 3D continuum mechanics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsLossCalculator:
    def __init__(self, physics_weight=1.0, data_weight=1.0, boundary_weight=10.0):
        self.physics_weight = physics_weight
        self.data_weight = data_weight
        self.boundary_weight = boundary_weight
    
    def equilibrium_loss(self, batch, outputs):
        """
        Simplified equilibrium loss: f_int ≈ f_ext
        Using energy-based approach for efficiency
        """
        u_pred = outputs['displacement']
        f_ext = batch.f
        
        # Spring-like energy: Π = 0.5 * u^T * K * u - f^T * u
        # Simplified: assume K = I (identity) for efficiency
        energy = 0.5 * torch.sum(u_pred**2) - torch.sum(u_pred * f_ext)
        
        return torch.abs(energy) / batch.num_nodes
    
    def boundary_loss(self, batch, outputs):
        """Penalize displacement on fixed boundary nodes"""
        u_pred = outputs['displacement']
        
        # Boundary mask: 1 for fixed DOFs, 0 for free
        boundary_mask = batch.bc  # [N, 3]
        
        # Displacement should be zero on boundary
        loss = torch.sum((u_pred * boundary_mask)**2) / torch.sum(boundary_mask + 1e-8)
        
        return loss
    
    def constitutive_loss(self, batch, outputs):
        """Stress-strain constitutive relationship"""
        u_pred = outputs['displacement']
        
        # Material properties
        E = batch.material_params[0, 0]
        
        # Simplified strain calculation (average edge deformation)
        src, dst = batch.edge_index
        u_src = u_pred[src]
        u_dst = u_pred[dst]
        
        # Edge vectors
        pos_src = batch.pos[src]
        pos_dst = batch.pos[dst]
        edge_vec = pos_dst - pos_src
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
        
        # Normalized edge direction
        edge_dir = edge_vec / (edge_length + 1e-8)
        
        # Axial strain along edge
        du = u_dst - u_src
        axial_strain = torch.sum(du * edge_dir, dim=1, keepdim=True) / (edge_length + 1e-8)
        
        # Expected stress (simplified 1D)
        expected_stress = E * axial_strain
        
        # Predicted stress (average at nodes)
        stress_pred = outputs['stress']
        stress_src = stress_pred[src]
        stress_dst = stress_pred[dst]
        avg_stress = (stress_src + stress_dst) / 2
        
        loss = F.mse_loss(avg_stress, expected_stress)
        return loss
    
    def smoothness_loss(self, batch, outputs):
        """Encourage smooth displacement field"""
        u_pred = outputs['displacement']
        
        # Compare displacements of connected nodes
        src, dst = batch.edge_index
        u_src = u_pred[src]
        u_dst = u_pred[dst]
        
        loss = F.mse_loss(u_src, u_dst)
        return loss * 0.1  # Reduced weight
    
    def data_loss(self, batch, outputs):
        """Supervised loss if ground truth available"""
        losses = {}
        
        if hasattr(batch, 'u_true') and batch.u_true is not None:
            u_pred = outputs['displacement']
            losses['displacement'] = F.mse_loss(u_pred, batch.u_true)
        
        if hasattr(batch, 'stress_true') and batch.stress_true is not None:
            stress_pred = outputs['stress']
            losses['stress'] = F.mse_loss(stress_pred, batch.stress_true)
        
        return losses
    
    def compute_total_loss(self, batch, outputs, epoch=0):
        """Compute total loss with all components"""
        losses = {}
        
        # Physics-based losses
        losses['equilibrium'] = self.equilibrium_loss(batch, outputs)
        losses['boundary'] = self.boundary_loss(batch, outputs)
        losses['constitutive'] = self.constitutive_loss(batch, outputs)
        losses['smoothness'] = self.smoothness_loss(batch, outputs)
        
        # Data-based losses (if available)
        data_losses = self.data_loss(batch, outputs)
        losses.update(data_losses)
        
        # Weighted total loss
        weights = {
            'equilibrium': self.physics_weight,
            'boundary': self.boundary_weight,
            'constitutive': self.physics_weight * 0.5,
            'smoothness': 0.1,
            'displacement': self.data_weight,
            'stress': self.data_weight * 0.5
        }
        
        total_loss = torch.tensor(0.0, device=batch.x.device)
        for key, loss_val in losses.items():
            weight = weights.get(key, 1.0)
            total_loss = total_loss + weight * loss_val
        
        return total_loss, losses