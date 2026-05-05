"""
Representation alignment losses for stabilizing encoder training.

Two modes:
1. SimSiam style: negative cosine similarity + stop-gradient, O(n) complexity.
2. VICReg style: variance-invariance-covariance regularization with explicit
   anti-collapse; stable even without augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationAlignmentLoss(nn.Module):
    """
    SimSiam-style representation alignment.

    loss = -0.5 * (cos(z1, sg(z2)) + cos(sg(z1), z2))

    Complexity: O(n * d), each sample is independent.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: [B, d] fused representation from the first augmentation
            z2: [B, d] fused representation from the second augmentation

        Returns:
            loss: scalar; smaller means better aligned
        """
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)

        loss = -0.5 * (
            (z1_norm * z2_norm.detach()).sum(dim=-1).mean() +
            (z1_norm.detach() * z2_norm).sum(dim=-1).mean()
        )

        return loss


class VICRegLoss(nn.Module):
    """
    VICReg-style representation alignment.

    Three components:
    - Invariance: MSE(z1, z2); two augmentations of the same sample should match.
    - Variance: each dim must keep enough variance within a batch -- explicit
                anti-collapse.
    - Covariance: decorrelate different dims to prevent dim redundancy.

    Advantages: no stop-gradient or predictor needed; does not collapse even
    without augmentation. Complexity O(B * d), same order as SimSiam.

    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
               for Self-Supervised Learning", ICLR 2022.
    """

    def __init__(self, lambda_sim=25.0, lambda_var=25.0, lambda_cov=1.0):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: [B, D] fused representation from the first augmentation
            z2: [B, D] fused representation from the second augmentation

        Returns:
            loss: scalar
        """
        B, D = z1.shape

        sim_loss = F.mse_loss(z1, z2)

        # Variance: hinge loss penalizes any dim whose std falls below 1.
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = (torch.relu(1.0 - std_z1).mean()
                    + torch.relu(1.0 - std_z2).mean())

        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)
        cov_z1 = (z1_c.T @ z1_c) / (B - 1)
        cov_z2 = (z2_c.T @ z2_c) / (B - 1)
        # Penalize off-diagonal entries only; diagonals are variances handled by var_loss.
        off_diag_1 = cov_z1.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        off_diag_2 = cov_z2.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        cov_loss = (off_diag_1.pow(2).sum() / D
                    + off_diag_2.pow(2).sum() / D)

        return self.lambda_sim * sim_loss + self.lambda_var * var_loss + self.lambda_cov * cov_loss
