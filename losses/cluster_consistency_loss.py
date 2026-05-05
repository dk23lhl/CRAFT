import torch
import torch.nn as nn


class ClusterConsistencyLoss(nn.Module):
    """
    Cluster consistency loss.

    Formula: L = KL(p1 || sg(p2)) + KL(p2 || sg(p1))
    where sg() denotes stop-gradient.
    """

    def __init__(self,
                 symmetric: bool = True,
                 temperature: float = 1.0,
                 eps: float = 1e-8):
        super().__init__()
        self.symmetric = symmetric
        self.temperature = temperature
        self.eps = eps

    def sharpen(self, p: torch.Tensor, temperature: float) -> torch.Tensor:
        p_sharp = p ** (1.0 / temperature)
        return p_sharp / (p_sharp.sum(dim=-1, keepdim=True) + self.eps)

    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(p, min=self.eps)
        q = torch.clamp(q, min=self.eps)
        kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)
        return kl.mean()

    def forward(self,
                p1: torch.Tensor,
                p2: torch.Tensor,
                sharpen_target: bool = True) -> torch.Tensor:
        if sharpen_target and self.temperature != 1.0:
            p1_target = self.sharpen(p1.detach(), self.temperature)
            p2_target = self.sharpen(p2.detach(), self.temperature)
        else:
            p1_target = p1.detach()
            p2_target = p2.detach()

        loss1 = self.kl_divergence(p1, p2_target)

        if self.symmetric:
            loss2 = self.kl_divergence(p2, p1_target)
            loss = (loss1 + loss2) / 2.0
        else:
            loss = loss1

        return loss
