import torch
import torch.nn as nn


def cluster_entropy(cluster_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Global cluster-distribution entropy (prevents cluster collapse).

    H(p_bar) = -sum_c p_bar_c log(p_bar_c)

    Larger means the distribution is more uniform.
    """
    p_bar = cluster_probs.mean(dim=0)
    entropy = -torch.sum(p_bar * torch.log(p_bar + eps))
    return entropy


def batch_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean per-sample entropy within a batch.

    Encourages each sample's cluster assignment to be more confident
    (low entropy = high confidence).
    """
    sample_entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    return sample_entropy.mean()


class EntropyRegularization(nn.Module):
    """
    Entropy regularization loss.

    Total = -lambda_global * H(p_bar) + lambda_sample * mean(H(p_i))
    """

    def __init__(self,
                 global_weight: float = 1.0,
                 sample_weight: float = 0.0,
                 eps: float = 1e-8):
        super().__init__()
        self.global_weight = global_weight
        self.sample_weight = sample_weight
        self.eps = eps

    def forward(self, cluster_probs: torch.Tensor) -> dict:
        global_ent = cluster_entropy(cluster_probs, self.eps)
        sample_ent = batch_entropy(cluster_probs, self.eps)
        loss = -self.global_weight * global_ent + self.sample_weight * sample_ent

        return {
            'entropy_loss': loss,
            'global_entropy': global_ent,
            'sample_entropy': sample_ent
        }
