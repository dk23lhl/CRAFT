import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterHead(nn.Module):
    """Cluster head: maps representations to soft assignment probabilities over K clusters."""

    def __init__(self,
                 input_dim: int,
                 num_clusters: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 use_bn: bool = True,
                 temperature: float = 0.1):
        """
        Args:
            input_dim: input representation dim
            num_clusters: number of clusters K (= ground-truth class count of dataset)
            hidden_dim: hidden layer dim
            num_layers: number of MLP layers
            temperature: softmax temperature; smaller -> sharper distribution
        """
        super().__init__()

        self.num_clusters = num_clusters
        self.temperature = temperature

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_clusters))

        self.mlp = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Args:
            z: [B, input_dim]
            return_logits: also return logits

        Returns:
            p: [B, K] soft cluster assignment probabilities
        """
        logits = self.mlp(z)
        p = F.softmax(logits / self.temperature, dim=-1)

        if return_logits:
            return p, logits
        return p

    def get_cluster_assignment(self, z: torch.Tensor) -> torch.Tensor:
        """Hard cluster assignment (for evaluation)."""
        p = self.forward(z)
        return torch.argmax(p, dim=-1)
