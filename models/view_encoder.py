import torch
import torch.nn as nn
from typing import List


class ViewEncoder(nn.Module):
    """
    Per-view MLP encoder (CRAFT default, shallow).

    Maps raw features to a unified representation space. Each view's encoder
    produces the same output dim, so they can be fed into ViewFusionTransformer
    for self-attention.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_bn: bool = True):
        super().__init__()

        layers = []
        in_d = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = hidden_dim

        # Output layer: no activation.
        layers.append(nn.Linear(in_d, output_dim))

        self.net = nn.Sequential(*layers)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepViewEncoder(nn.Module):
    """
    MFLVC/DCMVC-style 4-layer deep-wide encoder.

    Architecture: input_dim -> 500 -> ReLU -> 500 -> ReLU -> 2000 -> ReLU -> output_dim
    No BatchNorm, no Dropout (matches the original MFLVC/DCMVC implementation).

    Used for control experiments: tests whether encoder depth/width is the key
    factor for resolving mode C.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ViewDecoder(nn.Module):
    """Per-view MLP decoder (CRAFT default, shallow)."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_bn: bool = True):
        super().__init__()

        layers = []
        in_d = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = hidden_dim

        layers.append(nn.Linear(in_d, output_dim))

        self.net = nn.Sequential(*layers)
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DeepViewDecoder(nn.Module):
    """
    MFLVC/DCMVC-style 4-layer deep-wide decoder (symmetric to DeepViewEncoder).

    Architecture: output_dim -> 2000 -> ReLU -> 500 -> ReLU -> 500 -> ReLU -> input_dim
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def build_view_encoders(view_dims: List[int],
                        output_dim: int = 128,
                        hidden_dim: int = 256,
                        num_layers: int = 2,
                        dropout: float = 0.1,
                        encoder_type: str = 'shallow') -> nn.ModuleList:
    """
    Build encoders for all views.

    Args:
        view_dims: list of input dims, one per view
        output_dim: unified output dim
        hidden_dim: hidden layer dim (only used in shallow mode)
        num_layers: number of MLP layers (only used in shallow mode)
        dropout: dropout rate (only used in shallow mode)
        encoder_type: 'shallow' = CRAFT default 2-layer MLP
                      'deep' = MFLVC-style 4-layer 500-500-2000

    Returns:
        nn.ModuleList of ViewEncoder or DeepViewEncoder
    """
    if encoder_type == 'deep':
        print(f"[Encoder] Using MFLVC-style deep-wide architecture: input->500->500->2000->{output_dim}")
        encoders = nn.ModuleList([
            DeepViewEncoder(input_dim=d, output_dim=output_dim)
            for d in view_dims
        ])
    else:
        print(f"[Encoder] Using CRAFT shallow architecture: input->{hidden_dim}->{output_dim} ({num_layers} layers)")
        encoders = nn.ModuleList([
            ViewEncoder(
                input_dim=d,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
            for d in view_dims
        ])
    return encoders


def build_view_decoders(view_dims: List[int],
                        input_dim: int = 128,
                        hidden_dim: int = 256,
                        num_layers: int = 2,
                        dropout: float = 0.1,
                        encoder_type: str = 'shallow') -> nn.ModuleList:
    """
    Build decoders for all views (symmetric to encoders).

    Args:
        view_dims: list of original feature dims, one per view
        input_dim: encoded dim (embed_dim)
        hidden_dim: hidden layer dim (only used in shallow mode)
        num_layers: number of MLP layers (only used in shallow mode)
        dropout: dropout rate (only used in shallow mode)
        encoder_type: 'shallow' or 'deep', symmetric to encoder

    Returns:
        nn.ModuleList of ViewDecoder or DeepViewDecoder
    """
    if encoder_type == 'deep':
        print(f"[Decoder] Using MFLVC-style deep-wide architecture: {input_dim}->2000->500->500->output")
        decoders = nn.ModuleList([
            DeepViewDecoder(input_dim=input_dim, output_dim=d)
            for d in view_dims
        ])
    else:
        decoders = nn.ModuleList([
            ViewDecoder(
                input_dim=input_dim,
                output_dim=d,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
            for d in view_dims
        ])
    return decoders
