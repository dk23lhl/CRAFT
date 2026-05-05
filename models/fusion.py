"""
CRAFT Multi-View Fusion Module

ViewFusionTransformer: self-attention based multi-view fusion.
- Each view encoding becomes a token -> [B, V, d]
- A CLS token is prepended -> [B, V+1, d]
- Self-attention lets all views see each other
- The CLS token output serves as the fused representation

Missing-view strategies:
- 'learnable': replace missing views with a learnable mask_token
- 'zero': replace missing views with zero vectors
- 'mean': replace missing views with the mean of available views in the batch
- 'attention_mask': missing views are excluded from attention via key_padding_mask
  (recommended; "train-once, infer-anywhere", same mechanism BERT uses for padding)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from collections import OrderedDict


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Self-attention block with pre-norm"""

    def __init__(self, d_model: int, n_head: int,
                 dropout: float = 0.):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        h = self.ln_1(x)
        attn_out, attn_weights = self.attn(
            h, h, h,
            need_weights=return_attention,
            key_padding_mask=key_padding_mask,
            average_attn_weights=True,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if return_attention:
            return x, attn_weights
        return x


class ViewFusionTransformer(nn.Module):
    """
    Self-attention based multi-view fusion module.

    Missing-view handling (four strategies):
    - 'learnable': replace missing views with a learnable mask_token
    - 'zero': replace missing views with zero vectors
    - 'mean': replace missing views with the mean of available views in the batch
    - 'attention_mask': missing views are excluded from attention via key_padding_mask;
      no fill value is needed. At training time without missingness, no mask is passed
      and the model trains normally; at inference with missing views, the mask is
      passed and attention skips missing positions automatically -- "train-once,
      infer-anywhere".
    """

    def __init__(self,
                 embed_dim: int = 128,
                 num_views: int = 6,
                 n_heads: int = 4,
                 n_layers: int = 1,
                 dropout: float = 0.1,
                 mask_strategy: str = 'learnable',
                 aggregation: str = 'cls'):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_views = num_views
        self.mask_strategy = mask_strategy
        self.aggregation = aggregation

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.view_embed = nn.Parameter(torch.randn(1, num_views + 1, embed_dim))

        # mask_token is always allocated so random-init state stays consistent across
        # mask strategies; only used in forward when mask_strategy='learnable'.
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._initialize()

    def _initialize(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.view_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        proj_std = (self.embed_dim ** -0.5) * ((2 * len(self.blocks)) ** -0.5)
        attn_std = self.embed_dim ** -0.5
        fc_std = (2 * self.embed_dim) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    @property
    def width(self):
        return self.embed_dim

    def _build_key_padding_mask(self, view_mask: torch.Tensor, B: int, V: int) -> torch.Tensor:
        """
        Build key_padding_mask for nn.MultiheadAttention.

        Args:
            view_mask: [B, V] bool, True=available, False=missing
            B: batch size
            V: num views

        Returns:
            key_padding_mask: [B, V+1] bool, True=ignore (PyTorch convention).
            CLS position (index 0) is always False (never ignored).
        """
        # PyTorch key_padding_mask: True means ignore -- inverse of view_mask semantics.
        view_padding = ~view_mask

        cls_padding = torch.zeros(B, 1, dtype=torch.bool, device=view_mask.device)

        key_padding_mask = torch.cat([cls_padding, view_padding], dim=1)
        return key_padding_mask

    def forward(self,
                view_features: List[torch.Tensor],
                view_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False
                ):
        """
        Args:
            view_features: list of V view encodings, each [B, d]
            view_mask: [B, V] bool, True=keep, False=missing. None means all available.
            return_attention: whether to return per-layer attention weights

        Returns:
            z: [B, d] fused representation (CLS token output)
            attn_weights: (only if return_attention=True) list of [B, V+1, V+1], one per layer
        """
        B = view_features[0].shape[0]
        device = view_features[0].device
        V = len(view_features)

        tokens = torch.stack(view_features, dim=1)

        key_padding_mask = None

        if view_mask is not None:
            if self.mask_strategy == 'attention_mask':
                # Token values at missing positions don't matter (attention won't see them),
                # but we zero them out anyway for numerical safety against NaNs.
                mask_expanded = view_mask.unsqueeze(-1).float()
                tokens = tokens * mask_expanded

                key_padding_mask = self._build_key_padding_mask(view_mask, B, V)

            else:
                mask_expanded = view_mask.unsqueeze(-1).float()

                if self.mask_strategy == 'learnable':
                    fill = self.mask_token.expand(B, V, -1)
                elif self.mask_strategy == 'zero':
                    fill = torch.zeros_like(tokens)
                elif self.mask_strategy == 'mean':
                    # Fill with mean of available views in the batch; detach to block gradients.
                    available = tokens * mask_expanded
                    n_available = mask_expanded.sum(dim=(0, 1)).clamp(min=1)
                    global_mean = available.sum(dim=(0, 1)) / n_available
                    fill = global_mean.detach().unsqueeze(0).unsqueeze(0).expand(B, V, -1)
                else:
                    raise ValueError(f"Unknown mask_strategy: {self.mask_strategy}")

                tokens = tokens * mask_expanded + fill * (1 - mask_expanded)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        tokens = tokens + self.view_embed[:, :V + 1, :]

        attn_weights_all = []
        for block in self.blocks:
            if return_attention:
                tokens, attn_w = block(tokens,
                                       key_padding_mask=key_padding_mask,
                                       return_attention=True)
                attn_weights_all.append(attn_w)
            else:
                tokens = block(tokens, key_padding_mask=key_padding_mask)

        tokens = self.norm(tokens)

        # Aggregation: CLS picks index 0; mean/max only pool over observed view tokens
        # (use view_mask to exclude missing positions).
        if self.aggregation == 'cls':
            z = tokens[:, 0]
        else:
            view_tokens = tokens[:, 1:]
            if view_mask is None:
                if self.aggregation == 'mean':
                    z = view_tokens.mean(dim=1)
                elif self.aggregation == 'max':
                    z = view_tokens.max(dim=1).values
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggregation}")
            else:
                m = view_mask.unsqueeze(-1)
                if self.aggregation == 'mean':
                    mf = m.float()
                    z = (view_tokens * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)
                elif self.aggregation == 'max':
                    z = view_tokens.masked_fill(~m, -1e9).max(dim=1).values
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggregation}")

        if return_attention:
            return z, attn_weights_all
        return z


class ConcatMLPFusion(nn.Module):
    """
    Fusion baseline that violates C1+C2 (used only in B8 ablation).

    Violates C1: view ordering is baked into the MLP weights -- not permutation-equivariant.
    Violates C2: missing views can only be zero-padded; there is no architectural
                 missing-view mechanism, so the training distribution and inference
                 distribution (with missingness) cannot be aligned at the architecture level.

    The interface matches ViewFusionTransformer exactly (forward signature, return value,
    width), so build_model only needs to swap the fusion module while keeping the rest
    of the pipeline unchanged.
    """

    def __init__(self,
                 embed_dim: int = 128,
                 num_views: int = 6,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views
        hidden = hidden_dim if hidden_dim is not None else embed_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(num_views * embed_dim, hidden),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def width(self):
        return self.embed_dim

    def forward(self,
                view_features: List[torch.Tensor],
                view_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        B = view_features[0].shape[0]
        V = len(view_features)
        assert V == self.num_views, \
            f"ConcatMLPFusion was built with num_views={self.num_views} but got V={V}"

        tokens = torch.stack(view_features, dim=1)

        if view_mask is not None:
            tokens = tokens * view_mask.unsqueeze(-1).float()

        flat = tokens.reshape(B, V * self.embed_dim)
        z = self.norm(self.mlp(flat))

        if return_attention:
            return z, []
        return z


class SetMLPFusion(nn.Module):
    """
    Permutation-invariant, variable-length MLP fusion (DeepSets style).

    Architecture: shared per-view MLP (phi) -> masked mean pool -> post-pool MLP (rho).
    Strictly satisfies C2: missing views are zeroed before sum and never enter rho's
    input; phi shares weights across views with no view-position binding. The fusion
    layer is permutation-equivariant; view identity is carried by the upstream
    per-view encoder.

    Forward signature/return value matches ConcatMLPFusion / ViewFusionTransformer,
    so build_model only needs to switch fusion_type.

    Parameter budget (4 Linear layers):
        phi: Linear(d, h) + Linear(h, h)  -> dh + h^2
        rho: Linear(h, h) + Linear(h, d)  -> h^2 + hd
        Total ~ 2dh + 2h^2
    """

    def __init__(self,
                 embed_dim: int = 128,
                 num_views: int = 6,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views
        hidden = hidden_dim if (hidden_dim is not None and hidden_dim > 0) else embed_dim * 2
        self.hidden = hidden

        # phi: shared per-view MLP (d -> h), weights shared across all view positions.
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(hidden),
            QuickGELU(),
            nn.Linear(hidden, hidden),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def width(self):
        return self.embed_dim

    def forward(self,
                view_features: List[torch.Tensor],
                view_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        Args:
            view_features: V tensors of shape [B, d]
            view_mask: [B, V] bool or float, 1=available, 0=missing. None means all available.
            return_attention: kept for interface compatibility; SetMLP has no attention,
                              returns an empty list.

        Returns:
            z: [B, d]
        """
        tokens = torch.stack(view_features, dim=1)
        phi_out = self.phi(tokens)

        if view_mask is None:
            pooled = phi_out.mean(dim=1)
        else:
            m = view_mask.unsqueeze(-1).to(phi_out.dtype)
            s = (phi_out * m).sum(dim=1)
            c = view_mask.sum(dim=1, keepdim=True).to(phi_out.dtype).clamp(min=1.0)
            pooled = s / c

        z = self.norm(self.rho(pooled))

        if return_attention:
            return z, []
        return z


class MultiViewEncoder(nn.Module):
    """Full multi-view encoder: ViewEncoders + ViewFusionTransformer."""

    def __init__(self,
                 view_encoders: nn.ModuleList,
                 fusion: ViewFusionTransformer):
        super().__init__()
        self.view_encoders = view_encoders
        self.fusion = fusion
        self.num_modalities = len(view_encoders)

    @property
    def fusion_transformer(self):
        return self.fusion

    def forward(self,
                x_list: List[torch.Tensor],
                view_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False,
                **kwargs
                ):
        encoded = []
        for i, (enc, x) in enumerate(zip(self.view_encoders, x_list)):
            z_i = enc(x)
            encoded.append(z_i)

        result = self.fusion(encoded, view_mask=view_mask,
                             return_attention=return_attention)
        if return_attention:
            z, attn_weights = result
            return z, None, attn_weights
        return z, None

    def encode_views(self,
                     x_list: List[torch.Tensor]
                     ) -> List[torch.Tensor]:
        """Encode only, no fusion."""
        return [enc(x) for enc, x in zip(self.view_encoders, x_list)]

    def fuse(self,
             encoded_list: List[torch.Tensor],
             view_mask: Optional[torch.Tensor] = None,
             return_attention: bool = False
             ):
        """Fuse only, no encoding."""
        return self.fusion(encoded_list, view_mask=view_mask,
                           return_attention=return_attention)
