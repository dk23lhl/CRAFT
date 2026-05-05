"""
CRAFT: A Fusion Transformer with Two-Stage Optimization
       for Efficient and Robust Multi-View Clustering

Core architecture:
1. Stage 1: representation alignment (SimSiam or VICReg) + MSE reconstruction; learns
   a robust fused representation with no clustering loss.
2. Transition: L2-normalize features -> spherical K-Means -> initialize cluster head.
3. Stage 2: spherical cosine cluster head + cluster-consistency KL fine-tune.

Key components:
- ViewEncoder (MLP per view) + ViewFusionTransformer (CLS + self-attention)
- PureClusterHead (L2-normalized cosine similarity, spherical geometry)
- ClusterConsistencyLoss (symmetric KL + stop-gradient)
- EntropyRegularization (global entropy maximization + sample entropy minimization)
- RepresentationAlignmentLoss (SimSiam-style) or VICRegLoss (variance-invariance-covariance)
- MSE Reconstruction (reconstruct each view's raw features from z_full)

Incomplete MVC support:
- ViewFusionTransformer has a built-in mask_token; missing views are auto-replaced.
- COMPLETER-protocol missing masks are passed via the view_mask argument.
- Reconstruction loss automatically skips missing views.

v2 additions - Mask-Aware Fine-Tuning:
- train_mask_prob: probability per batch of enabling random view masking during training.
- train_mask_max_views: max views to mask per sample (default V-2, keeps at least 2 usable).
- Used when fine-tuning from a complete-trained ckpt so that mask_token sees forward passes
  and gets optimized.
- Reconstruction loss skips masked views to avoid spurious gradients.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from pl_modules.base import BaseModel
from models.fusion import MultiViewEncoder
from losses.cluster_consistency_loss import ClusterConsistencyLoss
from losses.entropy import EntropyRegularization
from losses.representation_loss import RepresentationAlignmentLoss, VICRegLoss


class PureClusterHead(torch.nn.Module):
    """L2-normalized cosine similarity cluster head, matches spherical K-Means exactly."""
    def __init__(self, input_dim, num_clusters, temperature=0.1):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(num_clusters, input_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.temperature = temperature

    def forward(self, x):
        x_norm = torch.nn.functional.normalize(x, dim=-1, p=2)
        w_norm = torch.nn.functional.normalize(self.weight, dim=-1, p=2)
        logits = torch.matmul(x_norm, w_norm.t())
        return torch.nn.functional.softmax(logits / self.temperature, dim=-1)


class CRAFT(BaseModel):
    """
    CRAFT: A Fusion Transformer with Two-Stage Optimization
           for Efficient and Robust Multi-View Clustering
    """

    def __init__(self,
                 encoder: MultiViewEncoder,
                 num_clusters: int,
                 view_decoders: nn.ModuleList = None,
                 num_views: int = 1,
                 num_masked_views: int = 0,
                 train_mask_prob: float = 0.0,
                 train_mask_max_views: int = -1,
                 cluster_temperature: float = 0.1,
                 lambda_cluster: float = 1.0,
                 lambda_entropy: float = 5.0,
                 lambda_repr: float = 1.0,
                 lambda_recon: float = 1.0,
                 cluster_loss_symmetric: bool = True,
                 entropy_sample_weight: float = 0.1,
                 repr_mode: str = 'simsiam',  # 'simsiam' or 'vicreg'
                 entropy_warmup_epochs: int = 10,
                 entropy_warmup_multiplier: float = 2.0,
                 optim_kwargs: Dict = None,
                 **kwargs):
        super().__init__(optim_kwargs)

        self.encoder = encoder
        self.view_decoders = view_decoders

        embed_dim = encoder.fusion_transformer.width

        self.cluster_head = PureClusterHead(
            input_dim=embed_dim,
            num_clusters=num_clusters,
            temperature=cluster_temperature
        )

        self.cluster_consistency_loss = ClusterConsistencyLoss(
            symmetric=cluster_loss_symmetric
        )
        self.entropy_regularization = EntropyRegularization(
            global_weight=1.0,
            sample_weight=entropy_sample_weight
        )

        self.repr_mode = repr_mode
        if repr_mode == 'vicreg':
            self.representation_loss = VICRegLoss(
                lambda_sim=25.0, lambda_var=25.0, lambda_cov=1.0
            )
            print(f"[CRAFT] Representation alignment mode: VICReg (variance-invariance-covariance)")
        else:
            self.representation_loss = RepresentationAlignmentLoss()
            print(f"[CRAFT] Representation alignment mode: SimSiam (stop-gradient cosine)")

        self.num_clusters = num_clusters
        self.lambda_cluster = lambda_cluster
        self.lambda_entropy = lambda_entropy
        self.lambda_repr = lambda_repr
        self.lambda_recon = lambda_recon
        self.entropy_warmup_epochs = entropy_warmup_epochs
        self.entropy_warmup_multiplier = entropy_warmup_multiplier

        # Legacy: fixed num_masked_views per sample
        self.num_views = num_views
        self.num_masked_views = num_masked_views

        # train_mask_prob: per-batch probability of enabling random view masking
        #   0.0 = disabled (default, backward compatible)
        #   1.0 = mask every batch
        #   0.5 = mask 50% of batches (keeps some complete forward passes)
        self.train_mask_prob = train_mask_prob

        # train_mask_max_views: max views to mask per sample
        #   -1 = auto-set to V-2 (keeps at least 2 views usable)
        if train_mask_max_views < 0:
            self.train_mask_max_views = max(1, num_views - 2)
        else:
            self.train_mask_max_views = train_mask_max_views

        if train_mask_prob > 0:
            print(f"[CRAFT] Mask-Aware Fine-Tuning enabled:")
            print(f"  train_mask_prob     = {train_mask_prob}")
            print(f"  train_mask_max_views = {self.train_mask_max_views}")
            print(f"  num_views           = {num_views}")
            print(f"  per-sample mask count: Uniform(0, {self.train_mask_max_views})")

        self.epoch = 0
        self.total_epochs = 200

        self.save_hyperparameters(ignore=['encoder', 'view_decoders'])

    def generate_random_view_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate a fixed-size random view mask for training (legacy interface).
        Each sample randomly masks num_masked_views views.

        Returns:
            view_mask: [B, V] bool, True=observed, False=masked
        """
        mask = torch.ones(batch_size, self.num_views, dtype=torch.bool, device=device)
        for i in range(batch_size):
            masked_indices = torch.randperm(self.num_views, device=device)[:self.num_masked_views]
            mask[i, masked_indices] = False
        return mask

    def generate_variable_view_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Variable mask generator used by Mask-Aware Fine-Tuning.

        Per sample, draw k ~ Uniform(0, train_mask_max_views) and mask k random views.

        Design notes:
        - k=0 samples keep a full forward pass so complete-data performance doesn't collapse.
        - Sampling k uniformly exposes the model to many missing-rate levels.
        - At least 2 views are always kept (k_max=4 when V=6).

        Returns:
            view_mask: [B, V] bool, True=observed, False=masked
        """
        V = self.num_views
        k_max = self.train_mask_max_views

        mask = torch.ones(batch_size, V, dtype=torch.bool, device=device)

        # Sample k per sample in 0..k_max (k=0 means a fully observed sample).
        k_per_sample = torch.randint(0, k_max + 1, (batch_size,), device=device)

        for i in range(batch_size):
            k = k_per_sample[i].item()
            if k > 0:
                drop_idx = torch.randperm(V, device=device)[:k]
                mask[i, drop_idx] = False

        return mask

    def set_current_epoch(self, epoch: int):
        self.epoch = epoch

    def set_total_epochs(self, total_epochs: int):
        self.total_epochs = total_epochs

    def forward(self,
                views_aug1: List[torch.Tensor],
                views_aug2: List[torch.Tensor],
                view_mask: Optional[torch.Tensor] = None
                ) -> Dict:
        """
        Args:
            views_aug1: augmented view set 1, V tensors of shape [B, d_i]
            views_aug2: augmented view set 2, V tensors of shape [B, d_i]
            view_mask: [B, V] bool, True=view observed, None=all observed
        """
        encoded_1 = self.encoder.encode_views(views_aug1)
        encoded_2 = self.encoder.encode_views(views_aug2)

        # Full-view fusion (missing views are auto-replaced by mask_token).
        z1_full = self.encoder.fuse(encoded_1, view_mask=view_mask)
        z2_full = self.encoder.fuse(encoded_2, view_mask=view_mask)

        p1_full = self.cluster_head(z1_full)
        p2_full = self.cluster_head(z2_full)

        return {
            'z1_full': z1_full,
            'z2_full': z2_full,
            'p1_full': p1_full,
            'p2_full': p2_full,
            'views_aug1': views_aug1,
            'views_aug2': views_aug2,
            'view_mask': view_mask,
        }

    def init_cluster_head(self, cluster_centers: torch.Tensor):
        """Initialize cluster-head weights with spherical K-Means centroids."""
        if hasattr(self.cluster_head, 'weight') and isinstance(self.cluster_head.weight, torch.nn.Parameter):
            centers_norm = torch.nn.functional.normalize(cluster_centers, dim=-1, p=2)
            expected_shape = self.cluster_head.weight.shape
            actual_shape = centers_norm.shape
            if expected_shape == actual_shape:
                self.cluster_head.weight.data = centers_norm.clone().detach().to(self.device)
                print(f"\n[Success] Cluster head initialized from spherical K-Means centroids {actual_shape}")
            else:
                print(f"\n[Error] Shape mismatch! expected {expected_shape}, got {actual_shape}")
        else:
            print("\n[Error] No initializable weight Parameter found")

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze encoder + fusion + decoder, keeping only the cluster head trainable."""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        if self.view_decoders is not None:
            for param in self.view_decoders.parameters():
                param.requires_grad = not freeze
        for param in self.cluster_head.parameters():
            param.requires_grad = True

        status = "FROZEN" if freeze else "UNFROZEN"
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"\nBackbone {status}")
        print(f"  Trainable: {n_trainable:,} / {n_total:,} parameters "
              f"({n_trainable/n_total*100:.1f}%)")

    def freeze_encoder_only(self, freeze: bool = True):
        """Freeze only the view encoder; fusion + cluster_head + decoder stay trainable."""
        for enc in self.encoder.view_encoders:
            for param in enc.parameters():
                param.requires_grad = not freeze
        for param in self.encoder.fusion_transformer.parameters():
            param.requires_grad = True
        for param in self.cluster_head.parameters():
            param.requires_grad = True
        if self.view_decoders is not None:
            for param in self.view_decoders.parameters():
                param.requires_grad = True

        status = "FROZEN" if freeze else "UNFROZEN"
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        n_encoder = sum(p.numel() for enc in self.encoder.view_encoders for p in enc.parameters())
        print(f"\nView Encoders {status} ({n_encoder:,} params)")
        print(f"  Fusion/ClusterHead/Decoder: trainable")
        print(f"  Trainable: {n_trainable:,} / {n_total:,} parameters "
              f"({n_trainable/n_total*100:.1f}%)")

    def loss(self, outputs: Dict) -> Dict:
        """Compute total loss; lambda gating effectively switches Stage 1/2 behavior."""
        losses = {}
        device = outputs['z1_full'].device
        total_loss = torch.tensor(0.0, device=device)

        # Cluster consistency (Stage 2 only)
        if self.lambda_cluster > 0:
            cluster_loss = self.cluster_consistency_loss(
                outputs['p1_full'], outputs['p2_full']
            )
            losses['cluster_loss'] = cluster_loss
            total_loss += self.lambda_cluster * cluster_loss
        else:
            losses['cluster_loss'] = torch.tensor(0.0, device=device)

        # Entropy regularization (Stage 2 only)
        if self.lambda_entropy > 0:
            avg_probs = (outputs['p1_full'] + outputs['p2_full']) / 2.0
            entropy_dict = self.entropy_regularization(avg_probs)
            losses['entropy_loss'] = entropy_dict['entropy_loss']
            losses['global_entropy'] = entropy_dict['global_entropy']

            if self.epoch < self.entropy_warmup_epochs:
                effective_lambda_entropy = self.lambda_entropy * self.entropy_warmup_multiplier
            else:
                effective_lambda_entropy = self.lambda_entropy

            total_loss += effective_lambda_entropy * entropy_dict['entropy_loss']
            losses['effective_lambda_entropy'] = effective_lambda_entropy
        else:
            losses['entropy_loss'] = torch.tensor(0.0, device=device)
            losses['global_entropy'] = torch.tensor(0.0, device=device)
            losses['effective_lambda_entropy'] = 0.0

        # Representation alignment (always on)
        if self.lambda_repr > 0:
            repr_loss = self.representation_loss(
                outputs['z1_full'], outputs['z2_full']
            )
            losses['repr_loss'] = repr_loss
            total_loss += self.lambda_repr * repr_loss

        # Reconstruction loss (always on; missing views are skipped)
        if self.lambda_recon > 0 and self.view_decoders is not None:
            recon_loss = torch.tensor(0.0, device=device)
            view_mask = outputs.get('view_mask', None)
            num_valid_terms = 0

            for i, decoder in enumerate(self.view_decoders):
                x1_recon = decoder(outputs['z1_full'])
                x2_recon = decoder(outputs['z2_full'])

                if view_mask is not None:
                    mask_i = view_mask[:, i]  # [B] bool
                    if mask_i.any():
                        loss_1 = ((x1_recon[mask_i] - outputs['views_aug1'][i][mask_i]) ** 2).mean()
                        loss_2 = ((x2_recon[mask_i] - outputs['views_aug2'][i][mask_i]) ** 2).mean()
                        recon_loss += loss_1 + loss_2
                        num_valid_terms += 2
                else:
                    recon_loss += nn.functional.mse_loss(x1_recon, outputs['views_aug1'][i])
                    recon_loss += nn.functional.mse_loss(x2_recon, outputs['views_aug2'][i])
                    num_valid_terms += 2

            if num_valid_terms > 0:
                recon_loss = recon_loss / num_valid_terms
            losses['recon_loss'] = recon_loss
            total_loss += self.lambda_recon * recon_loss
        else:
            losses['recon_loss'] = torch.tensor(0.0, device=device)

        losses['loss'] = total_loss
        return losses

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if len(batch) == 4:
            views_aug1, views_aug2, labels, view_mask = batch
        else:
            views_aug1, views_aug2, labels = batch
            view_mask = None

        # Mask source priority: batch-provided > train_mask_prob > num_masked_views
        if view_mask is None and self.train_mask_prob > 0:
            if torch.rand(1).item() < self.train_mask_prob:
                B = views_aug1[0].shape[0]
                view_mask = self.generate_variable_view_mask(B, views_aug1[0].device)
        elif view_mask is None and self.num_masked_views > 0:
            B = views_aug1[0].shape[0]
            view_mask = self.generate_random_view_mask(B, views_aug1[0].device)

        outputs = self.forward(views_aug1, views_aug2, view_mask=view_mask)
        loss_dict = self.loss(outputs)

        self.log('train/loss', loss_dict['loss'],
                 on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/cluster_loss', loss_dict['cluster_loss'],
                 on_step=False, on_epoch=True)
        self.log('train/entropy_loss', loss_dict['entropy_loss'],
                 on_step=False, on_epoch=True)
        self.log('train/repr_loss', loss_dict.get('repr_loss', 0),
                 on_step=False, on_epoch=True)
        self.log('train/recon_loss', loss_dict.get('recon_loss', 0),
                 on_step=False, on_epoch=True)
        self.log('train/global_entropy', loss_dict['global_entropy'],
                 on_step=False, on_epoch=True)
        self.log('train/effective_entropy_weight',
                 loss_dict['effective_lambda_entropy'],
                 on_step=False, on_epoch=True)

        # Extra log: mask stats (only during mask-aware fine-tuning)
        if view_mask is not None and self.train_mask_prob > 0:
            avg_missing = (~view_mask).float().sum(dim=1).mean()
            self.log('train/avg_masked_views', avg_missing,
                     on_step=False, on_epoch=True)

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        view_mask = None
        if len(batch) == 4:
            views_aug1, views_aug2, labels, view_mask = batch
        elif len(batch) == 3:
            views_aug1, views_aug2, labels = batch
        else:
            views, labels = batch
            views_aug1 = views
            views_aug2 = views

        outputs = self.forward(views_aug1, views_aug2, view_mask=view_mask)
        loss_dict = self.loss(outputs)

        self.log('val/loss', loss_dict['loss'],
                 on_epoch=True, sync_dist=True, prog_bar=True)

        return loss_dict['loss']
