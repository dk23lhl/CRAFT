import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ViewMasker(nn.Module):
    """
    View-level random masking.

    Operates on encoded view features (not raw inputs); each masked view drops
    1..max_drop views at random. Dropped views are later replaced by the
    mask_token at fusion time (handled by the fusion module).

    Difference vs. v9 SynergyMasker:
    - v9: modality-level binary switch, zeroes the entire modality input before encoding.
    - This version: view-level random drop at the encoded stage, fusion replaces the
      dropped views with mask_token.

    Why: each view's encoder always sees a complete input -> better representations;
    fusion's mask_token replacement (vs. zero-padding) lets attention sense the
    semantic of "missing".
    """

    def __init__(self,
                 num_views: int,
                 num_masked_views: int = 4,
                 min_keep: int = 1,
                 max_drop: Optional[int] = None,
                 adaptive: bool = True,
                 warmup_epochs: int = 20):
        """
        Args:
            num_views: total number of views in the dataset (V)
            num_masked_views: how many masked variants to produce per call
            min_keep: minimum views kept per masked variant
            max_drop: max views dropped per masked variant
            adaptive: enable adaptive masking schedule
            warmup_epochs: warmup length for adaptive schedule
        """
        super().__init__()

        self.num_views = num_views
        self.num_masked_views = num_masked_views
        self.min_keep = min_keep
        self.max_drop = max_drop or (num_views - min_keep)
        self.adaptive = adaptive
        self.warmup_epochs = warmup_epochs

        self.current_epoch = 0
        self.max_epochs = 200

        assert 1 <= min_keep <= num_views
        assert 1 <= self.max_drop <= num_views - min_keep + 1

    def set_epoch(self, epoch: int, max_epochs: int = None):
        self.current_epoch = epoch
        if max_epochs is not None:
            self.max_epochs = max_epochs

    @property
    def effective_max_drop(self) -> int:
        """Adaptive masking: gentle at the start of training, aggressive later."""
        if not self.adaptive:
            return self.max_drop

        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            max_drop = 1 + int((self.max_drop - 1) * progress)
        else:
            max_drop = self.max_drop

        return max(1, min(max_drop, self.num_views - self.min_keep))

    def forward(self,
                encoded_views: List[torch.Tensor],
                return_masks: bool = False
                ) -> Tuple[List[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Generate multiple masked configurations.

        Note: this does not modify features themselves, only generates the mask
        configuration. The mask_token replacement is performed by the fusion module.

        Args:
            encoded_views: V encoded view features, each [B, d]
            return_masks: whether to return the mask configurations

        Returns:
            masked_views_list: num_masked_views lists, each containing V tensors of
                shape [B, d] where dropped views are zeroed.
            masks_list: num_masked_views boolean masks of shape [B, V],
                True=keep, False=drop.
        """
        B = encoded_views[0].shape[0]
        device = encoded_views[0].device
        V = self.num_views

        effective_max_drop = self.effective_max_drop
        masked_views_list = []
        masks_list = []

        for _ in range(self.num_masked_views):
            mask = self._generate_mask(B, V, device, effective_max_drop)

            # Apply mask: zero out dropped views (fusion module decides whether to
            # replace with mask_token).
            view = []
            for v in range(V):
                m = mask[:, v].float().unsqueeze(-1)
                view.append(encoded_views[v] * m)
            masked_views_list.append(view)
            masks_list.append(mask)

        if return_masks:
            return masked_views_list, masks_list
        return masked_views_list, None

    def _generate_mask(self,
                       B: int, V: int,
                       device: torch.device,
                       max_drop: int) -> torch.Tensor:
        """
        Generate one mask configuration.

        Strategy: each sample independently drops 1..max_drop views at random.

        Returns:
            mask: [B, V] bool tensor, True=keep
        """
        mask = torch.ones(B, V, device=device, dtype=torch.bool)

        for b in range(B):
            num_drop = torch.randint(1, max_drop + 1, (1,)).item()
            num_drop = min(num_drop, V - self.min_keep)

            drop_idx = torch.randperm(V, device=device)[:num_drop]
            mask[b, drop_idx] = False

        return mask

    def generate_complementary_masks(self,
                                     B: int, V: int,
                                     device: torch.device
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a complementary mask pair: mask1 | mask2 covers all views.

        Used for optional complementary-consistency training.

        Returns:
            mask1, mask2: two [B, V] masks with mask1 | mask2 = all True
        """
        mask1 = torch.zeros(B, V, device=device, dtype=torch.bool)
        mask2 = torch.zeros(B, V, device=device, dtype=torch.bool)

        for b in range(B):
            perm = torch.randperm(V, device=device)
            split = torch.randint(self.min_keep, V - self.min_keep + 1, (1,)).item()
            mask1[b, perm[:split]] = True
            mask2[b, perm[split:]] = True

        return mask1, mask2
