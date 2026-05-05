import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional


class MultiViewDataset(Dataset):
    """
    Generic multi-view dataset.

    Two modes:
    - Training (augment=True): returns two augmented view sets + label
    - Eval (augment=False): returns raw views + label
    """

    def __init__(self,
                 features_list: List[np.ndarray],
                 labels: np.ndarray,
                 augment: bool = True,
                 noise_std: float = 0.1,
                 dropout_rate: float = 0.1,
                 missing_rate: float = 0.0,
                 missing_seed: int = 42,
                 mask_protocol: str = 'completer'):
        """
        Args:
            features_list: list of V view feature matrices, each [N, d_i]
            labels: [N] label array
            augment: whether to apply data augmentation
            noise_std: Gaussian noise std
            dropout_rate: feature dropout ratio
            missing_rate: missing rate (0 = complete)
            missing_seed: RNG seed for the missing mask
            mask_protocol: 'completer' (P1, sample-level drop 1 view, default for backward compat)
                           or 'cell_level' (P4, cell-level Bernoulli, same distribution as
                           incomplete_eval._mask_cell_level)
        """
        super().__init__()
        self.num_views = len(features_list)
        self.views = [torch.FloatTensor(f) for f in features_list]
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate

        N = len(self.labels)
        for i, v in enumerate(self.views):
            assert v.shape[0] == N, (
                f"View {i} has {v.shape[0]} samples, expected {N}"
            )

        self.view_dims = [v.shape[1] for v in self.views]

        self.missing_rate = missing_rate
        if missing_rate > 0:
            rng = np.random.default_rng(missing_seed)
            if mask_protocol == 'cell_level':
                # P4: cell-level Bernoulli, every sample keeps at least 1 view.
                # Matches the distribution of incomplete_eval._mask_cell_level exactly.
                N_cells = N * self.num_views
                n_missing = min(int(round(N_cells * missing_rate)),
                                N * (self.num_views - 1))
                protected = rng.integers(0, self.num_views, size=N)
                candidates = [(i, vi) for i in range(N)
                              for vi in range(self.num_views)
                              if vi != protected[i]]
                rng.shuffle(candidates)
                mask_np = np.ones((N, self.num_views), dtype=bool)
                for idx in range(n_missing):
                    i, vi = candidates[idx]
                    mask_np[i, vi] = False
                self.view_mask = torch.BoolTensor(mask_np)
                actual = n_missing / N_cells
                print(f"[Cell-Level Mask P4] {n_missing}/{N_cells} cells, "
                      f"actual_rate={actual:.2%}")
            elif mask_protocol == 'completer':
                # P1: sample-level deletion (protected) — pick m samples and drop 1 view each.
                m = int(round(N * missing_rate))
                self.view_mask = np.ones((N, self.num_views), dtype=bool)
                selected = rng.choice(N, size=m, replace=False)
                for idx in selected:
                    drop_v = rng.integers(0, self.num_views)
                    self.view_mask[idx, drop_v] = False
                self.view_mask = torch.BoolTensor(self.view_mask)
                actual = m / N
                print(f"[COMPLETER Mask P1] {m}/{N} samples x 1 view, "
                      f"actual_rate={actual:.2%}")
            else:
                raise ValueError(f"Unknown mask_protocol: {mask_protocol}; "
                                 f"expected 'completer' or 'cell_level'")
        else:
            self.view_mask = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        views = [v[idx] for v in self.views]
        label = self.labels[idx]

        # view_mask convention: True = available; None means all available.
        vm = self.view_mask[idx] if self.view_mask is not None else None

        if self.augment:
            views_aug1 = [self._augment_feature(v) for v in views]
            views_aug2 = [self._augment_feature(v) for v in views]
            if vm is not None:
                return views_aug1, views_aug2, label, vm
            return views_aug1, views_aug2, label
        else:
            if vm is not None:
                return views, label, vm
            return views, label

    def _augment_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Feature-level augmentation: Gaussian noise + dropout."""
        noise = torch.randn_like(x) * self.noise_std
        x_aug = x + noise
        mask = (torch.rand_like(x) > self.dropout_rate).float()
        x_aug = x_aug * mask
        return x_aug


class MultiViewDataModule:
    """
    DataModule wrapper for multi-view data.

    Simplified: not a LightningDataModule, just exposes dataloader methods.
    """

    def __init__(self,
                 features_list: List[np.ndarray],
                 labels: np.ndarray,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 noise_std: float = 0.1,
                 dropout_rate: float = 0.1,
                 train_ratio: float = 1.0,
                 seed: int = 42,
                 missing_rate: float = 0.0,
                 mask_protocol: str = 'completer'):

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Unsupervised clustering: train on the full set by default (train_ratio=1.0).
        # No validation split is needed since labels do not participate in training.
        if train_ratio >= 1.0:
            self.train_dataset = MultiViewDataset(
                features_list, labels,
                augment=True, noise_std=noise_std, dropout_rate=dropout_rate,
                missing_rate=missing_rate, missing_seed=seed,
                mask_protocol=mask_protocol,
            )
            # Val set = train set (augmented); only used to monitor loss.
            self.val_dataset = MultiViewDataset(
                features_list, labels,
                augment=True, noise_std=noise_std, dropout_rate=dropout_rate,
                missing_rate=missing_rate, missing_seed=seed,
                mask_protocol=mask_protocol,
            )
        else:
            N = len(labels)
            rng = np.random.RandomState(seed)
            indices = rng.permutation(N)
            split = int(N * train_ratio)

            train_idx = indices[:split]
            val_idx = indices[split:]

            train_features = [f[train_idx] for f in features_list]
            val_features = [f[val_idx] for f in features_list]
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]

            self.train_dataset = MultiViewDataset(
                train_features, train_labels,
                augment=True, noise_std=noise_std, dropout_rate=dropout_rate,
                missing_rate=missing_rate, missing_seed=seed,
                mask_protocol=mask_protocol,
            )
            self.val_dataset = MultiViewDataset(
                val_features, val_labels,
                augment=True, noise_std=noise_std, dropout_rate=dropout_rate,
                missing_rate=missing_rate, missing_seed=seed + 1,
                mask_protocol=mask_protocol,
            )

        # Full dataset (no augmentation) for clustering evaluation.
        self.full_dataset = MultiViewDataset(
            features_list, labels, augment=False
        )

        self.view_dims = self.train_dataset.view_dims
        self.num_views = self.train_dataset.num_views
        self.num_classes = len(np.unique(labels))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def full_dataloader(self):
        return torch.utils.data.DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
