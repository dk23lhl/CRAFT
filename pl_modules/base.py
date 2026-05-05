import pytorch_lightning as pl
import torch
from typing import Dict


class BaseModel(pl.LightningModule):
    """Base model with optimizer configuration."""

    def __init__(self, optim_kwargs: Dict = None):
        super().__init__()
        self.optim_kwargs = optim_kwargs or {}

    def configure_optimizers(self):
        optimizer_name = self.optim_kwargs.get('optimizer', 'adam').lower()
        lr = self.optim_kwargs.get('lr', 1e-3)
        weight_decay = self.optim_kwargs.get('weight_decay', 1e-4)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = self.optim_kwargs.get('momentum', 0.9)
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        use_scheduler = self.optim_kwargs.get('use_scheduler', True)
        if use_scheduler:
            T_max = self.optim_kwargs.get('max_epochs', 150)
            min_lr = self.optim_kwargs.get('min_lr', 1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=min_lr
            )
            return [optimizer], [scheduler]

        return optimizer
