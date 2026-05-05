"""
CRAFT Multi-View Clustering training entry point.

CRAFT: A Fusion Transformer with Two-Stage Optimization
       for Efficient and Robust Multi-View Clustering

v2 additions:
- --resume_from: load weights from an existing ckpt for fine-tune (does not restore optimizer/epoch)
- --train_mask_prob: Mask-Aware Fine-Tuning, randomly masks views during training so mask_token gets optimized
- --train_mask_max_views: max views to mask per sample (default V-2)
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from datasets.load_data import load_dataset
from datasets.multiview_dataset import MultiViewDataModule
from models.view_encoder import build_view_encoders, build_view_decoders
from models.fusion import ViewFusionTransformer, ConcatMLPFusion, SetMLPFusion, MultiViewEncoder
from pl_modules.craft_mvc import CRAFT
from evaluation.clustering_metrics import evaluate_model, clustering_accuracy


class EpochInfoCallback(Callback):
    """Pass epoch info to the model."""
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.set_current_epoch(trainer.current_epoch)

    def on_train_start(self, trainer, pl_module):
        pl_module.set_total_epochs(trainer.max_epochs)


class ClusteringEvalCallback(Callback):
    """Run a clustering evaluation every N epochs."""
    def __init__(self, full_dataloader, eval_every: int = 5):
        super().__init__()
        self.full_dataloader = full_dataloader
        self.eval_every = eval_every
        self.best_acc = 0.0
        self.best_metrics = {}
        self.best_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.eval_every == 0 or epoch == 0:
            device = pl_module.device
            metrics = evaluate_model(pl_module, self.full_dataloader, device)

            for k, v in metrics.items():
                pl_module.log(f'eval/{k}', v, on_epoch=True, prog_bar=(k == 'ACC'))

            if metrics['ACC'] > self.best_acc:
                self.best_acc = metrics['ACC']
                self.best_metrics = dict(metrics)
                self.best_epoch = epoch + 1

            print(f"\n[Epoch {epoch+1}] "
                  f"ACC={metrics['ACC']:.4f} "
                  f"NMI={metrics['NMI']:.4f} "
                  f"ARI={metrics['ARI']:.4f}"
                  f"  (Best: ACC={self.best_acc:.4f} @ Epoch {self.best_epoch})")


def diagnose_random_encoder(model, datamodule, device, num_clusters):
    """Encode all data with the randomly initialized encoder+fusion and
    run K-Means on fused features to report ACC/NMI."""
    model.eval()
    model.to(device)
    all_z = []
    all_labels = []

    dl = datamodule.full_dataloader()

    with torch.no_grad():
        for batch in dl:
            if len(batch) == 4:
                views, _, labels, _ = batch
            elif len(batch) == 3:
                views, _, labels = batch
            else:
                views, labels = batch

            views = [v.to(device) for v in views]

            encoded_views = model.encoder.encode_views(views)
            fused_rep = model.encoder.fuse(encoded_views)
            if isinstance(fused_rep, tuple):
                fused_rep = fused_rep[0]

            all_z.append(fused_rep.cpu())
            all_labels.append(labels.numpy() if isinstance(labels, np.ndarray) else labels.cpu().numpy())

    all_z = torch.cat(all_z, dim=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_z_norm = torch.nn.functional.normalize(all_z, dim=-1, p=2).numpy()

    km = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
    preds = km.fit_predict(all_z_norm)

    acc = clustering_accuracy(all_labels, preds)
    nmi = normalized_mutual_info_score(all_labels, preds)

    print(f"\n{'='*60}")
    print(f"[DIAG] Random Encoder Diagnostic (before any training)")
    print(f"[DIAG]   K-Means ACC = {acc:.4f}")
    print(f"[DIAG]   K-Means NMI = {nmi:.4f}")
    print(f"[DIAG]   Feature shape = {all_z.shape}")
    print(f"[DIAG]   num_clusters = {num_clusters}")
    print(f"[DIAG] If this value is far below the Concat baseline, the architecture (e.g. embed_dim) doesn't fit this dataset")
    print(f"{'='*60}\n")

    return acc, nmi


def build_model(num_views: int, view_dims: list, num_clusters: int, cfg: dict) -> CRAFT:
    embed_dim = cfg.get('embed_dim', 128)
    encoder_hidden = cfg.get('encoder_hidden_dim', 256)
    encoder_layers = cfg.get('encoder_num_layers', 2)
    encoder_dropout = cfg.get('encoder_dropout', 0.1)
    encoder_type = cfg.get('encoder_type', 'shallow')

    view_encoders = build_view_encoders(
        view_dims=view_dims,
        output_dim=embed_dim,
        hidden_dim=encoder_hidden,
        num_layers=encoder_layers,
        dropout=encoder_dropout,
        encoder_type=encoder_type
    )

    fusion_type = cfg.get('fusion_type', 'transformer')
    if fusion_type == 'transformer':
        fusion = ViewFusionTransformer(
            embed_dim=embed_dim,
            num_views=num_views,
            n_heads=cfg.get('fusion_n_heads', 4),
            n_layers=cfg.get('fusion_n_layers', 1),
            dropout=cfg.get('fusion_dropout', 0.1),
            mask_strategy=cfg.get('mask_strategy', 'learnable'),
            aggregation=cfg.get('aggregation', 'cls'),
        )
    elif fusion_type == 'concat_mlp':
        concat_hidden = cfg.get('concat_mlp_hidden', -1)
        if concat_hidden is None or concat_hidden < 0:
            concat_hidden = embed_dim * 2
        fusion = ConcatMLPFusion(
            embed_dim=embed_dim,
            num_views=num_views,
            hidden_dim=concat_hidden,
            dropout=cfg.get('fusion_dropout', 0.1),
        )
    elif fusion_type == 'set_mlp':
        set_hidden = cfg.get('set_mlp_hidden', -1)
        if set_hidden is None or set_hidden < 0:
            set_hidden = embed_dim * 2
        fusion = SetMLPFusion(
            embed_dim=embed_dim,
            num_views=num_views,
            hidden_dim=set_hidden,
            dropout=cfg.get('fusion_dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")

    encoder = MultiViewEncoder(view_encoders, fusion)

    lambda_recon = cfg.get('lambda_recon', 1.0)
    view_decoders = None
    if lambda_recon > 0:
        view_decoders = build_view_decoders(
            view_dims=view_dims,
            input_dim=embed_dim,
            hidden_dim=encoder_hidden,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
            encoder_type=encoder_type
        )

    model = CRAFT(
        encoder=encoder,
        num_clusters=num_clusters,
        view_decoders=view_decoders,
        num_views=num_views,
        num_masked_views=cfg.get('num_masked_views', 0),
        train_mask_prob=cfg.get('train_mask_prob', 0.0),
        train_mask_max_views=cfg.get('train_mask_max_views', -1),
        cluster_temperature=cfg.get('cluster_temperature', 0.1),
        lambda_cluster=cfg.get('lambda_cluster', 1.0),
        lambda_entropy=cfg.get('lambda_entropy', 5.0),
        lambda_repr=cfg.get('lambda_repr', 1.0),
        lambda_recon=cfg.get('lambda_recon', 1.0),
        cluster_loss_symmetric=cfg.get('cluster_loss_symmetric', True),
        entropy_sample_weight=cfg.get('entropy_sample_weight', 0.1),
        entropy_warmup_epochs=cfg.get('entropy_warmup_epochs', 10),
        entropy_warmup_multiplier=cfg.get('entropy_warmup_multiplier', 2.0),
        repr_mode=cfg.get('repr_mode', 'simsiam'),
        optim_kwargs={
            'optimizer': cfg.get('optimizer', 'adam'),
            'lr': cfg.get('lr', 1e-3),
            'weight_decay': cfg.get('weight_decay', 1e-4),
            'use_scheduler': cfg.get('use_scheduler', True),
            'max_epochs': cfg.get('max_epochs', 150),
            'min_lr': cfg.get('min_lr', 1e-6),
        }
    )

    return model


def load_weights_from_checkpoint(model: CRAFT, ckpt_path: str):
    """Load weights only from ckpt (does not restore optimizer/epoch/scheduler).
    Used for fine-tune: keep model params, restart training."""
    print(f"\n{'='*60}")
    print(f"Loading weights from: {ckpt_path}")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    # Filter out legacy keys not present in the current model
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('masker.') and 'asymmetric_consistency_loss' not in k
    }

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        print(f"  [Warning] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [Info] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    print(f"\n  --- Learnable Token Status ---")
    for name, param in model.named_parameters():
        if any(t in name for t in ['mask_token', 'cls_token']):
            p = param.detach().cpu().float()
            print(f"  {name}: norm={p.norm().item():.4f}, "
                  f"std={p.std().item():.4f}, "
                  f"requires_grad={param.requires_grad}")

    print(f"  Weights loaded successfully.\n")
    return model


def main():
    parser = argparse.ArgumentParser(description='CRAFT Multi-View Clustering')

    parser.add_argument('--dataset', type=str, default='caltech101-20',
                        choices=['caltech101-20', 'handwritten', 'ngs', 'bbc',
                                 'uci-digit', 'hdigit', 'aloi-100', 'out-scene',
                                 'scene-15', 'landuse-21', 'cub', 'youtubeface',
                                 'noisymnist-30k', 'noisymnist-70k', 'multifashion'])
    parser.add_argument('--data_dir', type=str, default='./data')

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--encoder_hidden_dim', type=int, default=256)
    parser.add_argument('--encoder_num_layers', type=int, default=2)
    parser.add_argument('--encoder_type', type=str, default='shallow',
                        choices=['shallow', 'deep'],
                        help='shallow=original CRAFT 2-layer MLP, deep=MFLVC-style 4-layer 500-500-2000')
    parser.add_argument('--fusion_type', type=str, default='transformer',
                        choices=['transformer', 'concat_mlp', 'set_mlp'],
                        help='Fusion architecture: transformer=CRAFT baseline, concat_mlp=violates C1/C2 (B8), set_mlp=DeepSets-style strict C2 (B9)')
    parser.add_argument('--fusion_n_heads', type=int, default=4)
    parser.add_argument('--fusion_n_layers', type=int, default=1)
    parser.add_argument('--concat_mlp_hidden', type=int, default=-1,
                        help='ConcatMLPFusion hidden dim (-1=auto, 2*embed_dim)')
    parser.add_argument('--set_mlp_hidden', type=int, default=-1,
                        help='SetMLPFusion phi/rho hidden dim (-1=auto, 2*embed_dim)')
    parser.add_argument('--mask_strategy', type=str, default='learnable',
                        choices=['learnable', 'zero', 'mean', 'attention_mask'],
                        help='Missing-view fill strategy: learnable=learnable mask_token, zero=zero fill, mean=batch mean fill, attention_mask=key_padding_mask')
    parser.add_argument('--aggregation', type=str, default='cls',
                        choices=['cls', 'mean', 'max'],
                        help='Fusion output aggregation: cls=CLS token, mean/max=mean/max over observed view tokens (B7 ablation)')
    parser.add_argument('--cluster_temperature', type=float, default=0.1)

    parser.add_argument('--lambda_cluster', type=float, default=1.0)
    parser.add_argument('--lambda_entropy', type=float, default=5.0)
    parser.add_argument('--lambda_repr', type=float, default=1.0)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--entropy_sample_weight', type=float, default=0.1)
    parser.add_argument('--entropy_warmup_epochs', type=int, default=10)

    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--scheduler_T_max', type=int, default=100)
    parser.add_argument('--kmeans_n_init', type=int, default=20)

    parser.add_argument('--stage1_epochs', type=int, default=0,
                        help='Stage 1 pre-training epochs (0=single-stage training)')
    parser.add_argument('--stage2_lr', type=float, default=1e-4,
                        help='Stage 2 clustering fine-tune learning rate')
    parser.add_argument('--skip_stage2', action='store_true',
                        help='Skip Stage 2 and run K-Means on Stage 1 pretrained features only (ablation)')
    parser.add_argument('--repr_mode', type=str, default='simsiam',
                        choices=['simsiam', 'vicreg'],
                        help='Representation alignment mode: simsiam or vicreg')

    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze view encoder for the entire training (Stage 1 + Stage 2) to protect random projection')
    parser.add_argument('--stage2_freeze', type=str, default='none',
                        choices=['none', 'all', 'encoder_only'],
                        help='Stage 2 freeze policy: none=full fine-tune, all=only train cluster head, encoder_only=freeze encoder, train fusion+cluster head')
    parser.add_argument('--freeze_encoder_stage2', action='store_true',
                        help='(Legacy, equivalent to --stage2_freeze all)')

    # Incomplete MVC (training-time missing protocol)
    parser.add_argument('--missing_rate', type=float, default=0.0,
                        help='Training-time view-mask missing rate (0=complete)')
    parser.add_argument('--train_mask_protocol', type=str, default='completer',
                        choices=['completer', 'cell_level'],
                        help='Training-time mask protocol (active only when missing_rate>0): '
                             'completer=P1 (delete 1 view per selected sample, default for backward compat); '
                             'cell_level=P4 (cell-level Bernoulli, same distribution as incomplete_eval cell_level)')

    parser.add_argument('--selected_views', type=str, default=None,
                        help='Comma-separated view indices, e.g. "1,4" means use only Fou+KAR (default uses all views)')

    parser.add_argument('--num_masked_views', type=int, default=0,
                        help='Number of views to randomly mask per sample during training (0=no mask, legacy interface)')

    # Mask-Aware Fine-Tuning
    parser.add_argument('--train_mask_prob', type=float, default=0.0,
                        help='Probability per batch of enabling random view masking (0=disabled, 1.0=every batch)')
    parser.add_argument('--train_mask_max_views', type=int, default=-1,
                        help='Max views to mask per sample (-1=auto, V-2)')

    parser.add_argument('--resume_from', type=str, default=None,
                        help='Load weights from this ckpt for fine-tune (weights only, does not restore optimizer/epoch)')
    parser.add_argument('--skip_diag', action='store_true',
                        help='Skip random encoder diagnostic (not needed during fine-tune)')

    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=1)

    args = parser.parse_args()
    cfg = vars(args)

    # Backward compat: legacy --freeze_encoder_stage2 maps to --stage2_freeze all
    if cfg.get('freeze_encoder_stage2', False) and cfg.get('stage2_freeze', 'none') == 'none':
        cfg['stage2_freeze'] = 'all'

    if args.config is not None:
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        cfg.update(yaml_cfg)

    pl.seed_everything(cfg['seed'])

    print(f"\n{'='*60}")
    print(f"Loading dataset: {cfg['dataset']}")
    print(f"{'='*60}")

    features_list, labels = load_dataset(cfg['dataset'], cfg['data_dir'])
    if cfg.get('selected_views'):
        view_indices = [int(x) for x in cfg['selected_views'].split(',')]
        features_list = [features_list[i] for i in view_indices]
        print(f"  Selected views: {view_indices}, dims: {[f.shape[1] for f in features_list]}")

    num_views = len(features_list)
    view_dims = [f.shape[1] for f in features_list]
    num_clusters = len(np.unique(labels))
    num_samples = len(labels)

    data_module = MultiViewDataModule(
        features_list=features_list, labels=labels,
        batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
        noise_std=cfg.get('noise_std', 0.1), dropout_rate=cfg.get('dropout_rate', 0.1),
        train_ratio=cfg.get('train_ratio', 1.0), seed=cfg['seed'],
        missing_rate=cfg.get('missing_rate', 0.0),
        mask_protocol=cfg.get('train_mask_protocol', 'completer'),
    )

    print(f"\n{'='*60}")
    print(f"Building model")
    print(f"{'='*60}")

    model = build_model(num_views, view_dims, num_clusters, cfg)

    if cfg.get('resume_from'):
        model = load_weights_from_checkpoint(model, cfg['resume_from'])

    if cfg.get('freeze_encoder', False):
        model.freeze_encoder_only(True)
        print(">>> View encoders frozen for ENTIRE training (Stage 1 + Stage 2) <<<")

    if not cfg.get('skip_diag', False):
        diag_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        diagnose_random_encoder(model, data_module, diag_device, num_clusters)
        model = model.cpu()
    else:
        print("[Info] Skipping random encoder diagnostic (--skip_diag)")

    exp_name = cfg.get('exp_name') or f"{cfg['dataset']}_d{cfg['embed_dim']}"
    output_dir = os.path.join(cfg['output_dir'], 'CRAFT_MVC', exp_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger = TensorBoardLogger(output_dir, name="logs")

    # Save in exp root with fixed filename best.ckpt so downstream eval scripts can glob a stable path.
    # auto_insert_metric_name=False prevents the '/' in 'eval/ACC' from being treated as a sub-path separator.
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best',
        save_top_k=1,
        monitor='eval/ACC',
        mode='max',
        auto_insert_metric_name=False,
    )

    stage1_epochs = cfg.get('stage1_epochs', 0)

    if stage1_epochs > 0:
        # Stage 1: representation pre-training (cluster losses fully disabled)
        print(f"\n{'=' * 60}")
        print(f"Stage 1: Representation Pre-training ({stage1_epochs} epochs)")
        print(f"  Loss: SimSiam repr_loss + MSE recon_loss")
        print(f"  Cluster/Entropy: OFF")
        if cfg.get('freeze_encoder', False):
            print(f"  View Encoders: FROZEN (protecting random projection)")
        print(f"{'=' * 60}\n")

        original_lambda_cluster = model.lambda_cluster
        original_lambda_entropy = model.lambda_entropy
        model.lambda_cluster = 0.0
        model.lambda_entropy = 0.0
        model.set_total_epochs(stage1_epochs)

        stage1_callbacks = [
            EpochInfoCallback(),
            ClusteringEvalCallback(full_dataloader=data_module.full_dataloader(), eval_every=cfg['eval_every']),
        ]

        trainer_s1 = pl.Trainer(
            max_epochs=stage1_epochs,
            accelerator='gpu' if cfg['gpus'] > 0 and torch.cuda.is_available() else 'cpu',
            devices=cfg['gpus'] if cfg['gpus'] > 0 and torch.cuda.is_available() else 'auto',
            logger=TensorBoardLogger(output_dir, name="logs_stage1"),
            callbacks=stage1_callbacks,
            default_root_dir=output_dir,
            log_every_n_steps=10,
            enable_progress_bar=True,
            gradient_clip_val=cfg.get('gradient_clip_val', 1.0),
        )

        trainer_s1.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

        s1_eval = [c for c in stage1_callbacks if isinstance(c, ClusteringEvalCallback)][0]
        print(f"\n[Stage 1 Finished] Best ACC={s1_eval.best_acc:.4f} @ Epoch {s1_eval.best_epoch}")
        print(">>> Stage 1 ACC is based on a random cluster head; treat as diagnostic. True feature quality is reflected in Stage 2. <<<\n")

        # Transition: initialize cluster head from K-Means
        print(f"\n{'=' * 60}")
        print("Transition: K-Means Initialization on L2-normalized features")
        print(f"{'=' * 60}\n")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        val_features = []
        with torch.no_grad():
            for batch in data_module.full_dataloader():
                views, _ = batch[:2]
                views = [v.to(device) for v in views]
                encoded_views = model.encoder.encode_views(views)
                fused_rep = model.encoder.fuse(encoded_views)
                if isinstance(fused_rep, tuple):
                    fused_rep = fused_rep[0]
                val_features.append(fused_rep.cpu())

        X = torch.cat(val_features, dim=0)
        X_norm = torch.nn.functional.normalize(X, dim=-1, p=2).numpy()

        print("Running K-Means on L2-normalized features...")
        kmeans = KMeans(n_clusters=num_clusters, n_init=cfg.get('kmeans_n_init', 20), random_state=cfg['seed'])
        kmeans.fit(X_norm)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        model.init_cluster_head(centroids)

        # Stage 2: clustering fine-tune
        if cfg.get('skip_stage2', False):
            print(f"\n{'=' * 60}")
            print(f"[ABLATION] --skip_stage2: skipping Stage 2, evaluating Stage 1 features + K-Means directly")
            print(f"{'=' * 60}\n")
            eval_callback = s1_eval
            # On the normal path best.ckpt is saved by Stage 2's ModelCheckpoint when eval/ACC improves.
            # This branch never enters Stage 2, so save manually to capture the K-Means-initialized cluster_head.
            best_ckpt_path = os.path.join(output_dir, 'best.ckpt')
            trainer_s1.save_checkpoint(best_ckpt_path)
            print(f"[skip_stage2] saved checkpoint: {best_ckpt_path}")
        else:
            stage2_epochs = cfg['max_epochs'] - stage1_epochs
            stage2_lr = cfg.get('stage2_lr', 1e-4)
            stage2_freeze = cfg.get('stage2_freeze', 'none')

            print(f"\n{'=' * 60}")
            print(f"Stage 2: Clustering Fine-tuning ({stage2_epochs} epochs, lr={stage2_lr})")
            print(f"  Loss: cluster_consistency + entropy + repr_loss + recon_loss")
            print(f"  Freeze mode: {stage2_freeze}")
            print(f"{'=' * 60}\n")

            model.lambda_cluster = original_lambda_cluster
            model.lambda_entropy = original_lambda_entropy
            model.set_total_epochs(stage2_epochs)

            model.optim_kwargs['lr'] = stage2_lr
            model.optim_kwargs['max_epochs'] = stage2_epochs

            if stage2_freeze == 'all':
                model.freeze_backbone(True)
            elif stage2_freeze == 'encoder_only':
                model.freeze_encoder_only(True)

            stage2_callbacks = [
                EpochInfoCallback(),
                ClusteringEvalCallback(full_dataloader=data_module.full_dataloader(), eval_every=cfg['eval_every']),
                checkpoint_callback,
            ]

            trainer_s2 = pl.Trainer(
                max_epochs=stage2_epochs,
                accelerator='gpu' if cfg['gpus'] > 0 and torch.cuda.is_available() else 'cpu',
                devices=cfg['gpus'] if cfg['gpus'] > 0 and torch.cuda.is_available() else 'auto',
                logger=logger,
                callbacks=stage2_callbacks,
                default_root_dir=output_dir,
                log_every_n_steps=10,
                enable_progress_bar=True,
                gradient_clip_val=cfg.get('gradient_clip_val', 1.0),
            )

            trainer_s2.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

            eval_callback = [c for c in stage2_callbacks if isinstance(c, ClusteringEvalCallback)][0]

            if s1_eval.best_acc > eval_callback.best_acc:
                eval_callback.best_acc = s1_eval.best_acc
                eval_callback.best_metrics = s1_eval.best_metrics
                eval_callback.best_epoch = s1_eval.best_epoch

    else:
        # Single-stage training
        print(f"\n{'='*60}")
        print(f"Starting single-stage training")
        if cfg.get('train_mask_prob', 0) > 0:
            print(f"  Mask-Aware Fine-Tuning: ON")
            print(f"    train_mask_prob = {cfg['train_mask_prob']}")
            print(f"    train_mask_max_views = {model.train_mask_max_views}")
        print(f"{'='*60}\n")

        callbacks = [
            EpochInfoCallback(),
            ClusteringEvalCallback(full_dataloader=data_module.full_dataloader(), eval_every=cfg['eval_every']),
            checkpoint_callback
        ]

        trainer = pl.Trainer(
            max_epochs=cfg['max_epochs'],
            accelerator='gpu' if cfg['gpus'] > 0 and torch.cuda.is_available() else 'cpu',
            devices=cfg['gpus'] if cfg['gpus'] > 0 and torch.cuda.is_available() else 'auto',
            logger=logger,
            callbacks=callbacks,
            default_root_dir=output_dir,
            log_every_n_steps=10,
            enable_progress_bar=True,
            gradient_clip_val=cfg.get('gradient_clip_val', 1.0),
        )

        trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
        eval_callback = [c for c in callbacks if isinstance(c, ClusteringEvalCallback)][0]

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Final evaluation")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    last_metrics = evaluate_model(model, data_module.full_dataloader(), device)

    print(f"\n--- Last Epoch ---")
    print(f"  ACC: {last_metrics['ACC']:.4f}")
    print(f"  NMI: {last_metrics['NMI']:.4f}")
    print(f"  ARI: {last_metrics['ARI']:.4f}")

    print(f"\n--- Best during training (Epoch {eval_callback.best_epoch}) ---")
    print(f"  ACC: {eval_callback.best_metrics.get('ACC', 0):.4f}")
    print(f"  NMI: {eval_callback.best_metrics.get('NMI', 0):.4f}")
    print(f"  ARI: {eval_callback.best_metrics.get('ARI', 0):.4f}")

    if eval_callback.best_acc > last_metrics['ACC']:
        final_metrics = eval_callback.best_metrics
        final_source = f"best_epoch_{eval_callback.best_epoch}"
    else:
        final_metrics = last_metrics
        final_source = "last_epoch"

    print(f"\n*** Final Results (from {final_source}) ***")
    print(f"  ACC: {final_metrics['ACC']:.4f}")
    print(f"  NMI: {final_metrics['NMI']:.4f}")
    print(f"  ARI: {final_metrics['ARI']:.4f}")

    results = {
        'final': {k: float(v) for k, v in final_metrics.items()},
        'final_source': final_source,
        'last_epoch': {k: float(v) for k, v in last_metrics.items()},
        'best_during_training': {k: float(v) for k, v in eval_callback.best_metrics.items()},
        'best_epoch': eval_callback.best_epoch,
    }
    results_path = os.path.join(output_dir, 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
