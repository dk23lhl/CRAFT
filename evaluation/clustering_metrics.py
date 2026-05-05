"""
Clustering evaluation metrics.

Standard trio: ACC (Hungarian matching), NMI, ARI.
This is the unified evaluation standard in multi-view clustering.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from typing import Dict, Tuple


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Clustering accuracy with Hungarian matching.

    Cluster labels have no fixed correspondence to ground-truth labels; the
    Hungarian algorithm finds the optimal matching before computing accuracy.
    """
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    assert y_pred.size == y_true.size, (
        f"Size mismatch: {y_pred.size} vs {y_true.size}"
    )

    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        cost_matrix[y_pred[i], y_true[i]] += 1

    # Hungarian algorithm: maximize matching = minimize (max - matrix).
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)

    return cost_matrix[row_ind, col_ind].sum() / y_pred.size


def evaluate_clustering(y_true: np.ndarray,
                        y_pred: np.ndarray) -> Dict[str, float]:
    """
    Full clustering evaluation.

    Returns:
        dict with keys: ACC, NMI, ARI
    """
    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    return {
        'ACC': acc,
        'NMI': nmi,
        'ARI': ari
    }


@torch.no_grad()
def evaluate_model(model,
                   data_loader: torch.utils.data.DataLoader,
                   device: torch.device) -> Dict[str, float]:
    """
    End-to-end evaluation of the model's clustering performance.

    Args:
        model: SyncMaskMVC model
        data_loader: DataLoader over the full dataset (augment=False)
        device: compute device

    Returns:
        metrics: {'ACC': ..., 'NMI': ..., 'ARI': ...}
    """
    model.eval()
    all_preds = []
    all_labels = []

    for batch in data_loader:
        # augment=False mode returns (views, labels).
        if len(batch) == 2:
            views, labels = batch
        elif len(batch) == 3:
            views, _, labels = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        views = [v.to(device) for v in views]

        encoded = model.encoder.encode_views(views)
        z = model.encoder.fuse(encoded)
        p = model.cluster_head(z)
        preds = torch.argmax(p, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return evaluate_clustering(all_labels, all_preds)


@torch.no_grad()
def extract_features_and_labels(model,
                                data_loader: torch.utils.data.DataLoader,
                                device: torch.device
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels for the full dataset.

    Used for t-SNE visualization or other analyses.
    """
    model.eval()
    all_features = []
    all_labels = []

    for batch in data_loader:
        if len(batch) == 2:
            views, labels = batch
        elif len(batch) == 3:
            views, _, labels = batch
        else:
            raise ValueError(f"Unexpected batch format")

        views = [v.to(device) for v in views]

        encoded = model.encoder.encode_views(views)
        z = model.encoder.fuse(encoded)

        all_features.append(z.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.concatenate(all_features, axis=0), np.array(all_labels)
