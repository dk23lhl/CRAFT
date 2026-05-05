"""
Loaders for common multi-view clustering datasets.

Supported datasets:
- Caltech-101-20: 6 views, 20 classes, 2386 samples
- MNIST-USPS: 2 views, 10 classes, 5000 samples
- Scene-15: 3 views, 15 classes, 4485 samples
- HandWritten: 6 views, 10 classes, 2000 samples
- NGs: 3 views, 5 classes, 500 samples
- BBC: 4 views, 5 classes, 685 samples
- UCI-digit: 3 views, 10 classes, 2000 samples (3 of 6 views: 64/76/216)
- Hdigit: 2 views, 10 classes, 10000 samples (MNIST 784d + USPS 256d)
- ALOI-100: 4 views, 100 classes, 10800 samples
- LandUse-21: 3 views, 21 classes, 2100 samples (GIST 20d / PHOG 59d / LBP 40d)
- CUB: 2 views, 10 classes, 600 samples (GoogLeNet 1024d / Doc2Vec 300d)
- YouTubeFace: 5 views, 31 classes, 101499 samples (64/512/64/647/838d, npy format)
- NoisyMNIST: 2 views, 10 classes, 70000 samples (784d / 784d, npz format)
  - supports stratified subsample to 30000 samples
- MultiFashion: 3 views, 10 classes, 10000 samples (784d / 784d / 784d, npz format)

Dataset format: .mat / .npz / .npy files in the standard MVC benchmark layout.
"""

import os
import numpy as np
import scipy.io as sio
from typing import Tuple, List, Optional


def normalize_features(features: np.ndarray, method: str = "standard") -> np.ndarray:
    """
    Feature normalization.

    Args:
        features: [N, d] feature matrix
        method: "standard" (zero mean, unit variance), "minmax" (0-1), or "l2"
    """
    if method == "standard":
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        return (features - mean) / std
    elif method == "minmax":
        fmin = features.min(axis=0, keepdims=True)
        fmax = features.max(axis=0, keepdims=True)
        return (features - fmin) / (fmax - fmin + 1e-8)
    elif method == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return features / norms
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def load_caltech101_20(data_dir: str,
                       normalize: str = "standard"
                       ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load Caltech-101-20 dataset.

    6 views:
      0: WM (Wavelet Moments) - 48d
      1: CENTRIST - 254d
      2: LBP (Local Binary Pattern) - 928d
      3: GIST - 512d
      4: HOG (Histogram of Oriented Gradients) - 1984d
      5: Gabor - 48d

    Args:
        data_dir: directory containing the .mat file
        normalize: normalization method

    Returns:
        features_list: list of 6 view feature matrices
        labels: [N] label array, zero-based
    """
    possible_names = [
        'Caltech101-20.mat',
        'caltech101-20.mat',
        'Caltech101_20.mat',
        'caltech-101-20.mat',
        'Caltech-101-20.mat',
    ]

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"Caltech-101-20 dataset not found in {data_dir}. "
            f"Tried: {possible_names}\n"
            f"Please download the dataset and place the .mat file in {data_dir}"
        )

    mat = sio.loadmat(mat_path)

    # Features typically live under 'X' or 'x' as a cell array.
    features_list = []
    if 'X' in mat:
        X = mat['X']
        for i in range(X.shape[1]):
            feat = X[0, i].astype(np.float32)
            if normalize:
                feat = normalize_features(feat, normalize)
            features_list.append(feat)
    elif 'x1' in mat:
        # Some variants store views as x1, x2, ...
        i = 1
        while f'x{i}' in mat:
            feat = mat[f'x{i}'].astype(np.float32)
            if normalize:
                feat = normalize_features(feat, normalize)
            features_list.append(feat)
            i += 1
    else:
        raise KeyError(
            f"Cannot find feature keys in .mat file. Available keys: {list(mat.keys())}"
        )

    if 'Y' in mat:
        labels = mat['Y'].flatten().astype(np.int64)
    elif 'y' in mat:
        labels = mat['y'].flatten().astype(np.int64)
    elif 'gt' in mat:
        labels = mat['gt'].flatten().astype(np.int64)
    else:
        raise KeyError(
            f"Cannot find label key in .mat file. Available keys: {list(mat.keys())}"
        )

    labels = labels - labels.min()

    print(f"[Caltech-101-20] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_mnist_usps(data_dir: str,
                    n_samples: int = 5000,
                    normalize: str = "standard",
                    seed: int = 42
                    ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load MNIST-USPS dataset.

    2 views:
      0: MNIST handwritten digits - 784d
      1: USPS handwritten digits - 256d
    """
    possible_names = ['MNIST-USPS.mat', 'mnist_usps.mat', 'MNISTUSPS.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"MNIST-USPS dataset not found in {data_dir}. "
            f"Tried: {possible_names}"
        )

    mat = sio.loadmat(mat_path)

    features_list = []
    if 'X' in mat:
        X = mat['X']
        for i in range(X.shape[1]):
            feat = X[0, i].astype(np.float32)
            features_list.append(feat)
    elif 'X1' in mat:
        features_list.append(mat['X1'].astype(np.float32))
        features_list.append(mat['X2'].astype(np.float32))
    else:
        raise KeyError(f"Cannot find feature keys. Available: {list(mat.keys())}")

    if 'Y' in mat:
        labels = mat['Y'].flatten().astype(np.int64)
    elif 'y' in mat:
        labels = mat['y'].flatten().astype(np.int64)
    elif 'gt' in mat:
        labels = mat['gt'].flatten().astype(np.int64)
    else:
        raise KeyError(f"Cannot find label key. Available: {list(mat.keys())}")

    labels = labels - labels.min()

    N = features_list[0].shape[0]
    if n_samples and n_samples < N:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, n_samples, replace=False)
        features_list = [f[idx] for f in features_list]
        labels = labels[idx]

    if normalize:
        features_list = [normalize_features(f, normalize) for f in features_list]

    print(f"[MNIST-USPS] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")

    return features_list, labels


def load_scene15(data_dir: str,
                 normalize: str = "standard"
                 ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load Scene-15 dataset.

    3 views:
      0: GIST - 20d
      1: PHOG - 59d
      2: LBP - 40d
    """
    possible_names = ['Scene-15.mat', 'scene15.mat', 'Scene15.mat', 'Scene_15.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(f"Scene-15 dataset not found in {data_dir}")

    mat = sio.loadmat(mat_path)

    features_list = []
    if 'X' in mat:
        X = mat['X']
        for i in range(X.shape[1]):
            feat = X[0, i].astype(np.float32)
            if normalize:
                feat = normalize_features(feat, normalize)
            features_list.append(feat)
    else:
        raise KeyError(f"Cannot find feature keys. Available: {list(mat.keys())}")

    if 'Y' in mat:
        labels = mat['Y'].flatten().astype(np.int64)
    elif 'y' in mat:
        labels = mat['y'].flatten().astype(np.int64)
    else:
        raise KeyError(f"Cannot find label key. Available: {list(mat.keys())}")

    labels = labels - labels.min()

    print(f"[Scene-15] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")

    return features_list, labels


def load_handwritten(data_dir: str,
                     normalize: str = "standard"
                     ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load HandWritten dataset.

    6 views:
      0: Pix (Pixel averages) - 240d
      1: Fou (Fourier coefficients) - 76d
      2: Fac (Profile correlations) - 216d
      3: ZER (Zernike moments) - 47d
      4: KAR (Karhunen-Loeve coefficients) - 64d
      5: MOR (Morphological features) - 6d

    Args:
        data_dir: directory containing the .mat file
        normalize: normalization method

    Returns:
        features_list: list of 6 view feature matrices
        labels: [N] label array, zero-based
    """
    possible_names = ['HandWritten.mat', 'handwritten.mat', 'Handwritten.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"HandWritten dataset not found in {data_dir}. "
            f"Tried: {possible_names}"
        )

    mat = sio.loadmat(mat_path)

    features_list = []
    if 'X' in mat:
        X = mat['X']
        for i in range(X.shape[1]):
            feat = X[0, i].astype(np.float32)
            if normalize:
                feat = normalize_features(feat, normalize)
            features_list.append(feat)
    else:
        raise KeyError(
            f"Cannot find feature keys in .mat file. Available keys: {list(mat.keys())}"
        )

    if 'Y' in mat:
        labels = mat['Y'].flatten().astype(np.int64)
    elif 'y' in mat:
        labels = mat['y'].flatten().astype(np.int64)
    else:
        raise KeyError(
            f"Cannot find label key in .mat file. Available keys: {list(mat.keys())}"
        )

    labels = labels - labels.min()

    print(f"[HandWritten] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_ngs(data_dir: str,
             normalize: str = "l2"
             ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load NGs (20Newsgroups subset) dataset.

    3 views, each 2000d (TF-IDF features).
    500 samples, 5 classes.

    .mat layout: X=(1,3) cell array, Y=(500,1)
    """
    possible_names = ['NGs.mat', 'ngs.mat', 'NGS.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"NGs dataset not found in {data_dir}. Tried: {possible_names}"
        )

    mat = sio.loadmat(mat_path)
    X = mat['X'][0]
    features_list = []
    for i in range(len(X)):
        feat = X[i].astype(np.float32)
        if hasattr(feat, 'toarray'):
            feat = feat.toarray()
        if normalize:
            feat = normalize_features(feat, normalize)
        features_list.append(feat)

    labels = mat['Y'].flatten().astype(np.int64)
    if labels.min() == 1:
        labels -= 1

    # PCA pre-reduction: high-dim sparse TF-IDF features are incompatible with the MLP encoder.
    from sklearn.decomposition import PCA
    pca_dim = 200
    for i in range(len(features_list)):
        pca = PCA(n_components=pca_dim, random_state=42)
        features_list[i] = pca.fit_transform(features_list[i]).astype(np.float32)

    print(f"[NGs] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_bbc(data_dir: str,
             normalize: str = "l2"
             ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load BBC dataset.

    4 views (4659/4633/4665/4684d), TF-IDF text features.
    685 samples, 5 classes.

    Note: this is BBC (685 samples, 4 views), not BBCSport (544 samples, 2 views).

    .mat layout: X=(1,4) cell array, Y=(685,1)
    """
    possible_names = ['BBC.mat', 'bbc.mat', 'BBC4view.mat', 'BBC4View_685.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"BBC dataset not found in {data_dir}. Tried: {possible_names}"
        )

    mat = sio.loadmat(mat_path)
    X = mat['X'][0]
    features_list = []
    for i in range(len(X)):
        feat = X[i].astype(np.float32)
        if hasattr(feat, 'toarray'):
            feat = feat.toarray()
        if normalize:
            feat = normalize_features(feat, normalize)
        features_list.append(feat)

    labels = mat['Y'].flatten().astype(np.int64)
    if labels.min() == 1:
        labels -= 1

    # PCA pre-reduction: high-dim sparse TF-IDF features are incompatible with the MLP encoder.
    from sklearn.decomposition import PCA
    pca_dim = 500
    for i in range(len(features_list)):
        n_comp = min(pca_dim, features_list[i].shape[0] - 1, features_list[i].shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        features_list[i] = pca.fit_transform(features_list[i]).astype(np.float32)

    print(f"[BBC] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_uci_digit(data_dir: str,
                   normalize: str = "standard"
                   ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load UCI-digit dataset (3-view variant, matching the EPFMVC paper).

    The original UCI_Digits.mat contains 6 views (same six features as HandWritten):
      view 0: Pix (240d)
      view 1: Fou (76d)
      view 2: Fac (216d)
      view 3: ZER (47d)
      view 4: KAR (64d)
      view 5: MOR (6d)

    EPFMVC selects 3 views: 64d / 76d / 216d, i.e. view 4 (KAR) / view 1 (FOU) / view 2 (FAC).

    2000 samples, 10 classes.

    .mat layout: fea=(1,6) cell array, gt=(2000,1)
    """
    possible_names = ['UCI_Digits.mat', 'UCI.mat', 'uci_digits.mat', 'UCI_digit.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"UCI-digit dataset not found in {data_dir}. Tried: {possible_names}"
        )

    mat = sio.loadmat(mat_path)
    fea = mat['fea'][0]

    # EPFMVC uses 3 views: 64d (KAR) / 76d (FOU) / 216d (FAC) = original view 4 / 1 / 2.
    selected_views = [4, 1, 2]
    features_list = []
    for idx in selected_views:
        feat = fea[idx].astype(np.float32)
        if normalize:
            feat = normalize_features(feat, normalize)
        features_list.append(feat)

    labels = mat['gt'].flatten().astype(np.int64)
    if labels.min() == 1:
        labels -= 1

    print(f"[UCI-digit] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    print(f"  Selected from 6-view file: view indices {selected_views}")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_hdigit(data_dir: str,
                normalize: str = "standard"
                ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load Hdigit dataset.

    2 views:
      0: MNIST handwritten digits - 784d
      1: USPS handwritten digits - 256d

    10000 samples, 10 classes.

    .mat format quirks:
      data=(1,2) cell array, each entry shape=(features, samples) -> needs transpose
      truelabel=(1,2) cell array, both copies are identical, take the first
      Labels are 1..10, subtract 1.
    """
    possible_names = ['Hdigit.mat', 'hdigit.mat', 'HDigit.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"Hdigit dataset not found in {data_dir}. Tried: {possible_names}"
        )

    mat = sio.loadmat(mat_path)
    d = mat['data'][0]

    # data[i] shape = (features, samples), transpose to (samples, features)
    features_list = []
    for i in range(len(d)):
        feat = d[i].T.astype(np.float32)
        if normalize:
            feat = normalize_features(feat, normalize)
        features_list.append(feat)

    # truelabel is (1,2) cell array with identical copies; take the first.
    labels = mat['truelabel'][0][0].flatten().astype(np.int64)
    if labels.min() == 1:
        labels -= 1

    print(f"[Hdigit] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_aloi100(data_dir: str,
                 normalize: str = "standard"
                 ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load ALOI-100 dataset.

    4 views:
      0: 77d  (RGB color histograms)   - range [0.15, 0.82], dense
      1: 13d  (Haralick texture)       - range [-2.9e4, 8.2e13], must be normalized
      2: 64d  (Color similarity)       - range [0, 0.48], 34% sparse
      3: 125d (HSB color histograms)   - range [0, 1.0], 64% sparse

    10800 samples, 100 classes (108 per class).

    Note: view 1 (13d Haralick) is 10+ orders of magnitude beyond the others; standard
    normalization is mandatory, otherwise this view drowns out every other signal.

    .mat layout: X=(1,4) cell array, Y=(10800,1), labels are 1-based (1..100).
    """
    possible_names = ['slow_ALOI-100.mat', 'ALOI-100.mat', 'aloi-100.mat',
                      'ALOI100.mat', 'aloi100.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"ALOI-100 dataset not found in {data_dir}. Tried: {possible_names}"
        )

    mat = sio.loadmat(mat_path)
    X = mat['X'][0]

    features_list = []
    for i in range(len(X)):
        feat = X[i].astype(np.float32)
        if normalize:
            feat = normalize_features(feat, normalize)
        features_list.append(feat)

    labels = mat['Y'].flatten().astype(np.int64)
    if labels.min() == 1:
        labels -= 1

    print(f"[ALOI-100] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_outscene(data_dir: str,
                  normalize: str = "standard"
                  ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load Out-Scene (Outdoor Scene) dataset.

    4 views:
      0: GIST - 512d    range [0.0002, 0.4243], dense
      1: HOG  - 432d    range [0, 0.2], dense
      2: LBP  - 256d    raw uint16 range [0, 45567], must be cast to float
      3: Gabor - 48d    range [-0.0188, 10.4746], dense

    2688 samples, 8 classes.

    Note: view 2 (LBP) raw data is uint16 and far larger in scale than the other views,
    so standard normalization is mandatory.

    .mat layout: X=(1,4) cell array, Y=(2688,1), labels are 1-based (1..8).
    """
    possible_names = ['Out_Scene.mat', 'outdoor_scene.mat', 'OutdoorScene.mat',
                      'out-scene.mat', 'Out-Scene.mat', 'Scene.mat',
                      'out_scene.mat', 'Outdoor_Scene.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"Out-Scene dataset not found in {data_dir}. Tried: {possible_names}"
        )

    mat = sio.loadmat(mat_path)
    X = mat['X'][0]

    features_list = []
    for i in range(len(X)):
        feat = X[i].astype(np.float32)  # uint16 -> float32
        if normalize:
            feat = normalize_features(feat, normalize)
        features_list.append(feat)

    # PCA pre-reduction for views > embed_dim, to avoid losing cluster structure
    # via the MLP's random projection.
    from sklearn.decomposition import PCA
    pca_dim = 256
    for i in range(len(features_list)):
        if features_list[i].shape[1] > pca_dim:
            orig_dim = features_list[i].shape[1]
            pca = PCA(n_components=pca_dim, random_state=42)
            features_list[i] = pca.fit_transform(features_list[i]).astype(np.float32)
            print(f"  View {i}: PCA {orig_dim}d -> {pca_dim}d")

    labels = mat['Y'].flatten().astype(np.int64)
    if labels.min() == 1:
        labels -= 1

    print(f"[Out-Scene] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_landuse21(data_dir: str,
                   normalize: str = "standard"
                   ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load LandUse-21 dataset.

    3 views:
      0: GIST - 20d     range [0.13, 5.86], dense
      1: PHOG - 59d     range [0.00, 16.02], dense, 0.1% sparse
      2: LBP  - 40d     range [0.00, 13.98], dense, 0.1% sparse

    2100 samples, 21 classes (100 per class).

    Same feature types and dimensions as Scene-15 (GIST/PHOG/LBP, 20/59/40d):
    pattern A (all views <= embed_dim), the very-low-dim regime.

    .mat layout: X=(1,3) cell array, Y=(2100,1), labels are 1-based (1..21).
    """
    possible_names = ['LandUse_21.mat', 'LandUse-21.mat', 'landuse_21.mat',
                      'landuse-21.mat', 'LandUse21.mat', 'landuse21.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"LandUse-21 dataset not found in {data_dir}. Tried: {possible_names}"
        )

    mat = sio.loadmat(mat_path)
    X = mat['X'][0]

    features_list = []
    for i in range(len(X)):
        feat = X[i].astype(np.float32)
        if normalize:
            feat = normalize_features(feat, normalize)
        features_list.append(feat)

    labels = mat['Y'].flatten().astype(np.int64)
    if labels.min() == 1:
        labels -= 1

    print(f"[LandUse-21] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_cub(data_dir: str,
             normalize: str = "standard"
             ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load CUB dataset (multi-view clustering variant).

    2 views:
      0: GoogLeNet image features - 1024d
      1: Doc2Vec text description features - 300d

    600 samples, 10 classes.

    .mat layout: X=(1,2) cell array, gt=(600,1), labels are 1-based.
    Filename: cub_googlenet_doc2vec_c10.mat
    """
    possible_names = ['cub_googlenet_doc2vec_c10.mat', 'CUB.mat', 'cub.mat']

    mat_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is None:
        raise FileNotFoundError(
            f"CUB dataset not found in {data_dir}. Tried: {possible_names}"
        )

    mat = sio.loadmat(mat_path)

    features_list = []
    if 'X' in mat:
        X = mat['X']
        for i in range(X.shape[1]):
            feat = X[0, i].astype(np.float32)
            if normalize:
                feat = normalize_features(feat, normalize)
            features_list.append(feat)
    else:
        raise KeyError(f"Cannot find feature keys. Available: {list(mat.keys())}")

    # CUB stores labels under 'gt'.
    if 'gt' in mat:
        labels = mat['gt'].flatten().astype(np.int64)
    elif 'Y' in mat:
        labels = mat['Y'].flatten().astype(np.int64)
    elif 'y' in mat:
        labels = mat['y'].flatten().astype(np.int64)
    else:
        raise KeyError(f"Cannot find label key. Available: {list(mat.keys())}")

    labels = labels - labels.min()

    print(f"[CUB] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_youtubeface(data_dir: str,
                     normalize: str = "standard"
                     ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load YouTubeFace (YTF31) dataset.

    5 views:
      0: 64d
      1: 512d
      2: 64d
      3: 647d
      4: 838d

    101499 samples, 31 classes.

    Class imbalance: min 1493, max 27022.

    Two formats are accepted (probed in this order):

    Format 1 (preferred): .mat file (scipy v7.2)
      YoutubeFace_sel_fea.mat
      - X: (5, 1) cell array, each cell (101499, d_v)
      - Y: (101499, 1) uint8, range 1..31

    Format 2 (fallback): npy folder
      YoutubeFace/train_x/0.npy ~ 4.npy  (float64)
      YoutubeFace/test_x/0.npy ~ 4.npy   (float64)
      YoutubeFace/train_y.npy, test_y.npy (int64)
    """
    mat_names = ['YoutubeFace_sel_fea.mat', 'YouTubeFace_sel_fea.mat',
                 'youtubeface_sel_fea.mat', 'YTF_sel_fea.mat',
                 'YoutubeFace.mat', 'YouTubeFace.mat', 'ytf.mat']
    mat_path = None
    for name in mat_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            mat_path = path
            break

    if mat_path is not None:
        print(f"[YouTubeFace] Loading from .mat: {mat_path}")
        mat = sio.loadmat(mat_path)

        features_list = []
        if 'X' in mat:
            X = mat['X']
            # X may be (5, 1) or (1, 5); flatten before iterating.
            X_flat = X.ravel()
            for i in range(len(X_flat)):
                feat = X_flat[i].astype(np.float32)
                if normalize:
                    feat = normalize_features(feat, normalize)
                features_list.append(feat)
        else:
            raise KeyError(
                f"Cannot find 'X' key in .mat. Available: {list(mat.keys())}"
            )

        if 'Y' in mat:
            labels = mat['Y'].flatten().astype(np.int64)
        elif 'y' in mat:
            labels = mat['y'].flatten().astype(np.int64)
        elif 'gt' in mat:
            labels = mat['gt'].flatten().astype(np.int64)
        else:
            raise KeyError(
                f"Cannot find label key in .mat. Available: {list(mat.keys())}"
            )

        labels = labels - labels.min()

        print(f"[YouTubeFace] Loaded: {len(labels)} samples, "
              f"{len(features_list)} views, {len(np.unique(labels))} classes")
        for i, f in enumerate(features_list):
            print(f"  View {i}: {f.shape[1]}d")

        from collections import Counter
        counts = Counter(labels.tolist())
        min_c, max_c = min(counts.values()), max(counts.values())
        print(f"  Class imbalance: min={min_c}, max={max_c}, "
              f"ratio={max_c/min_c:.1f}x")

        return features_list, labels

    possible_dirs = ['YoutubeFace', 'YouTubeFace', 'youtubeface', 'ytf', 'YTF']

    root = None
    for d in possible_dirs:
        path = os.path.join(data_dir, d)
        if os.path.isdir(path):
            root = path
            break

    if root is None:
        raise FileNotFoundError(
            f"YouTubeFace dataset not found in {data_dir}. "
            f"Tried .mat files: {mat_names}\n"
            f"Tried subdirectories: {possible_dirs}"
        )

    num_views = 5

    features_list = []
    for v in range(num_views):
        train_path = os.path.join(root, 'train_x', f'{v}.npy')
        test_path = os.path.join(root, 'test_x', f'{v}.npy')

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"View file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"View file not found: {test_path}")

        train_v = np.load(train_path).astype(np.float32)
        test_v = np.load(test_path).astype(np.float32)
        feat = np.concatenate([train_v, test_v], axis=0)

        if normalize:
            feat = normalize_features(feat, normalize)
        features_list.append(feat)

    train_y = np.load(os.path.join(root, 'train_y.npy'))
    test_y = np.load(os.path.join(root, 'test_y.npy'))
    labels = np.concatenate([train_y, test_y], axis=0).astype(np.int64)

    labels = labels - labels.min()

    print(f"[YouTubeFace] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    from collections import Counter
    counts = Counter(labels.tolist())
    min_c, max_c = min(counts.values()), max(counts.values())
    print(f"  Class imbalance: min={min_c}, max={max_c}, ratio={max_c/min_c:.1f}x")

    return features_list, labels


def load_noisy_mnist(data_dir: str,
                     normalize: str = "standard",
                     n_samples: Optional[int] = None,
                     seed: int = 42
                     ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load NoisyMNIST dataset.

    2 views:
      0: 784d (noisy MNIST variant 1)
      1: 784d (noisy MNIST variant 2)

    70000 samples, 10 classes.
    n_samples enables stratified subsampling (e.g. 30000).

    Note: raw labels are 1..10; loader subtracts 1 to get 0..9.

    .npz layout: view_0, view_1, labels, n_views
    """
    possible_names = ['NoisyMNIST.npz', 'noisymnist.npz', 'Noisy_MNIST.npz',
                      'noisy_mnist.npz']

    npz_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            npz_path = path
            break

    if npz_path is None:
        raise FileNotFoundError(
            f"NoisyMNIST dataset not found in {data_dir}. Tried: {possible_names}"
        )

    data = np.load(npz_path, allow_pickle=True)

    n_views = int(data['n_views'])
    features_list = []
    for i in range(n_views):
        feat = data[f'view_{i}'].astype(np.float32)
        features_list.append(feat)

    labels = data['labels'].flatten().astype(np.int64)
    labels = labels - labels.min()

    if n_samples is not None and n_samples < len(labels):
        rng = np.random.RandomState(seed)
        unique_labels = np.unique(labels)
        n_per_class = n_samples // len(unique_labels)
        remainder = n_samples - n_per_class * len(unique_labels)

        selected_idx = []
        for i, lbl in enumerate(unique_labels):
            class_idx = np.where(labels == lbl)[0]
            # First `remainder` classes get one extra sample to hit the exact total.
            n_select = n_per_class + (1 if i < remainder else 0)
            n_select = min(n_select, len(class_idx))
            chosen = rng.choice(class_idx, n_select, replace=False)
            selected_idx.append(chosen)

        idx = np.concatenate(selected_idx)
        rng.shuffle(idx)
        features_list = [f[idx] for f in features_list]
        labels = labels[idx]

    if normalize:
        features_list = [normalize_features(f, normalize) for f in features_list]

    print(f"[NoisyMNIST] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_multi_fashion(data_dir: str,
                       normalize: str = "standard"
                       ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load MultiFashion dataset.

    3 views:
      0: 784d (Fashion-MNIST variant 1)
      1: 784d (Fashion-MNIST variant 2)
      2: 784d (Fashion-MNIST variant 3)

    10000 samples, 10 classes.

    Labels are already 0..9.

    .npz layout: view_0, view_1, view_2, labels, n_views
    """
    possible_names = ['Multi-Fashion.npz', 'MultiFashion.npz', 'multi_fashion.npz',
                      'multifashion.npz', 'Multi_Fashion.npz']

    npz_path = None
    for name in possible_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            npz_path = path
            break

    if npz_path is None:
        raise FileNotFoundError(
            f"MultiFashion dataset not found in {data_dir}. Tried: {possible_names}"
        )

    data = np.load(npz_path, allow_pickle=True)

    n_views = int(data['n_views'])
    features_list = []
    for i in range(n_views):
        feat = data[f'view_{i}'].astype(np.float32)
        features_list.append(feat)

    labels = data['labels'].flatten().astype(np.int64)
    labels = labels - labels.min()  # already 0-based; keep as a safety net

    if normalize:
        features_list = [normalize_features(f, normalize) for f in features_list]

    print(f"[MultiFashion] Loaded: {len(labels)} samples, "
          f"{len(features_list)} views, {len(np.unique(labels))} classes")
    for i, f in enumerate(features_list):
        print(f"  View {i}: {f.shape[1]}d")

    return features_list, labels


def load_dataset(name: str, data_dir: str, **kwargs) -> Tuple[List[np.ndarray], np.ndarray]:
    """Unified dataset loading entry point."""
    loaders = {
        'caltech101-20': load_caltech101_20,
        'caltech': load_caltech101_20,
        'mnist-usps': load_mnist_usps,
        'mnist_usps': load_mnist_usps,
        'scene-15': load_scene15,
        'scene15': load_scene15,
        'handwritten': load_handwritten,
        'ngs': load_ngs,
        'bbc': load_bbc,
        'uci-digit': load_uci_digit,
        'uci_digit': load_uci_digit,
        'hdigit': load_hdigit,
        'aloi-100': load_aloi100,
        'aloi100': load_aloi100,
        'out-scene': load_outscene,
        'outscene': load_outscene,
        'landuse-21': load_landuse21,
        'landuse21': load_landuse21,
        'cub': load_cub,
        'youtubeface': load_youtubeface,
        'ytf': load_youtubeface,
        # NoisyMNIST: 70k full / 30k stratified subsample
        'noisymnist': load_noisy_mnist,
        'noisy-mnist': load_noisy_mnist,
        'noisy_mnist': load_noisy_mnist,
        'noisymnist-30k': lambda data_dir, **kw: load_noisy_mnist(data_dir, n_samples=30000, **kw),
        'noisymnist-70k': load_noisy_mnist,
        'multifashion': load_multi_fashion,
        'multi-fashion': load_multi_fashion,
        'multi_fashion': load_multi_fashion,
    }

    name_lower = name.lower()
    if name_lower not in loaders:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(loaders.keys())}"
        )

    return loaders[name_lower](data_dir, **kwargs)
