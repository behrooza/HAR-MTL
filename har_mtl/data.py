"""UCI-HAR dataset download and preparation."""
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import DEFAULT_SEED

UCI_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATASET_DIRNAME = "UCI HAR Dataset"

ACTIVITY_LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def download_uci_har(data_dir="data"):
    """Download and extract UCI-HAR into ``data_dir`` (skipped if already present).

    Returns the path of the extracted ``UCI HAR Dataset`` folder.
    """
    data_dir = Path(data_dir)
    root = data_dir / DATASET_DIRNAME
    if root.is_dir():
        return root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "uci_har.zip"
    print(f"Downloading UCI-HAR to {archive} ...")
    urllib.request.urlretrieve(UCI_HAR_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    archive.unlink()

    # The UCI archive wraps the dataset in a second zip file.
    inner = data_dir / f"{DATASET_DIRNAME}.zip"
    if inner.exists():
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(data_dir)
        inner.unlink()

    if not root.is_dir():
        raise FileNotFoundError(f"Expected '{root}' after extraction.")
    return root


def _read_split(root, split):
    """Read the raw inertial signals of one split as (N, 128, 9) plus integer labels in [0, 5]."""
    signals_dir = root / split / "Inertial Signals"
    labels = pd.read_csv(root / split / f"y_{split}.txt", sep=r"\s+", header=None).to_numpy().ravel() - 1
    x = np.empty((labels.shape[0], 128, len(SIGNALS)))
    for i, name in enumerate(SIGNALS):
        x[:, :, i] = pd.read_csv(signals_dir / f"{name}_{split}.txt", sep=r"\s+", header=None).to_numpy()
    return x, labels


def _add_acc_magnitude(x):
    """Append the magnitude of the total acceleration as a 10th channel."""
    magnitude = np.sqrt((x[:, :, 6:9] ** 2).sum(axis=-1, keepdims=True))
    return np.concatenate((x, magnitude), axis=-1)


def load_uci_har(data_dir="data", val_size=0.2, seed=DEFAULT_SEED, download=True):
    """Load UCI-HAR and return a dict with ``train``/``validation``/``test`` splits.

    Each split is ``{'data': (N, 128, 10) float array, 'label': (N, 6) one-hot array}``;
    the dict also holds the ``labels`` (activity names). The official subject-independent
    train/test split of the dataset is kept, and ``val_size`` of the training windows are
    held out for validation.
    """
    root = Path(data_dir) / DATASET_DIRNAME
    if not root.is_dir():
        if not download:
            raise FileNotFoundError(f"'{root}' not found. Run with download enabled or place the dataset there.")
        root = download_uci_har(data_dir)

    x_train, y_train = _read_split(root, "train")
    x_test, y_test = _read_split(root, "test")
    x_train, x_test = _add_acc_magnitude(x_train), _add_acc_magnitude(x_test)

    # Standardise every channel with statistics fitted on the training windows only.
    # Note: the scaler sees an (N, 128) matrix, i.e. each time step is standardised separately.
    for c in range(x_train.shape[-1]):
        scaler = StandardScaler()
        x_train[:, :, c] = scaler.fit_transform(x_train[:, :, c])
        x_test[:, :, c] = scaler.transform(x_test[:, :, c])

    n_classes = len(ACTIVITY_LABELS)
    y_train, y_test = to_categorical(y_train, n_classes), to_categorical(y_test, n_classes)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed, shuffle=True)

    return {
        "train": {"data": x_tr, "label": y_tr},
        "validation": {"data": x_val, "label": y_val},
        "test": {"data": x_test, "label": y_test},
        "labels": ACTIVITY_LABELS,
    }
