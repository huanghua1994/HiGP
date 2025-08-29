"""UCI Dataset loading and caching utilities."""

import os
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod


class UCIDataset(ABC):
    """Abstract base class for UCI datasets."""

    def __init__(self, cache_dir: str = "~/.higp_cache/datasets"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self) -> None:
        """Download dataset if not cached."""
        pass

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and return the full dataset."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass

    @property
    @abstractmethod
    def url(self) -> str:
        """Download URL."""
        pass


class BikeDataset(UCIDataset):
    """Bike Sharing dataset from UCI.

    Features: 15
    Samples: ~17,000 hourly records
    """

    name = "bike"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"

    def download(self) -> None:
        """Download and extract Bike dataset."""
        dataset_path = self.cache_dir / "bike"
        dataset_path.mkdir(exist_ok=True)

        hour_file = dataset_path / "hour.csv"
        if hour_file.exists():
            return

        print(f"Downloading Bike dataset from {self.url}...")
        zip_path = dataset_path / "bike.zip"

        response = requests.get(self.url)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path)

        zip_path.unlink()
        print("Bike dataset downloaded successfully.")

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Bike dataset.

        Returns:
            X: Features array (N, 15)
            y: Target array (N,)
        """
        self.download()

        dataset_path = self.cache_dir / "bike"
        hour_file = dataset_path / "hour.csv"

        df = pd.read_csv(hour_file)

        df["dteday"] = pd.to_datetime(df["dteday"]).dt.dayofyear
        df.drop(columns=["instant"], inplace=True)

        data = df.values
        X = data[:, :-1]
        y = data[:, -1]

        return X, y


class Road3DDataset(UCIDataset):
    """3D Road Network dataset from UCI.

    Features: 2
    Samples: ~434,874 GPS points
    """

    name = "road3d"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt"

    def download(self) -> None:
        """Download Road3D dataset."""
        dataset_path = self.cache_dir / "road3d"
        dataset_path.mkdir(exist_ok=True)

        data_file = dataset_path / "3D_spatial_network.txt"
        if data_file.exists():
            return

        print(f"Downloading Road3D dataset from {self.url}...")

        response = requests.get(self.url)
        response.raise_for_status()

        with open(data_file, "wb") as f:
            f.write(response.content)

        print("Road3D dataset downloaded successfully.")

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load Road3D dataset.

        Returns:
            X: Features array (N, 2) - longitude and latitude
            y: Target array (N,) - elevation
        """
        self.download()

        dataset_path = self.cache_dir / "road3d"
        data_file = dataset_path / "3D_spatial_network.txt"

        df = pd.read_csv(data_file, sep=",", header=None)
        data_array = df.values[:, 1:]

        X = data_array[:, 1:3]
        y = data_array[:, -1]

        return X, y


def load_uci_dataset(
    dataset_name: str,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    cache_dir: str = "~/.higp_cache/datasets",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and split a UCI dataset.

    Args:
        dataset_name: Name of dataset ('bike' or 'road3d')
        n_train: Number of training samples (None for default split)
        n_test: Number of test samples (None for default split)
        train_ratio: Train/test split ratio if n_train/n_test not specified
        seed: Random seed for reproducible splits
        cache_dir: Directory for caching downloaded datasets

    Returns:
        train_x: Training features (n_train, n_features)
        train_y: Training labels (n_train,)
        test_x: Test features (n_test, n_features)
        test_y: Test labels (n_test,)
    """
    if dataset_name.lower() == "bike":
        dataset = BikeDataset(cache_dir)
    elif dataset_name.lower() == "road3d":
        dataset = Road3DDataset(cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X, y = dataset.load()
    n_total = len(X)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    X = X[indices]
    y = y[indices]

    if n_train is None and n_test is None:
        n_train = int(n_total * train_ratio)
        n_test = n_total - n_train
    elif n_train is None:
        n_train = n_total - n_test
    elif n_test is None:
        n_test = n_total - n_train

    n_train = min(n_train, n_total)
    n_test = min(n_test, n_total - n_train)

    train_x = X[:n_train]
    train_y = y[:n_train]
    test_x = X[n_train : n_train + n_test]
    test_y = y[n_train : n_train + n_test]

    return train_x, train_y, test_x, test_y
