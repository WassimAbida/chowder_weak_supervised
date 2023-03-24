"""Tiles module."""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class TilesDataset(Dataset):
    """Dataset constructor of tiles extracted from whole-slide-imaging (WSI)

    Args
        labels_path: (str) Path to csv file containing slides labels
            ID, Target
            ID_042, 1
            ID_303, 0...

        embeddings_path: (str) Path to npy arrays containing features for a slide
            resnet_features/<slide-ID>.npy
              array of size (n_tiles, 2051) where :
               1st column defines the zoom level of a tile
               2nd and 3rd column define the coordinates of a tile in the slide
               Remaining 2048 columns define the pre-computed ResNet features

    """

    def __init__(self, labels_path: str, embeddings_path: str) -> None:
        super().__init__()
        self.labels_path = labels_path
        self.embeddings_path = embeddings_path

        self.df_targets = self.load_labels(path_to_labels=self.labels_path)
        self.embeddings = self.load_pre_computed_features(
            path_to_embeddings=self.embeddings_path
        )

    def __len__(self) -> int:
        """Get lenght of dataset content, aka number of slides in dataset."""
        return len(self.df_targets)

    def __getitem__(
        self, idx: int
    ) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, int]:
        """Get samples (X, Y)

        Args:
            idx (int): sample ids
        """

        labels = int(self.df_targets["Target"][idx])
        WSI_ID = self.df_targets["Numeric_ID"][idx]

        embeddings_sample = self.embeddings[
            WSI_ID
        ]  # np.transpose(self.embeddings[WSI_ID],(1, 0))

        zoom_level = torch.as_tensor(embeddings_sample[:, 0])
        tile_x_coords = torch.as_tensor(embeddings_sample[:, 1])
        tile_y_coords = torch.as_tensor(embeddings_sample[:, 2])
        features = torch.as_tensor(embeddings_sample[:, 3:])

        return (features, labels, zoom_level, tile_x_coords, tile_y_coords, WSI_ID)

    @staticmethod
    def load_labels(path_to_labels: str) -> pd.DataFrame:
        """Load labels data
        Args
            path_to_labels: (str) Path to csv file containing slides labels

        Returns
            pd.DataFrame

        """
        logging.info(f"Loading targets from {path_to_labels}.")
        if os.path.isfile(path_to_labels):
            df_targets = pd.read_csv(path_to_labels)
        else:
            logging.error(f"Invalid path to labels: {path_to_labels}")
            return

        df_targets["Numeric_ID"] = df_targets["ID"].apply(
            lambda x: int(x.split("_")[1])
        )
        return df_targets

    @staticmethod
    def load_pre_computed_features(path_to_embeddings: str) -> dict[str, np.ndarray]:
        """Load pre-computed features
        Args
            path_to_embeddings: (str) Path to npy arrays containing features for a slide

        Returns
            dict[str, np.ndarray]

        """
        logging.info(f"Loading tiles features from {path_to_embeddings}.")
        list_embeddings = os.listdir(path_to_embeddings)
        embeddings_dict = {}
        for embeddings_id in list_embeddings:
            data_sample = np.load(os.path.join(path_to_embeddings, embeddings_id))

            tile_embeddings_id = int(Path(embeddings_id).stem.split("_")[1])
            embeddings_dict[tile_embeddings_id] = data_sample

        return embeddings_dict
