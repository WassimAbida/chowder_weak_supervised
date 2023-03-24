import os
import unittest
import torch
from pathlib import Path
from chowder_weak_supervised.dataset.tiles import TilesDataset
from chowder_weak_supervised.model.chowder import CHOWDER
from torch.utils.data import DataLoader


DATASET_DATA_PATH: Path = (
    Path(os.path.realpath(__file__)).parents[1] / "test_dataset/assets/"
)

# Model Params
DROPOUT_RATE = 0.5
MLP_LAYER_1_SIZE = 200
MLP_LAYER_2_SIZE = 100
R_value = 3


def collate_fn_custom(batch):
    return tuple(zip(*batch))


class TestChowder(unittest.TestCase):
    def setUp(self):

        # Dataset parameters
        self.dataset_parameters = {
            "batch_size": 4,
            "shuffle": True,
        }

        # Compute one model sample
        dataset = TilesDataset(
            labels_path=os.path.join(DATASET_DATA_PATH, "sample.csv"),
            embeddings_path=os.path.join(DATASET_DATA_PATH, "features"),
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self.dataset_parameters["batch_size"],
            shuffle=self.dataset_parameters["shuffle"],
            collate_fn=collate_fn_custom,
        )
        self.batch = next(iter(data_loader))

    def test_forward(self):

        # Get data from batch
        features, labels, zoom_level, tile_x_coords, tile_y_coords, WSI_ID = self.batch

        # Model config
        # - Config of the chowder model
        config_chowder = {
            "dropout_rate": DROPOUT_RATE,
            "mlp_layer_1_size": MLP_LAYER_1_SIZE,
            "mlp_layer_2_size": MLP_LAYER_2_SIZE,
            "n_extreme": R_value,
        }

        # Test forward
        model = CHOWDER(**config_chowder)
        model.to(torch.double)

        predictions = model(features[0])
        # - Test output size
        self.assertEqual(predictions.shape, (1,))
        # - Test chowder parameter
        self.assertEqual(model._n_extreme, R_value)
