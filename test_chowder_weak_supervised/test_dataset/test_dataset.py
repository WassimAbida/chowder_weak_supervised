import unittest
from chowder_weak_supervised.dataset.tiles import TilesDataset
from pathlib import Path
import os

DATASET_DATA_PATH: Path = (
    Path(os.path.realpath(__file__)).parents[1] / "test_dataset/assets/"
)


BATCH_CONTENT_LENGTH = 6
SAMPLE_DATA_IDS = [210, 222, 278]


class TestDatasetTiles(unittest.TestCase):
    def setUp(self):
        self.dataset = TilesDataset(
            labels_path=os.path.join(DATASET_DATA_PATH, "sample.csv"),
            embeddings_path=os.path.join(DATASET_DATA_PATH, "features"),
        )

        self.nb_samples = 3

    def test_init(self):

        # Test len of dataset
        self.assertEqual(len(self.dataset), self.nb_samples)

    def test_get_item(self):

        # Take the first element of the dataset
        self.assertEqual([*self.dataset.embeddings], SAMPLE_DATA_IDS)

        batch_content = next(iter(self.dataset))
        self.assertEqual(len(batch_content), BATCH_CONTENT_LENGTH)

        (
            features,
            labels,
            zoom_level,
            tile_x_coords,
            tile_y_coords,
            WSI_ID,
        ) = batch_content
        self.assertIn(WSI_ID, SAMPLE_DATA_IDS)
