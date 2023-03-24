"""Data lightning module."""

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from chowder_weak_supervised.dataset.tiles import TilesDataset


class TilesDataModule(LightningDataModule):
    """Lightning data module for CHOWDER

    Args:

            batch_size (int): batch size. Defaults to 2.
            num_workers (int): num of worker used to produce samples. Defaults to 8.
            train_features_path (str): Path to npy arrays containing training features for a slide
            test_features_path (str):Path to npy arrays containing test features for a slide
            train_labels (str): Path to csv file containing training slides labels
            test_labels (str):  Path to csv file containing test slides labels

    """

    def __init__(
        self,
        batch_size: int = 2,
        num_workers: int = 8,
        train_features_path: str = "data/train_input/resnet_features/",
        test_features_path: str = "data/test_input/resnet_features/",
        train_labels: str = "data/train_output.csv",
        test_labels: str = "data/test_output.csv",
    ) -> None:
        super().__init__()
        self.train_features_path = train_features_path
        self.test_features_path = test_features_path

        self.train_labels = train_labels
        self.test_labels = test_labels

        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def collate_fn_custom(batch):
        return tuple(zip(*batch))

    def train_dataloader(self) -> DataLoader:
        train_dataset = TilesDataset(
            labels_path=self.train_labels, embeddings_path=self.train_features_path
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_custom,
        )

    def validation_dataloader(self) -> DataLoader:
        validation_dataset = TilesDataset(
            labels_path=self.test_labels, embeddings_path=self.test_features_path
        )

        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_custom,
        )
