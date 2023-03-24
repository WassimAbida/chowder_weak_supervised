"""Metrics module."""

import os
import fire
import pandas as pd

import torch
from torchmetrics.classification import BinaryROC
from sklearn.metrics import RocCurveDisplay

import matplotlib.pyplot as plt
from sklearn.metrics import auc

from chowder_weak_supervised.lightning.chowder_module import ChowderModule
from chowder_weak_supervised.lightning.data_module import TilesDataModule
from chowder_weak_supervised.logger import logger


R = 2
BATCH_SIZE = 4
NUM_WORKERS = os.cpu_count()
logger.info("Number of workers:", NUM_WORKERS)

MODEL_VERSION = "version_0"
CHECKPOINT_PATH = f"data/saved_model/CHOWDER/lightning_logs/{MODEL_VERSION}/checkpoints/epoch=6-step=483.ckpt"
logger.info(f"Model version {MODEL_VERSION} + Used checkpoint {CHECKPOINT_PATH}")


def get_validation_data_loader(
    num_workers: int = NUM_WORKERS, batch_size: int = BATCH_SIZE
):
    """Get validation dataloader for tiles Dataset.
    Args

        num_workers (int)
        batch_size (int)

    Returns
        Torch DataLoader
    """

    data_module_pl = TilesDataModule(batch_size=batch_size, num_workers=num_workers)
    return data_module_pl.validation_dataloader()


def setup_model(n_extreme: int = R, checkpoint_path: str = CHECKPOINT_PATH):
    """Setup Chowder Model & load its weight from a checkpoint.
    Args

        n_extreme (int)
        checkpoint_path (str)

    Returns
        Chowder Model
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Top & Bottom instances - Model Parameter 'n_extreme':", n_extreme)
    module_pl = ChowderModule(n_extreme=n_extreme)
    module_pl.load_state_dict(
        torch.load(checkpoint_path, map_location=device)["state_dict"]
    )
    return module_pl


def inference(
    n_extreme: int = R,
    path_to_predictions_file: str = "./data/test_predictions.csv",
    checkpoint_path: str = CHECKPOINT_PATH,
    num_workers: int = NUM_WORKERS,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Launch inference on Validation set & dump predictions results in csv format
    Compute AUC & ROC curves as well.

    Args
        n_extreme
        path_to_predictions_file
        checkpoint_path
        num_workers
        batch_size

    Returns
        None

    """

    # Load model
    module_pl = setup_model(n_extreme=n_extreme, checkpoint_path=checkpoint_path)

    # Get Dataloader
    val_dataloader = get_validation_data_loader(
        num_workers=num_workers, batch_size=batch_size
    )

    model_predictions = []
    all_targets = []
    tile_identifiers = []
    all_probas = []
    sigmoid = torch.nn.Sigmoid()

    # inference
    with torch.no_grad():
        for batch_content in val_dataloader:
            features = batch_content[0]
            targets = list(batch_content[1])
            # targets_x = torch.Tensor(targets)

            predictions = tuple([module_pl(elem) for elem in features])
            predictions_x = torch.cat(predictions, dim=0)
            probas = sigmoid(predictions_x)
            probas_bin = [1 if elem > 0.5 else 0 for elem in probas]

            all_probas.extend(probas_bin)
            model_predictions.extend(predictions_x)
            all_targets.extend(targets)
            tile_identifiers.extend(batch_content[5])

    # Get predicitons dataframe
    predictions_df = pd.DataFrame(
        {"Idx": tile_identifiers, "Prediction": all_probas, "Target": all_targets}
    )
    logger.info(f"Saving predictions Dataframe into {path_to_predictions_file}")
    predictions_df.to_csv(path_to_predictions_file)

    # Display ROC Curve
    metric = BinaryROC(thresholds=None)
    fpr, tpr, thresholds = metric(
        torch.tensor(model_predictions), torch.tensor(all_targets)
    )

    roc_auc = auc(fpr, tpr)
    logger.info(f"AUC Score: {roc_auc}")
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve - R={n_extreme} Model {MODEL_VERSION}")
    plt.legend()

    roc_curve_filename = f"./ROC_curve_{MODEL_VERSION}.png"
    logger.info(f"Saving ROC curve plot as png file.{roc_curve_filename}")
    plt.savefig(roc_curve_filename)


if __name__ == "__main__":
    fire.Fire(inference)
