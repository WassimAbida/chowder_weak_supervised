"""Experiment lightning module."""

import os
import fire
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from chowder_weak_supervised.lightning.chowder_module import ChowderModule
from chowder_weak_supervised.lightning.data_module import TilesDataModule
from chowder_weak_supervised.logger import logger

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info("Available device", device)
NUM_WORKERS = os.cpu_count()
logger.info("Number of workers:", NUM_WORKERS)

BATCH_SIZE = 4
N_EPOCHS = 30
EXPERIMENT_SEED = 42

ROOT_DIR = "data/saved_model/CHOWDER"
PRETRAINED_MODEL_WEIGHTS = f"{ROOT_DIR}/CHOWDER.ckpt"


# data/saved_model/CHOWDER/lightning_logs/version_0/checkpoints/epoch=6-step=483.ckpt
def training_expriment(
    n_extreme: int = 5,
    batch_size: int = BATCH_SIZE,
    n_epochs: int = N_EPOCHS,
    pretrained_model_weights: str = PRETRAINED_MODEL_WEIGHTS,
    root_dir: str = ROOT_DIR,
) -> None:
    pl.seed_everything(EXPERIMENT_SEED)
    # - Training

    # 1. Define DataLoaders
    data_module_pl = TilesDataModule(batch_size=batch_size, num_workers=NUM_WORKERS)

    # 2. Create a PyTorch Lightning trainer with the generation callback

    trainer = pl.Trainer(
        default_root_dir=root_dir,  # Where to save models
        accelerator=(
            "gpu" if str(device).startswith("cuda") else "cpu"
        ),  # We run on a GPU (if possible)
        devices=1,  # How many GPUs/CPUs we want to use (only 1 is available on my current instance)
        max_epochs=n_epochs,  # How many epochs to train for if no patience is set
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("step"),
        ],  # Log learning rate every step or epoch
        enable_progress_bar=True,
        # log_every_n_steps=5,
        gradient_clip_val=1.0,  # config_pl_trainer.get("gradient_clip_val", 0.0),  # Gradient clipping
        gradient_clip_algorithm="norm",
        # cf: https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html
    )

    trainer.logger._log_graph = (
        True  # If True, we plot the computation graph in tensorboard
    )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    # 3. Train
    # - Module chowder
    logger.info(f"Top & Bottom instances - Model Parameter 'n_extreme': {n_extreme}")
    module_pl = ChowderModule(n_extreme=n_extreme)

    # -Check whether pretrained model exists. If yes, load it and skip training
    if os.path.isfile(pretrained_model_weights):
        print(f"Found pretrained model at {pretrained_model_weights}, loading...")
        module_pl.load_state_dict(
            torch.load(pretrained_model_weights)["state_dict"]
        )  # Automatically loads the model with the saved hyperparameters
    else:
        logger.info("Start training.")
        # - seed
        trainer.fit(
            module_pl,
            data_module_pl.train_dataloader(),
            data_module_pl.validation_dataloader(),
        )
        logger.info("End training.")

    # 4. Test best model on validation and test set
    logger.info("Evaluating Model")
    val_result = trainer.test(
        module_pl, data_module_pl.validation_dataloader(), verbose=False
    )
    train_result = trainer.test(
        module_pl, data_module_pl.train_dataloader(), verbose=False
    )
    result = {
        "train_accuracy": train_result[0]["test_acc"],
        "val_accuracy": val_result[0]["test_acc"],
    }
    logger.info("Evaluation results")
    print(result)


if __name__ == "__main__":
    fire.Fire(training_expriment)
