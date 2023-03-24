"""Chowder lightning module."""

import os
import logging
import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy

from chowder_weak_supervised.model.chowder import CHOWDER
from chowder_weak_supervised.loss.chowder_loss import LossScore


class ChowderModule(pl.LightningModule):
    """Init Chowder pytorch Lightening Module
        This module is used to simplify pytorch training code. This object embarks:
            * Chowder model
            * Chowder loss
            * Chowder optimizers

    Args
        learning_rate: (float) Model learning rate. Defaults to 0.0001
        score_threshold: (float) threshold to predict the presence of metastases according to score. Defaults to 0.5.
        weight_decay: (float) L2 penalty defaults to 0.0001

        experiment_path: (str) Path to experiment path for weights storage
        weights_filename: (str) Chowder weights filename
        n_extreme: (int) Number of entries retained in top Instances and negative evidence

    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        score_threshold: float = 0.5,
        weight_decay: float = 0.0001,
        experiment_path: str = "data/experiment/tmp",
        weights_filename: str = "chowder_weights.pth",
        n_extreme: int = 5,
    ) -> None:
        super().__init__()

        # ! Do not use `self.device` because self.device='cpu'
        # see https://github.com/Lightning-AI/lightning/issues/13108
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weights_filename = weights_filename
        self.n_extreme = n_extreme

        self.model = CHOWDER(n_extreme=self.n_extreme)
        self.model.to(torch.double)
        self.loss_module = LossScore()
        self.metric = BinaryAccuracy(
            threshold=score_threshold  # set score<threshold = 0, and score>=threshold = 1, and compute binary accuracy
        )

        self.experiment_path = experiment_path
        os.makedirs(self.experiment_path, exist_ok=True)

    @torch.no_grad()
    def accuracy_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute accuracy for a batch of size N

        Args:
            predictions (torch.Tensor): shape [N, 1] (float numbers between 0 and 1)
            labels (torch.Tensor): shape [N, 1] (0 or 1)

        Returns:
            float: accuracy

        """
        return self.metric(
            predictions, targets
        ).float()  # Metric used is 'binary accuracy'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method is the one used for CHOWDER model."""
        return self.model(x)

    def configure_optimizers(self):
        """Configure optimizer."""
        # Set learning rate
        # - CHOWDER params
        model_params = [
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        l_parameters = [dict(params=model_params, lr=self.learning_rate)]

        # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
        optimizer = torch.optim.AdamW(l_parameters, weight_decay=self.weight_decay)
        # Optional - Add schduler for learning rate adaptation
        optimizer_milestone: tuple[int, int] = (10, 20)
        optimizer_gamma: float = 0.1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=optimizer_milestone, gamma=optimizer_gamma
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Get data ("batch" is the output of the training data loader).
        # Adapt Batch
        features = batch[0]
        targets = list(batch[1])

        # Forward
        predictions = tuple([self.forward(elem) for elem in features])
        # predictions_x = torch.Tensor(predictions).to(device)
        predictions_x = torch.cat(predictions, dim=0).to(device)

        # Compute loss & accuracy
        targets_x = torch.Tensor(targets).to(device)
        loss = self.loss_module(predictions_x, targets_x)

        # Since predictions are logits & out of [0,1] range,
        # an auto application of Sigmoid per element will be performed
        # we check https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
        acc = self.accuracy_score(predictions_x, targets_x)

        # Logs tensorboard
        self.log(
            "train_loss",
            loss,
            on_epoch=True,  #  batch_size=len(features)
        )  # logger=True, on_step=True,
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)  # on_step=True,
        return loss

    def validation_step(self, batch, batch_idx) -> None:

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Adapt Batch
        features = batch[0]
        targets = list(batch[1])

        # Forward
        predictions = tuple([self.forward(elem) for elem in features])
        predictions_x = torch.cat(predictions, dim=0).to(device)
        # predictions_x = torch.Tensor(predictions).to(device)

        # Compute validation loss & accuracy
        targets_x = torch.Tensor(targets).to(device)
        loss = self.loss_module(predictions_x, targets_x)
        acc = self.accuracy_score(predictions_x, targets_x)

        # Log results
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)  # on_step=True,
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)  # on_step=True,

    def test_step(self, batch, batch_idx) -> None:

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Adapt Batch
        targets = list(batch[1])
        features = batch[0]

        targets_x = torch.Tensor(targets).to(device)
        predictions = tuple([self.forward(elem) for elem in features])
        predictions_x = torch.cat(predictions, dim=0).to(device)
        # predictions_x = torch.Tensor(predictions).to(device)
        acc = self.accuracy_score(predictions_x, targets_x)
        # Logs
        self.log("test_acc", acc)

    def save_chowder_model(self, model_path: str) -> None:
        """Function to save chowder model.
        Args
            model_path: (str) Path to model storage.

        Returns
            None

        """
        logging.info(f" Save chowder model: `{model_path}` ...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def on_save_checkpoint(self, checkpoint) -> None:
        # Store model in diffrent location than the one automatically generated by lightning
        # See https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-save-checkpoint

        # Save checkpoint using lightning module
        super().on_save_checkpoint(checkpoint)
        # Save Chowder model
        file_name = os.path.basename(self.weights_filename)
        file_path = os.path.join(self.experiment_path, file_name)
        self.save_chowder_model(file_path)
