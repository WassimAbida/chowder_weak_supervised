import unittest
import pytest
from chowder_weak_supervised.lightning.experiment import training_expriment

CHECKPOINT_PATH = "data/saved_model/CHOWDER/lightning_logs/version_0/checkpoints/epoch=6-step=483.ckpt"
ROOT_DIR = "data/saved_model/CHOWDER"


# @pytest.mark.skip(reason="Skipped since time consuming but is ok.")
class TestChowderExperiment(unittest.TestCase):
    def test_chowder_experiment_mock(self):

        config_experiment = {
            "pretrained_model_weights": CHECKPOINT_PATH,
            "root_dir": ROOT_DIR,
            "n_extreme": 2,
            "batch_size": 4,
            "n_epochs": 2,
        }

        # Test with defaults parameters
        training_expriment(**config_experiment)
