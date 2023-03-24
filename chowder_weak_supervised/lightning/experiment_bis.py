import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from chowder_weak_supervised.lightning.chowder_module import ChowderModule
from chowder_weak_supervised.lightning.data_module import TilesDataModule
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_WORKERS = os.cpu_count()
print("Number of workers:", NUM_WORKERS)

BATCH_SIZE = 8
N_EPOCHS = 30
R = 2

# - Training
checkpoint_path = "data/saved_model"
model_name = "CHOWDER"

trainer = pl.Trainer(
    default_root_dir=os.path.join(checkpoint_path, model_name),  # Where to save models
    accelerator=(
        "gpu" if str(device).startswith("cuda") else "cpu"
    ),  # We run on a GPU (if possible)
    devices=1,  # How many GPUs/CPUs we want to use (1 is enough for the notebooks)
    max_epochs=N_EPOCHS,  # How many epochs to train for if no patience is set
    callbacks=[
        ModelCheckpoint(
            save_weights_only=True, mode="max", monitor="val_acc"
        ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
        LearningRateMonitor("step"),
    ],  # Log learning rate every epoch
    enable_progress_bar=True,
    log_every_n_steps=5,
    gradient_clip_val=1.0,  # config_pl_trainer.get("gradient_clip_val", 0.0),  # Gradient clipping
    gradient_clip_algorithm="norm",  # config_pl_trainer.get(
    # "gradient_clip_algo", "norm" ),
    # cf: https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html
)  # Set to False if you do not want a progress bar

trainer.logger._log_graph = (
    True  # If True, we plot the computation graph in tensorboard
)
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

# 3. Train
# - Module chowder

# -Check whether pretrained model exists. If yes, load it and skip training
os.makedirs(checkpoint_path, exist_ok=True)
pretrained_filename = os.path.join(checkpoint_path, model_name + ".ckpt")
print(pretrained_filename)

pl.seed_everything(42)
model = ChowderModule(n_extreme=R)
data_module = TilesDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

print("\n Accelerator:", trainer.accelerator)

trainer.fit(model, data_module.train_dataloader(), data_module.validation_dataloader())
print("End training.")


print("Evaluation")
module_pl = ChowderModule(n_extreme=R)
# model2 = module_pl.load_from_checkpoint(
#           trainer.checkpoint_callback.best_model_path
#       )  # Load best checkpoint after training
module_pl.load_state_dict(
    torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]
)

# 4. Test best model on validation and test set
val_result = trainer.test(module_pl, data_module.validation_dataloader(), verbose=False)
train_result = trainer.test(module_pl, data_module.train_dataloader(), verbose=False)
result = {
    "train_accuracy": train_result[0]["test_acc"],
    "val_accuracy": val_result[0]["test_acc"],
}

print(result)

# {'train': 0.9589903950691223, 'val': 0.849571168422699}
# BATCH_SIZE = 4
# N_EPOCHS = 30
# data/saved_model/CHOWDER/lightning_logs/version_0/checkpoints/epoch=26-step=1863.ckpt


# verison 10 to delete
# BATCH_SIZE = 8
# N_EPOCHS = 30
# R = 2
# {'train_accuracy': 0.884405791759491, 'val_accuracy': 0.84454345703125}
