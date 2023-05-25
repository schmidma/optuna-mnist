#!/bin/env python
import os
from datetime import timedelta

# there is a bug somewhere, this fixes things for progress bars
from tqdm import tqdm as _

import lightning
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
#from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning_pruning_callback import PyTorchLightningPruningCallback

from mnist_data import MnistDataModule
from mnist_model import MnistModel


def compose_hyperparameters(trial: optuna.trial.Trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    return {
        "learning_rate": learning_rate,
    }


def objective(trial: optuna.trial.Trial):
    study_name = trial.study.study_name
    hyperparameters = compose_hyperparameters(trial)

    data = MnistDataModule(
        data_dir="data",
        num_workers=32,
    )
    model = MnistModel(hyperparameters)

    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy_epoch",
        min_delta=0.01,
        patience=3,
        mode="max",
    )
    pruning_callback = PyTorchLightningPruningCallback(
        trial, monitor="val_accuracy_epoch"
    )
    checkpoint_callback = ModelCheckpoint(
        f"checkpoints/{study_name}/trial_{trial.number}",
        monitor="val_accuracy_epoch",
        mode="max",
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir="logs",
        name=study_name,
        version=f"trial_{trial.number}",
        default_hp_metric=False,
    )
    trainer = lightning.Trainer(
        max_epochs=15,
        callbacks=[
            early_stopping_callback,
            checkpoint_callback,
            pruning_callback,
        ],
        logger=tensorboard_logger,
        enable_model_summary=False,
        accelerator="cpu",
    )
    initial_performance = trainer.validate(model, data, verbose=False)[0][
        "val_accuracy_epoch"
    ]
    trainer.logger.log_hyperparams(
        hyperparameters, metrics={"val_accuracy_epoch": initial_performance}
    )
    trainer.fit(model, data)
    accuracy = trainer.callback_metrics["val_accuracy_epoch"].item()
    return accuracy


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        study_name="mnist",
        direction="maximize",
        storage=os.environ["OPTUNA_STORAGE"] or None,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    timeout = timedelta(hours=4)
    study.optimize(objective, timeout=timeout.total_seconds())
