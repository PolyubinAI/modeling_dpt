import sys

sys.path.append(".")

from configs.config import Config

config = Config.from_yaml("./configs/config.yaml")


import os

os.environ["CUDA_VISIBLE_DEVICES"] = config.device

import sys

sys.path.append(".")

import pytorch_lightning as light
import optuna
from lightning_dataset import DFC
from clearml import Task
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from utils import load_object
from dpt_lightning import LightDPT
from callbacks import PredictAfterValidationCallback, PredictAfterTrainingCallback

# from pytorch_lightning.strategies import DDPStrategy


LOG_EVERY_N_STEPS = 1

SEED_VALUE = 42


def objective(trial: optuna.Trial):  # noqa: WPS210
    config = Config.from_yaml("./configs/config.yaml")
    config.warmup_epochs = trial.suggest_int("warmup_epochs", 1, 10)
    config.optimizer_kwargs["lr"] = trial.suggest_float("lr", 1e-7, 1e-3)
    config.optimizer_kwargs["weight_decay"] = trial.suggest_loguniform(
        "weight_decay", 1e-4, 1e-2
    )
    config.data_config.batch_size = trial.suggest_int("batch_size", 8, 16)
    config.data_config.accumulate_grad_batches = trial.suggest_int(
        "accumulate_grad_batches", 1, 5
    )
    datamodule = DFC(config)
    model = load_object("my_models.{}".format(config.model_name))()
    task = Task.init(
        project_name=config.project_name,
        task_name=f"{config.experiment_name}",
        auto_connect_frameworks=True,
    )
    task.connect(config.model_dump())
    logger = task.get_logger()

    experiment_save_path = os.path.join(config.experiment_path, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor="val_loss",
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f"epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}",
    )

    model = LightDPT(config)
    # ddp = DDPStrategy(process_group_backend="gloo")

    trainer = light.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=config.device,
        # strategy=ddp,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=5, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval="epoch"),
            PredictAfterTrainingCallback(logger=logger),
            PredictAfterValidationCallback(logger=logger),
        ],
        accumulate_grad_batches=config.data_config.accumulate_grad_batches,
    )
    trainer.fit(model=model, datamodule=datamodule)

    return trainer.callback_metrics["train_loss"].item()


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize", pruner=pruner
    )  # We're minimizing the loss
    study.optimize(objective, n_trials=100)
    print("Number of finished trials: {}".format(len(study.trials)))

    print(f"Best trial: {study.best_trial.params}")
