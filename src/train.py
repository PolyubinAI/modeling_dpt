import sys
sys.path.append('.')
from configs.config import Config
config = Config.from_yaml('./configs/config.yaml')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = config.device

import pytorch_lightning as light
from lightning_dataset import DFC
from clearml import Task
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from utils import load_object
from dpt_lightning import LightDPT
from callbacks import PredictAfterValidationCallback, PredictAfterTrainingCallback


LOG_EVERY_N_STEPS = 1

SEED_VALUE = 42


def train(config: Config):  # noqa: WPS210
    datamodule = DFC(config)
    model = load_object('my_models.{}'.format(config.model_name))()
    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks={
        'pytorch': True, 'tensorboard': True, 'matplotlib': True, 'tensorflow': True,  
        'xgboost': True, 'scikit': True, 'fastai': True, 'lightgbm': True,
        'hydra': True, 'detect_repository': True, 'tfdefines': True, 'joblib': True,
        'megengine': True, 'catboost': True
        },
    )
    task.connect(config.model_dump())
    logger = task.get_logger()

    experiment_save_path = os.path.join(config.experiment_path, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=f"val_{config.data_config.dataset_names[0]}_MSE_loss",
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )

    model = LightDPT(config)

    trainer = light.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=True,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=f"val_{config.data_config.dataset_names[0]}_MSELoss", mode="min", patience=5),
            LearningRateMonitor(logging_interval='epoch'),
            PredictAfterTrainingCallback(logger=logger),
            PredictAfterValidationCallback(logger=logger, config=config),
        ],
        accumulate_grad_batches=config.data_config.accumulate_grad_batches,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    config = Config.from_yaml('./configs/config.yaml')
    train(config)
