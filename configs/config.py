from typing import List, Tuple

from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    name: str
    weight: float
    loss_fn: str
    scheduler_args: dict


class DataConfig(BaseModel):
    data_path: str
    data_path_photos_dict: dict[str, str]
    data_path_heights_dict: dict[str, str]
    crop_size: Tuple[int, int]
    dataset_names: List[str]
    train_count_samples: int
    test_count_samples: int
    batch_size: int
    n_workers: int
    train_size: float
    accumulate_grad_batches: int


class Config(BaseModel):
    project_name: str
    experiment_name: str
    experiment_path: str
    model_name: str
    data_config: DataConfig
    n_epochs: int
    accelerator: str
    device: str
    monitor_metric: str
    monitor_mode: str
    model_kwargs: dict
    optimizer: str
    warmup_epochs: int
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]
    metrics: List[str]

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
