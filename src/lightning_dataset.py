from typing import Optional, List
import sys

sys.path.append("..")

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from preprocess import get_preprocess
from configs.config import Config
from utils import load_object


class DFC(LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self._data_config = config.data_config
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            train_sets = []
            valid_sets = []
            for dataset_name in self._data_config.dataset_names:
                transform_func = get_preprocess(dataset_name)
                dataset = load_object(f"dataset.{dataset_name}")
                train_sets.append(dataset(dataset_mode="train", transform_func=transform_func))
                valid_sets.append(dataset(dataset_mode="valid", transform_func=transform_func))
            
            self.train_dataset = ConcatDataset(train_sets)
            self.valid_dataset = valid_sets

        elif stage == 'test':
            test_sets = []
            for dataset_name in self._data_config.dataset_names:
                transform_func = get_preprocess(dataset_name)
                dataset = load_object(f"dataset.{dataset_name}")
                test_sets.append(dataset(dataset_mode="test", transform_func=transform_func))
            
            self.test_dataset = test_sets

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._data_config.batch_size,
            num_workers=self._data_config.n_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        
        return [DataLoader(
            dataset=dataset,
            batch_size=self._data_config.batch_size,
            num_workers=self._data_config.n_workers,
            shuffle=False,
            pin_memory=True
        ) for dataset in self.valid_dataset]
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._data_config.batch_size,
            num_workers=self._data_config.n_workers,
            shuffle=False,
            pin_memory=True
        )
