from typing import Optional

import lightning.pytorch as pl
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class MNISTDatamodule(pl.LightningDataModule):
    def __init__(self, all_config):
        super().__init__()
        self.config = all_config

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.dataset = MNIST(
            root=self.config.dataset.path,
            download=False,
            transform=transforms.ToTensor(),
        )
        self.train_data, self.validation_data = train_test_split(
            self.dataset, test_size=self.config.dataset.validation_size
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.train_data, self.config.dataset.batch_size, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        # Validation data shouldn't be shuffled
        return DataLoader(
            self.validation_data, self.config.dataset.batch_size, shuffle=False
        )
