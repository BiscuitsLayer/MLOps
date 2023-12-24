import os

import fire
import torch
import torchvision.transforms as transforms
from dvc.repo import Repo
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import mnist.src.mnist_model as mnist_model
import mnist.src.tools as tools

# Import parsed configs
from config.config_handler import all_config


class Infer:
    def __init__(self):
        self.config = all_config

        if not os.path.isdir(self.config.dataset.path) or not os.path.isdir(
            self.config.model.path
        ):
            # Pull repository data from Google Drive using DVC
            repo = Repo(".")
            repo.pull()

    def run_infer(self):
        """
        Run inference on some batch from MNIST dataset
        """
        mnist_dataset = MNIST(
            root=self.config.dataset.path,
            download=False,
            transform=transforms.ToTensor(),
        )
        _, validation_data = tools.split_data(mnist_dataset)
        val_loader = DataLoader(
            validation_data, self.config.dataset.batch_size, shuffle=True
        )

        model = mnist_model.MNISTModel(
            self.config.model.input_size, self.config.model.num_classes
        )
        model.load_state_dict(
            torch.load(
                f"{self.config.model.path}/{self.config.model.pretrained_model_file}"
            )
        )
        print(f"Loaded from {self.config.model.pretrained_model_file} successfully!")

        result = tools.evaluate(model, val_loader)
        print(result)


def main():
    # Use fire to call Infer class methods from CLI
    fire.Fire(Infer)


if __name__ == "__main__":
    main()
