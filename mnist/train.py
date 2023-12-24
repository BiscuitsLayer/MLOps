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


class Train:
    def __init__(self):
        self.config = all_config

        if not os.path.isdir(self.config.dataset.path) or not os.path.isdir(
            self.config.model.path
        ):
            # Pull repository data from Google Drive using DVC
            repo = Repo(".")
            repo.pull()

    def run_train(self):
        """
        Run model train on MNIST dataset
        """
        mnist_dataset = MNIST(
            root=self.config.dataset.path,
            download=False,
            transform=transforms.ToTensor(),
        )
        train_data, validation_data = tools.split_data(mnist_dataset)
        train_loader = DataLoader(
            train_data, self.config.dataset.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            validation_data, self.config.dataset.batch_size, shuffle=True
        )

        model = mnist_model.MNISTModel(
            self.config.model.input_size, self.config.model.num_classes
        )
        if self.config.train.load_from_file:
            model.load_state_dict(
                torch.load(
                    f"{self.config.model.path}/{self.config.model.pretrained_model_file}"
                )
            )
            print(
                f"Loaded from {self.config.model.pretrained_model_file} successfully!"
            )

        _ = tools.fit(
            self.config.train.epoch_count,
            self.config.train.learning_rate,
            model,
            train_loader,
            val_loader,
        )

        torch.save(
            model.state_dict(),
            f"{self.config.model.path}/{self.config.model.new_model_file}",
        )
        print(f"Model saved to {self.config.model.new_model_file} successfully!")


def main():
    # Use fire to call Train class methods from CLI
    fire.Fire(Train)


if __name__ == "__main__":
    main()
