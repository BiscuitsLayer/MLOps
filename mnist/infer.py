import os

import fire
import torch
import torchvision.transforms as transforms
from dvc.repo import Repo
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import mnist.src.mnist_model as mnist_model
import mnist.src.tools as tools

BATCH_SIZE = 128


class Infer:
    def __init__(self):
        if not os.path.isdir("dataset") or not os.path.isdir("model"):
            # Pull repository data from Google Drive using DVC
            repo = Repo(".")
            repo.pull()

    def run_infer(self):
        """
        Run inference on some batch from MNIST dataset
        """
        mnist_dataset = MNIST(
            root="dataset/", download=False, transform=transforms.ToTensor()
        )
        _, validation_data = tools.split_data(mnist_dataset)
        val_loader = DataLoader(validation_data, BATCH_SIZE, shuffle=True)

        model = mnist_model.MNISTModel()
        model.load_state_dict(torch.load("model/mnist_pretrained.pth"))
        print("Model loaded from file successfully!")

        result = tools.evaluate(model, val_loader)
        print(result)


def main():
    fire.Fire(Infer)


if __name__ == "__main__":
    main()
