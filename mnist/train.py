import os

import fire
import torch
import torchvision.transforms as transforms
from dvc.repo import Repo
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import mnist.src.mnist_model as mnist_model
import mnist.src.tools as tools

LOAD_MODEL_FROM_FILE = False
BATCH_SIZE = 128
EPOCHS_COUNT = 5
LEARNING_RATE = 1e-3


class Train:
    def __init__(self):
        if not os.path.isdir("dataset") or not os.path.isdir("model"):
            # Pull repository data from Google Drive using DVC
            repo = Repo(".")
            repo.pull()

    def run_train(self):
        """
        Run model train on MNIST dataset
        """
        mnist_dataset = MNIST(
            root="dataset/", download=False, transform=transforms.ToTensor()
        )
        train_data, validation_data = tools.split_data(mnist_dataset)
        train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(validation_data, BATCH_SIZE, shuffle=True)

        model = mnist_model.MNISTModel()
        if LOAD_MODEL_FROM_FILE:
            model.load_state_dict(torch.load("model/mnist_pretrained.pth"))
            print("Model loaded from file successfully!")

        _ = tools.fit(EPOCHS_COUNT, LEARNING_RATE, model, train_loader, val_loader)

        torch.save(model.state_dict(), "model/mnist_new.pth")


def main():
    fire.Fire(Train)


if __name__ == "__main__":
    main()
