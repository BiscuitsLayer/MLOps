import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import src.mnist_model as mnist_model
import src.tools as tools

LOAD_MODEL_FROM_FILE = False
BATCH_SIZE = 128
EPOCHS_COUNT = 5
LEARNING_RATE = 1e-3


def train():
    mnist_dataset = MNIST(
        root="dataset/", download=True, transform=transforms.ToTensor()
    )
    train_data, validation_data = tools.split_data(mnist_dataset)
    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation_data, BATCH_SIZE, shuffle=False)

    model = mnist_model.MNISTModel()
    if LOAD_MODEL_FROM_FILE:
        model.load_state_dict(torch.load("saved_model/mnist.pth"))
        print("Model loaded from file successfully!")

    _ = tools.fit(EPOCHS_COUNT, LEARNING_RATE, model, train_loader, val_loader)

    torch.save(model.state_dict(), "saved_model/mnist.pth")


if __name__ == "__main__":
    train()
