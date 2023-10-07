import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import src.mnist_model as mnist_model
import src.tools as tools

BATCH_SIZE = 128


def train():
    mnist_dataset = MNIST(
        root="dataset/", download=True, transform=transforms.ToTensor()
    )
    _, validation_data = tools.split_data(mnist_dataset)
    val_loader = DataLoader(validation_data, BATCH_SIZE, shuffle=False)

    model = mnist_model.MNISTModel()
    model.load_state_dict(torch.load("saved_model/mnist.pth"))
    print("Model loaded from file successfully!")

    result = tools.evaluate(model, val_loader)
    print(result)


if __name__ == "__main__":
    train()
