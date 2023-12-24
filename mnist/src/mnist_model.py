import torch
import torch.nn as nn
import torch.nn.functional as F

import mnist.src.tools as tools


class MNISTModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.loss_function = F.cross_entropy
        self.linear = nn.Linear(self.input_size, self.num_classes)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, xb):
        xb = xb.reshape(-1, self.input_size)
        out = self.linear(xb)
        out = self.activation(out)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = self.loss_function(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = self.loss_function(out, labels)
        acc = tools.accuracy(out, labels)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["val_loss"], result["val_acc"]
            )
        )
