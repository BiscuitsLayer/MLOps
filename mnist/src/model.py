from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score


class MNISTModel(pl.LightningModule):
    def __init__(self, all_config):
        super().__init__()
        self.config = all_config

        # Save hyperparameters into checkpoint, so no need to reupload them
        # if we want to continue learning from some checkpoint
        self.save_hyperparameters()

        self.input_size = self.config.model.input_size
        self.num_classes = self.config.model.num_classes
        self.loss_function = F.cross_entropy
        self.linear = nn.Linear(self.input_size, self.num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, xb):
        xb = xb.reshape(-1, self.input_size)
        out = self.linear(xb)
        out = self.activation(out)
        return out

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        images, labels = batch
        out = self(images)
        loss = self.loss_function(out, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        images, labels = batch
        out = self(images)

        true_labels = labels.cpu()
        out_labels = out.cpu().argmax(1)
        loss = self.loss_function(out, labels)

        precision = precision_score(true_labels, out_labels, average="macro")
        recall = recall_score(true_labels, out_labels, average="macro")

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("precision", precision, on_step=True, on_epoch=True, prog_bar=False)
        self.log("recall", recall, on_step=True, on_epoch=True, prog_bar=False)

        return {"val_loss": loss, "precision": precision, "recall": recall}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.config.train.learning_rate
        )
        return {"optimizer": optimizer}
