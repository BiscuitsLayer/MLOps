import os

import fire
import lightning.pytorch as pl
import torch
from dvc.repo import Repo

# Import parsed configs
from config.config_handler import all_config
from mnist.src.data import MNISTDatamodule
from mnist.src.model import MNISTModel


class Infer:
    def __init__(self):
        torch.set_float32_matmul_precision("medium")
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
        datamodule = MNISTDatamodule(self.config)
        model = MNISTModel(self.config)

        model.load_state_dict(
            torch.load(
                f"{self.config.model.path}/{self.config.model.pretrained_model_file}"
            )
        )
        print(f"Loaded from {self.config.model.pretrained_model_file} successfully!")

        trainer = pl.Trainer(max_epochs=self.config.train.epoch_count)
        trainer.validate(model=model, datamodule=datamodule)


def main():
    # Use fire to call Infer class methods from CLI
    fire.Fire(Infer)


if __name__ == "__main__":
    main()
