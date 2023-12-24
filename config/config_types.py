from dataclasses import dataclass


@dataclass
class Model:
    path: str
    input_size: int
    num_classes: int


@dataclass
class Dataset:
    path: str
    batch_size: int


@dataclass
class Train:
    load_from_file: bool
    epoch_count: int
    learning_rate: float


@dataclass
class AllConfig:
    model: Model
    dataset: Dataset
    train: Train
