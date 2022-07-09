from typing import Dict, Tuple

from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import medmnist
from medmnist import INFO


def get_dataset_CIFAR10(path_to_data: Path, cid: str, partition: str):

    # generate path to cid's data
    path_to_data = path_to_data / cid / (partition + ".pt")

    return TorchVision_FL(path_to_data, transform=cifar10Transformation())


def get_dataset(download: bool = True, split: str = "train"):
    def get_pathmnist():
        data_flag = "pathmnist"
        info = INFO[data_flag]

        DataClass = getattr(medmnist, info["python_class"])

        return DataClass, info

    # preprocessing
    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    DataClass, info = get_pathmnist()

    # load the data
    dataset = DataClass(split=split, transform=data_transform, download=download)

    return dataset, info


def get_dataloader_cifar10(
    path_to_data: str, cid: str, is_train: bool, batch_size: int, workers: int
):
    """Generates trainset/valset object and returns appropiate dataloader."""

    partition = "train" if is_train else "val"
    dataset = get_dataset(Path(path_to_data), cid, partition)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


# TODO might need to take CID into account for split?
def get_dataloader(
    is_train: bool, batch_size: int, workers: int, shuffle: bool
) -> Tuple[DataLoader, Dict]:
    """Generates trainset/valset object and returns appropiate dataloader."""

    split = "train" if is_train else "test"
    dataset, info = get_dataset(download=True, split=split)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs), info


def getCIFAR10(path_to_data="./data"):
    """Downloads CIFAR10 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = datasets.CIFAR10(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data) / "cifar-10-batches-py"
    training_data = data_loc / "training.pt"
    print("Generating unified CIFAR dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = datasets.CIFAR10(
        root=path_to_data, train=False, transform=cifar10Transformation()
    )

    # returns path where training data is and testset
    return training_data, test_set
