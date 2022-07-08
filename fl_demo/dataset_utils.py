from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil
from PIL import Image
from torchvision.datasets import VisionDataset
from typing import Callable, Dict, Optional, Tuple, Any
from flwr.dataset.utils.common import create_lda_partitions

import medmnist
from medmnist import INFO


def get_dataset_CIFAR10(path_to_data: Path, cid: str, partition: str):

    # generate path to cid's data
    path_to_data = path_to_data / cid / (partition + ".pt")

    return TorchVision_FL(path_to_data, transform=cifar10Transformation())


def get_dataset(download: bool = True, split: str = "train"):

  def get_pathmnist():
    data_flag = 'pathmnist'
    info = INFO[data_flag]

    DataClass = getattr(medmnist, info['python_class'])

    return DataClass, info

  # preprocessing
  data_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[.5], std=[.5])
  ])

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
    is_train: bool,
    batch_size: int,
    workers: int,
    shuffle: bool
) -> Tuple[DataLoader, Dict]:
    """Generates trainset/valset object and returns appropiate dataloader."""

    split = "train" if is_train else "test"
    dataset, info = get_dataset(download=True, split=split)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs), info


def get_random_id_splits(total: int, val_ratio: float, shuffle: bool = True):
    """splits a list of length `total` into two following a
    (1-val_ratio):val_ratio partitioning.

    By default the indices are shuffled before creating the split and
    returning.
    """

    if isinstance(total, int):
        indices = list(range(total))
    else:
        indices = total

    split = int(np.floor(val_ratio * len(indices)))
    # print(f"Users left out for validation (ratio={val_ratio}) = {split} ")
    if shuffle:
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def do_fl_partitioning(path_to_dataset, pool_size, alpha, num_classes, val_ratio=0.0):
    """Torchvision (e.g. CIFAR-10) datasets using LDA.""" ## FIXME

    # images, labels = torch.load(path_to_dataset)
    images = path_to_dataset.imgs
    labels = path_to_dataset.labels

    idx = np.array(range(len(images)))
    dataset = [idx, labels]
    partitions, _ = create_lda_partitions(
        dataset, num_partitions=pool_size, concentration=alpha, accept_imbalanced=True
    )

    # Show label distribution for first partition (purely informative)
    partition_zero = partitions[0][1]
    hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
    print(
        f"Class histogram for 0-th partition (alpha={alpha}, {num_classes} classes): {hist}"
    )

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    # splits_dir = path_to_dataset.parent / "federated"
    splits_dir = Path(path_to_dataset.root) / "federated"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    Path.mkdir(splits_dir, parents=True)

    for p in range(pool_size):

        labels = partitions[p][1]
        image_idx = partitions[p][0]
        imgs = images[image_idx]

        # create dir
        Path.mkdir(splits_dir / str(p))

        if val_ratio > 0.0:
            # split data according to val_ratio
            train_idx, val_idx = get_random_id_splits(len(labels), val_ratio)
            val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            with open(splits_dir / str(p) / "val.pt", "wb") as f:
                torch.save([val_imgs, val_labels], f)

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / str(p) / "train.pt", "wb") as f:
            torch.save([imgs, labels], f)

    return splits_dir


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
