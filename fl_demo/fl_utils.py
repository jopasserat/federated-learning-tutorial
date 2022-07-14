import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from flwr.dataset.utils.common import create_lda_partitions

from fl_demo.dataset_utils import get_medmnist_data_info


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


def do_fl_partitioning(
    path_to_dataset, pool_size: int, alpha: float, num_classes: int, val_ratio=0.0
):
    """(non-)IID partitioning of Torchvision datasets using LDA."""

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
    # FIXME refactor not to assume a specific field name
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

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / str(p) / "pathmnist.npz", "wb") as f:
            d = {
                "train_images": imgs,
                "train_labels": labels,
                "val_images": val_imgs,
                "val_labels": val_labels,
            }

            np.savez(f, **d)

    return splits_dir


def get_federated_dataloader(
    base_path: Path,
    client_id: str,
    is_train: bool,
    batch_size: int,
    workers: int,
    shuffle: bool,
    transforms: Callable[[], Compose],
) -> DataLoader:
    """
    Generates trainset/valset object and returns appropiate dataloader.
    This is the federated version.
    Assumes dataset was already present on disk and partitioned using do_fl_partitioning
    See dataset_utils for the centralised equivalent
    """

    split = "train" if is_train else "val"

    path_to_data = base_path / client_id
    # FIXME pass pathmnist as input param
    DataClass, _ = get_medmnist_data_info("pathmnist")

    client_dataset = DataClass(
        root=str(path_to_data), download=False, split=split, transform=transforms
    )

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(client_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
