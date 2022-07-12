from typing import Dict, Tuple

from torch.utils.data import Dataset, DataLoader
import medmnist
from medmnist import INFO
from medmnist.info import DEFAULT_ROOT

from fl_demo.cnn_pathmnist import pathmnist_transforms


def get_medmnist_data_info(dataset_title: str) -> Tuple[Dataset, Dict]:
    data_flag = dataset_title
    info = INFO[data_flag]

    DataClass = getattr(medmnist, info["python_class"])

    return DataClass, info


def get_dataset(
    base_path: str = DEFAULT_ROOT,
    dataset_title: str = "pathmnist",
    download: bool = True,
    split: str = "train",
) -> Tuple[Dataset, Dict]:
    """Fetch centralised version of the dataset if needed."""

    # preprocessing
    data_transform = pathmnist_transforms()

    DataClass, info = get_medmnist_data_info(dataset_title)

    # load the data
    dataset = DataClass(
        root=base_path, split=split, transform=data_transform, download=download
    )

    return dataset, info


def get_dataloader(
    is_train: bool, batch_size: int, workers: int, shuffle: bool
) -> DataLoader:
    """
    Generates trainset/valset object and returns appropiate dataloader.
    This is a centralised version.
    See fl_utils for the federated equivalent
    """

    split = "train" if is_train else "test"
    dataset, _ = get_dataset(download=True, split=split)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
