from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from fl_demo.cnn_pathmnist import Net, pathmnist_transforms
from fl_demo import train_utils
from fl_demo import eval_utils
from fl_demo.dataset_utils import get_dataloader, get_dataset
from fl_demo.fl_utils import get_federated_dataloader

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001


#### read data and load in DataLoader

_, info = get_dataset(split="train")

# encapsulate data into dataloader form
train_loader = get_dataloader(
    is_train=True, batch_size=BATCH_SIZE, workers=2, shuffle=True
)

# simulate loading from first client
# FIXME data must be partitioned first
fed_train_loader = get_federated_dataloader(
    base_path=Path("/home/jopasserat/.medmnist/federated"),
    client_id="0",
    is_train=True,
    batch_size=BATCH_SIZE,
    workers=2,
    shuffle=True,
    transforms=pathmnist_transforms(),
)

train_loader_at_eval = get_dataloader(
    is_train=True, batch_size=2 * BATCH_SIZE, workers=2, shuffle=False
)
test_loader = get_dataloader(
    is_train=False, batch_size=2 * BATCH_SIZE, workers=2, shuffle=False
)

n_channels = info["n_channels"]
n_classes = len(info["label"])

# visualization (returns PIL image, should be displayed in Jupyter cell)
train_loader.dataset.montage(length=1)
train_loader.dataset.montage(length=20)

model = Net(in_channels=n_channels, num_classes=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train_utils.train(model, optimizer, criterion, train_loader, NUM_EPOCHS, device)
train_utils.train(model, optimizer, criterion, fed_train_loader, NUM_EPOCHS, device)
eval_utils.test(
    model,
    criterion=criterion,
    data_flag="pathmnist",
    eval_loader=train_loader_at_eval,
    split="train",
    device=device,
)
eval_utils.test(
    model,
    criterion=criterion,
    data_flag="pathmnist",
    eval_loader=test_loader,
    split="test",
    device=device,
)
