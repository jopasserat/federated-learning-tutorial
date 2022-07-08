import torch
import torch.nn as nn
import torch.optim as optim

from fl_demo.cnn_pathmnist import Net
import fl_demo.train_pathmnist as client_pathmnist
from fl_demo.dataset_utils import get_dataloader

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001


#### read data and load in DataLoader

# encapsulate data into dataloader form
train_loader, info      = get_dataloader(is_train=True, batch_size=BATCH_SIZE, workers=2, shuffle=True)
train_loader_at_eval, _ = get_dataloader(is_train=True, batch_size=2*BATCH_SIZE, workers=2, shuffle=False)
test_loader, _          = get_dataloader(is_train=False, batch_size=2*BATCH_SIZE, workers=2, shuffle=False)

n_channels = info['n_channels']
n_classes = len(info['label'])



# visualization
train_loader.dataset.montage(length=1)
# visualization
train_loader.dataset.montage(length=20)



model = Net(in_channels=n_channels, num_classes=n_classes)
criterion = nn.CrossEntropyLoss()    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

client_pathmnist.train(model, optimizer, criterion, train_loader, NUM_EPOCHS, device)
client_pathmnist.test(model, criterion=criterion, data_flag="pathmnist", eval_loader=train_loader_at_eval, split="train", device=device)
client_pathmnist.test(model, criterion=criterion, data_flag="pathmnist", eval_loader=test_loader, split="test", device=device)

# TODO expose dataset's root folder parameter somewhere