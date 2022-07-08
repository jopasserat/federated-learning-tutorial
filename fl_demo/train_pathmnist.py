from typing import Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from medmnist import Evaluator


def train(model: nn.Module,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          train_loader: data.DataLoader,
          NUM_EPOCHS: int,
          device: torch.device
):
  for epoch in range(NUM_EPOCHS):

    model.train()

    for inputs, targets in tqdm(train_loader):
      inputs, targets = inputs.to(device), targets.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)

      targets = targets.squeeze().long()
      loss = criterion(outputs, targets)

      loss.backward()
      optimizer.step()


def test(model: nn.Module,
         criterion: nn.Module,
         data_flag: str,
         eval_loader: data.DataLoader,
         split: str,
         device: torch.device
) -> Tuple[float, float, float]:
  model.eval()
  y_true = torch.tensor([])
  y_score = torch.tensor([])
  loss = 0.0

  with torch.no_grad():
    for inputs, targets in eval_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)

      targets = targets.squeeze().long()
      loss += criterion(outputs, targets).item()

      outputs = outputs.softmax(dim=-1)

      targets = targets.float().resize_(len(targets), 1)

      y_true = torch.cat((y_true, targets), 0)
      y_score = torch.cat((y_score, outputs), 0)

    y_true = y_true.numpy()
    y_score = y_score.detach().numpy()
    
    evaluator = Evaluator(data_flag, split)
    metrics = evaluator.evaluate(y_score)

    auc, acc = metrics
    print('%s  auc: %.3f  acc:%.3f' % (split, auc, acc))

    return loss, acc, auc
