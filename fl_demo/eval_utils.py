from typing import Tuple, OrderedDict, Callable, Optional

import torch
import torch.nn as nn
import torch.utils.data as data
from medmnist import Evaluator
import numpy as np
import flwr as fl

from fl_demo.cnn_pathmnist import Net


def get_eval_fn(
    testset: torch.utils.data.Dataset,
    criterion: torch.nn.Module,
    in_channels: int,
    num_classes: int
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net(num_classes=num_classes, in_channels=in_channels)
        set_weights(model, weights)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy, auc = test(
          model=model,
          criterion=criterion,
          data_flag=testset.flag,
          eval_loader=testloader,
          split="test",
          device=device
        )

        # return statistics
        return loss, {"accuracy": accuracy, "auc": auc}

    return evaluate


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


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
