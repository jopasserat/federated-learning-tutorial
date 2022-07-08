import argparse
import flwr as fl
from flwr.common.typing import Scalar
import ray
import torch
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
from dataset_utils import get_dataset, do_fl_partitioning, get_dataloader
from train_pathmnist import train, test
from cnn_pathmnist import Net


# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class SimulatedFLClient(fl.client.NumPyClient):
    def __init__(self, 
      cid: str, 
      fed_dir_data: str,
      in_channels: int,
      num_classes: int,
      criterion: torch.nn.Module
    ):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # instantiate model
        self.net = Net(in_channels=in_channels, num_classes=num_classes)
        # FIXME ideally the next 2 should not be class members..
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.criterion = criterion

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainloader, _ = get_dataloader(
            # self.fed_dir,
            # self.cid,
            is_train=True,
            batch_size=int(config["batch_size"]),
            workers=num_workers,
            shuffle=True
        )

        # send model to device
        self.net.to(self.device)

        # train
        train(
          model=self.net,
          optimizer=self.optimizer,
          criterion=self.criterion,
          train_loader=trainloader,
          NUM_EPOCHS=int(config["epochs"]),
          device=self.device
        )

        # return local model and statistics
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):

        print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader, _ = get_dataloader(
          is_train=False,
          batch_size=50,
          workers=num_workers,
          shuffle=False
        )

        # send model to device
        self.net.to(self.device)

        # evaluate
        loss, accuracy = test(
          model=self.net,
          criterion=self.criterion,
          data_flag="pathmnist",
          eval_loader=valloader,
          split="test",
          device=self.device
        )

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(2),  # number of local epochs
        "batch_size": str(128),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


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
          data_flag="pathmnist",
          eval_loader=testloader,
          split="test",
          device=device
        )


        # return statistics
        return loss, {"accuracy": accuracy, "auc": auc}

    return evaluate


# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Downloads the Pathology MedMNIST dataset
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a Ray-based simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    ### parse input arguments ###
    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

    parser.add_argument("--num_client_cpus", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=2)

    args = parser.parse_args()

    pool_size = 3  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs

    ### download  dataset ###
    train_path, info = get_dataset(split="train")
    testset, _       = get_dataset(split="test")

    n_classes  = len(info['label'])
    n_channels = info['n_channels']

    criterion = torch.nn.CrossEntropyLoss()

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with # FIXME
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1000, num_classes=n_classes, val_ratio=0.1
    )

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(testset, criterion=criterion, in_channels=n_channels, num_classes=n_classes),  # centralised testset evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        return SimulatedFLClient(cid, fed_dir, in_channels=n_channels, num_classes=n_classes, criterion=criterion)

    # (optional) specify ray config
    ray_config = {
      "include_dashboard": False
    }

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=args.num_rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    )
