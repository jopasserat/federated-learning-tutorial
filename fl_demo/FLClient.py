from collections import OrderedDict
from pathlib import Path
from typing import Dict

import flwr as fl
from flwr.common.typing import Scalar
import torch
import ray
import numpy as np

from fl_demo.cnn_pathmnist import Net
from fl_demo.dataset_utils import get_dataloader
from fl_demo.train_utils import train
from fl_demo.eval_utils import test


# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class SimulatedFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        fed_dir_data: str,
        in_channels: int,
        num_classes: int,
        criterion: torch.nn.Module,
    ):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # instantiate model
        self.net = Net(in_channels=in_channels, num_classes=num_classes)
        # FIXME ideally the next 2 fields should not be class members..
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
            shuffle=True,
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
            device=self.device,
        )

        # return local model and statistics
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):

        print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader, _ = get_dataloader(
            is_train=False, batch_size=50, workers=num_workers, shuffle=False
        )

        # send model to device
        self.net.to(self.device)

        # evaluate
        loss, accuracy = test(
            model=self.net,
            criterion=self.criterion,
            data_flag=valloader.dataset.flag,
            eval_loader=valloader,
            split="test",
            device=self.device,
        )

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
