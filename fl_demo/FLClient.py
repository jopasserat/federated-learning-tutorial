from collections import OrderedDict
from pathlib import Path
from typing import Dict

import flwr as fl
from flwr.common.typing import Scalar
import torch
import ray
import numpy as np

from fl_demo.cnn_pathmnist import pathmnist_transforms
from fl_demo.debug_utils import exception_message_with_line_number
from fl_demo.dp_utils import configure_dp_training
from fl_demo.fl_utils import get_federated_dataloader
from fl_demo.train_utils import dp_train, train
from fl_demo.eval_utils import test


# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class SimulatedFLClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        fed_data_dir: str,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_flag: str,
        with_dp: bool = False,
        dp_config: Dict = None,
    ):
        self.cid = cid
        self.fed_dir = Path(fed_data_dir)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.data_flag = data_flag

        # store DP setting for `train` method
        self.dp_config = dp_config
        self.with_dp = with_dp

        # instantiate model
        self.net = model

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
        trainloader = get_federated_dataloader(
            base_path=self.fed_dir,
            client_id=self.cid,
            is_train=True,
            batch_size=int(config["batch_size"]),
            workers=num_workers,
            shuffle=True,
            transforms=pathmnist_transforms(),
        )

        # send model to device
        self.net.to(self.device)

        # enable and configure DP if needed
        dp_stats = {}

        if self.with_dp:
            (
                self.net,
                self.optimizer,
                trainloader,
                privacy_engine,
            ) = configure_dp_training(
                dp_config=self.dp_config,
                model=self.net,
                optimizer=self.optimizer,
                train_loader=trainloader,
            )
            dp_stats = dp_train(
                model=self.net,
                optimizer=self.optimizer,
                criterion=self.criterion,
                train_loader=trainloader,
                NUM_EPOCHS=int(config["epochs"]),
                device=self.device,
                privacy_engine=privacy_engine,
                dp_config=self.dp_config,
            )
        # non-private train
        else:
            train(
                model=self.net,
                optimizer=self.optimizer,
                criterion=self.criterion,
                train_loader=trainloader,
                NUM_EPOCHS=int(config["epochs"]),
                device=self.device,
            )


        # return local model and statistics
        return self.get_parameters(), len(trainloader.dataset), dp_stats

    def evaluate(self, parameters, config):

        print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = (
            len(ray.worker.get_resource_ids()["CPU"]) if ray.is_initialized() else 1
        )

        valloader = get_federated_dataloader(
            base_path=self.fed_dir,
            client_id=self.cid,
            is_train=False,
            batch_size=50,
            workers=num_workers,
            shuffle=False,
            transforms=pathmnist_transforms(),
        )

        # send model to device
        self.net.to(self.device)

        # evaluate
        loss, accuracy, auc = test(
            model=self.net,
            criterion=self.criterion,
            data_flag=self.data_flag,
            eval_loader=valloader,
            split="val",
            device=self.device,
        )

        # return statistics
        return (
            float(loss),
            len(valloader.dataset),
            {"accuracy": float(accuracy), "auc": auc},
        )
