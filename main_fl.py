import argparse
from typing import Dict

import flwr as fl
import torch

from fl_demo.cnn_pathmnist import Net
from fl_demo.dataset_utils import get_dataset
from fl_demo.dp_utils import fix_model_layers
from fl_demo.fl_utils import do_fl_partitioning
from fl_demo.FLClient import SimulatedFLClient
from fl_demo.eval_utils import centralised_eval_fn


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(2),  # number of local epochs
        "batch_size": str(128),
    }
    return config


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
    parser.add_argument("--train_with_dp", action="store_true", default=True)

    args = parser.parse_args()

    pool_size = 3  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs
    dp_config = {
        "max_per_sample_grad_norm": 10.0,
        "noise_multiplier": 1.5,  # sigma
        "secure_rng": False,
        "delta": 1e-5,
        "clip_per_layer": False,
    }

    ### download  dataset ###
    trainset, info = get_dataset(split="train")
    testset, _ = get_dataset(split="test")

    n_classes = len(info["label"])
    n_channels = info["n_channels"]

    criterion = torch.nn.CrossEntropyLoss()

    model = Net(num_classes=n_classes, in_channels=n_channels)
    model = fix_model_layers(model) if args.train_with_dp else model

    # TODO expose alpha param
    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # the base dataset lives. Inside it, there will be N=pool_size sub-directories each with # FIXME
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        trainset,
        pool_size=pool_size,
        alpha=1000,
        num_classes=n_classes,
        val_ratio=0.1,
    )

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=centralised_eval_fn(
            testset,
            model=model,
            criterion=criterion
        ),  # centralised testset evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        return SimulatedFLClient(
            cid,
            fed_data_dir=fed_dir,
            model=model,
            criterion=criterion,
            data_flag=trainset.flag,
            with_dp=args.train_with_dp,
            dp_config=dp_config,
        )

    # (optional) specify ray config, set local_mode to true for serial debugging
    ray_config = {"include_dashboard": False, "local_mode": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=args.num_rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    )
