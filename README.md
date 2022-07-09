# FL simulation with medical imaging classification task


This code splits the Pathology MedMNIST dataset into `pool_size` partitions (user defined) and does a few rounds of training.


## Requirements

*    Flower 0.19.0
*    A recent version of PyTorch. This example has been tested with Pytorch 1.11.0
*    A recent version of Ray. This example has been tested with Ray 1.11.1.


### Install

Create a new Conda environment with Python 3.9, the following commands will isntall all the dependencies needed:
```
conda create --name my_project_env --file conda-linux-64.lock
poetry install
```

## How to run

This example:

1. Downloads Pathology MedMNIST
2. Partitions the dataset into N splits, where N is the total number of
   clients. We refere to this as `pool_size`. The partition can be IID or non-IID
4. Starts a Ray-based simulation where a % of clients are sample each round.
   This example uses N=3, so 3 clients will be sampled each round.
5. After the M rounds end, the global model is evaluated on the entire testset.
   Also, the global model is evaluated on the valset partition residing in each
   client. This is useful to get a sense on how well the global model can generalise
   to each client's data.

The command below will assign each client 1 CPU threads. If your system does not have 1xN(=3) = 3 threads to run all 3 clients in parallel, they will be queued but eventually run. The server will wait until all N clients have completed their local training stage before aggregating the results. After that, a new round will begin.

```bash
$ python main.py --num_client_cpus 2 # note that `num_client_cpus` should be <= the number of threads in your system.
```

## References

- MedMNIST code adapted from this [Getting Started](https://github.com/MedMNIST/MedMNIST/blob/d8422ac64028488133fd21ff54372729e12bbaba/examples/getting_started.ipynb) example.
- Flower code adapted from this example: https://github.com/adap/flower/tree/2d45f12189984c2901d54e295f5c684b07039bd8/examples/simulation_pytorch