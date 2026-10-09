"""Federated learning layer (Week 4 onward).

The pieces are deliberately separable, because each one fails in its own way and
you want to know which one broke:

    params.py   which tensors leave the client, and in what order
    data.py     partition parquet -> per-client datasets
    task.py     local training and evaluation
    client.py   Flower ClientApp adapter
    server.py   Flower ServerApp + FedAvg strategy

Nothing here builds a partition. The federated phases read the frozen Week-3
parquets, which regenerate byte-identically from seed 42 and are what every
committed measurement describes.
"""
