import torch

def mse(x, y):
    return ((x - y) ** 2).mean().item()