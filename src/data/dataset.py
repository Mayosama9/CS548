import numpy as np
from torchvision import datasets, transforms


def get_dataset(name):
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test = datasets.MNIST("./data", train=False, download=True, transform=transform)
        in_channels = 1

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        in_channels = 3

    return train, test, in_channels


def dirichlet_split(dataset, num_clients, alpha):
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    client_data = [[] for _ in range(num_clients)]

    for c in np.unique(labels):
        idx_c = idxs[labels == c]
        np.random.shuffle(idx_c)

        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]

        splits = np.split(idx_c, proportions)

        for i, split in enumerate(splits):
            if len(split) > 0:
                client_data[i].extend(split)

    return client_data