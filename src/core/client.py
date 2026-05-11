import torch
from torch.utils.data import DataLoader, Subset
import flwr as fl
from src.defenses.soteria import apply_soteria
from src.core.trainer import train


class FLClient(fl.client.NumPyClient):
    def __init__(self, model, dataset, idxs, config):
        self.model = model
        self.config = config

        subset = Subset(dataset, idxs)

        self.loader = DataLoader(
            subset,
            batch_size=config["training"]["batch_size"],
            shuffle=True
        )

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({
            k: torch.tensor(v) for k, v in state_dict.items()
        })

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config["training"]["lr"],
            momentum=self.config["training"]["momentum"]
        )

        loss_fn = torch.nn.CrossEntropyLoss()

        defense_fn=None
        if self.config["defense"]["enabled"]:
            defense_fn = lambda m, x: apply_soteria(m, x, self.config)

        loss, _ = train(
            self.model,
            self.loader,
            optimizer,
            loss_fn,
            self.config["training"]["local_epochs"],
            defense_fn
        )

        return (
            self.get_parameters(config),
            len(self.loader.dataset),
            {
                "train_loss": float(loss),
                "cid": int(self.config["cid"])
            }
        )