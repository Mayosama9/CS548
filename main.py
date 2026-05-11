import os
import argparse
import yaml
import torch
import flwr as fl
import random
import numpy as np
from datetime import datetime

from src.models.cnn import CNN
from src.data.dataset import get_dataset, dirichlet_split
from src.core.client import FLClient
from src.core.strategy import CustomStrategy
from src.utils.helpers import load_config
from src.evaluation.plots import plot_from_csv


# =========================
# SEED
# =========================
def set_seed(cfg):
    random.seed(cfg["seed"]["python"])
    np.random.seed(cfg["seed"]["numpy"])
    torch.manual_seed(cfg["seed"]["torch"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"]["torch"])


# =========================
# OUTPUT DIR
# =========================
def create_output_dir(base_path, cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    name = (
        f"{cfg['experiment']['name']}"
        f"_alpha={cfg['data']['alpha']}"
        # f"_c{cfg['federated']['num_clients']}"
        # f"_r{cfg['federated']['rounds']}"
        f"_def={cfg['defense']['enabled']}"
        f"_att={cfg['attack']['type']}"
        f"_{timestamp}"
    )

    path = os.path.join(base_path, name)

    for folder in ["plots", "reconstructions", "originals", "models", "metrics"]:
        os.makedirs(os.path.join(path, folder), exist_ok=True)

    return path


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    set_seed(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # OUTPUT DIR
    # =========================
    output_dir = create_output_dir(
        cfg["experiment"]["output_dir"],
        cfg
    )

    cfg["experiment"]["output_dir"] = output_dir

    print(f"Saving results to: {output_dir}")

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # =========================
    # DATA
    # =========================
    train_dataset, test_dataset, in_channels = get_dataset(
        cfg["dataset"]["name"]
    )

    client_data = dirichlet_split(
        train_dataset,
        cfg["federated"]["num_clients"],
        cfg["data"]["alpha"]
    )

    # =========================
    # MODEL
    # =========================
    model = CNN(
        num_classes=cfg["dataset"]["num_classes"],
        in_channels=in_channels
    ).to(device)

    # =========================
    # STRATEGY
    # =========================
    strategy = CustomStrategy(
        config=cfg,
        model=model,
        output_dir=output_dir,
        test_dataset=test_dataset,
        client_data=client_data,
        train_dataset=train_dataset,
        fraction_fit=cfg["federated"]["client_fraction"],
        min_fit_clients=int(
            cfg["federated"]["num_clients"]
            * cfg["federated"]["client_fraction"]
        ),
        min_available_clients=cfg["federated"]["num_clients"]
    )

    # =========================
    # CLIENT FUNCTION
    # =========================
    def client_fn(context):
        raw_id = int(context.node_id)
        cid = raw_id % cfg["federated"]["num_clients"]

        local_cfg = dict(cfg)
        local_cfg["cid"] = cid

        return FLClient(
            CNN(
                num_classes=cfg["dataset"]["num_classes"],
                in_channels=in_channels
            ).to(device),
            train_dataset,
            client_data[cid],
            local_cfg
        ).to_client()

    # =========================
    # RUN SIMULATION
    # =========================
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg["federated"]["num_clients"],
        config=fl.server.ServerConfig(
            num_rounds=cfg["federated"]["rounds"]
        ),
        strategy=strategy,
    )

    # =========================
    # PLOT RESULTS
    # =========================
    plot_from_csv(output_dir)

    print("Training completed!")


if __name__ == "__main__":
    main()