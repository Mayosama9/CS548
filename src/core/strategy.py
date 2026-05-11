import flwr as fl
import torch
import os
import random
import pandas as pd
import torchvision.utils as vutils

from flwr.common import parameters_to_ndarrays
from torch.utils.data import DataLoader

from src.attacks.dlg import run_dlg
from src.attacks.fedleak import run_fedleak


# =========================
# HELPERS
# =========================
def weights_to_torch(weights, model):
    keys = model.state_dict().keys()
    return {k: torch.tensor(v).float() for k, v in zip(keys, weights)}


def compute_gradients(global_w, client_w, lr):
    return [(gw - cw) / lr for gw, cw in zip(global_w, client_w)]


def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    model.eval()
    correct, total, loss_total = 0, 0, 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y)

            loss_total += loss.item() * x.size(0)
            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total, loss_total / total


# =========================
# STRATEGY
# =========================
class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, config, model, output_dir,
                 test_dataset, client_data, train_dataset,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cfg = config
        self.model = model
        self.output_dir = output_dir

        self.test_dataset = test_dataset
        self.client_data = client_data
        self.train_dataset = train_dataset

        self.prev_weights = None

        # Logs
        self.mse_log = []
        self.acc_log = []
        self.loss_log = []
        self.success_log = []

        self.convergence_round = None

        # Communication tracking
        self.model_size_mb = (
            sum(p.numel() for p in self.model.parameters()) * 4 / (1024**2)
        )

        self.clients_per_round = int(
            self.cfg["federated"]["num_clients"] *
            self.cfg["federated"]["client_fraction"]
        )

    # =========================
    # AGGREGATE FIT
    # =========================
    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated is None:
            return None

        weights = parameters_to_ndarrays(aggregated[0])
        device = next(self.model.parameters()).device

        self.model.load_state_dict(
            weights_to_torch(weights, self.model)
        )

        # =========================
        # EVALUATION
        # =========================
        acc, loss = evaluate(self.model, self.test_dataset, device)

        self.acc_log.append([rnd, acc])
        self.loss_log.append([rnd, loss])

        print(f"[Round {rnd}] Accuracy: {acc:.4f}, Loss: {loss:.4f}")

        if self.convergence_round is None and acc >= 0.80:
            self.convergence_round = rnd

        # First round skip attack
        if self.prev_weights is None:
            self.prev_weights = weights
            return aggregated

        print(f"\n[Round {rnd}] Running Privacy Attack...")

        success_count = 0
        total_count = 0

        # =========================
        # ATTACK LOOP
        # =========================
        for client, res in results:
            cid = int(res.metrics["cid"])

            grads_np = compute_gradients(
                self.prev_weights,
                parameters_to_ndarrays(res.parameters),
                self.cfg["training"]["lr"]
            )

            grads = [
                torch.tensor(g).float().to(device)
                for g in grads_np if g is not None
            ]

            self.model.load_state_dict(
                weights_to_torch(self.prev_weights, self.model)
            )

            # ===== ATTACK SWITCH =====
            if self.cfg["attack"]["type"] == "fedleak":
                recon, _ = run_fedleak(self.model, grads, self.cfg)
            else:
                recon, _ = run_dlg(self.model, grads, self.cfg)

            if torch.isnan(recon).any():
                continue

            # =========================
            # GET ORIGINAL SAMPLE
            # =========================
            idx = random.choice(self.client_data[cid])
            orig, _ = self.train_dataset[idx]

            os.makedirs(f"{self.output_dir}/originals", exist_ok=True)
            os.makedirs(f"{self.output_dir}/reconstructions", exist_ok=True)

            vutils.save_image(orig, f"{self.output_dir}/originals/r{rnd}_c{cid}.png")
            vutils.save_image(recon, f"{self.output_dir}/reconstructions/r{rnd}_c{cid}.png")

            recon = recon.squeeze().cpu()

            if orig.shape != recon.shape:
                orig = torch.nn.functional.interpolate(
                    orig.unsqueeze(0),
                    size=recon.shape[1:]
                ).squeeze(0)

            # =========================
            # MSE + SUCCESS
            # =========================
            mse = torch.mean((orig - recon) ** 2).item()
            self.mse_log.append([rnd, cid, mse])

            success = 1 if mse < 0.05 else 0
            success_count += success
            total_count += 1

            print(f"[Round {rnd}] Client {cid} MSE: {mse:.4f}")

        # =========================
        # SUCCESS RATE
        # =========================
        if total_count > 0:
            success_rate = success_count / total_count
            self.success_log.append([rnd, success_rate])

        # =========================
        # SAVE METRICS
        # =========================
        os.makedirs(f"{self.output_dir}/metrics", exist_ok=True)

        pd.DataFrame(self.acc_log, columns=["round", "accuracy"])\
            .to_csv(f"{self.output_dir}/metrics/accuracy.csv", index=False)

        pd.DataFrame(self.loss_log, columns=["round", "loss"])\
            .to_csv(f"{self.output_dir}/metrics/loss.csv", index=False)

        if len(self.mse_log) > 0:
            pd.DataFrame(self.mse_log, columns=["round", "client", "mse"])\
                .to_csv(f"{self.output_dir}/metrics/mse.csv", index=False)

        if len(self.success_log) > 0:
            pd.DataFrame(self.success_log, columns=["round", "attack_success"])\
                .to_csv(f"{self.output_dir}/metrics/attack_success.csv", index=False)

        # =========================
        # CONVERGENCE
        # =========================
        if self.convergence_round is not None:
            with open(f"{self.output_dir}/metrics/convergence.txt", "w") as f:
                f.write(str(self.convergence_round))

        # =========================
        # COMMUNICATION COST
        # =========================
        total_comm = self.model_size_mb * self.clients_per_round * rnd

        with open(f"{self.output_dir}/metrics/communication.txt", "w") as f:
            f.write(f"{total_comm:.4f}")

        self.prev_weights = weights

        return aggregated