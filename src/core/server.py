from src.attacks.fedleak import run_fedleak
from src.attacks.dlg import run_dlg
from src.evaluation.logger import save_reconstruction


def server_attack(model, gradients, config, round_num, output_dir, cid):

    if config["attack"]["enabled"]:
        if config["attack"]["type"] == "dlg":
            recon, _ = run_dlg(model, gradients, config)

        elif config["attack"]["type"] == "fedleak":
            recon, _ = run_fedleak(model, gradients, config)

        save_reconstruction(recon, round_num, output_dir, cid)