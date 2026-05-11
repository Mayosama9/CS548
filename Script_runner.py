import os
import yaml
import itertools
import subprocess
import tempfile

# =========================
# PARAM GRID
# =========================
attack_types = ["fedleak", "dlg"]
defense_enabled = [True, False]
alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 100]  # 100 ≈ IID

base_config_path = "configs/config.yaml"

# =========================
# LOAD BASE CONFIG
# =========================
with open(base_config_path, "r") as f:
    base_cfg = yaml.safe_load(f)

# =========================
# RUN ALL COMBINATIONS
# =========================
for attack, defense, alpha in itertools.product(
    attack_types,
    defense_enabled,
    alphas
):
    cfg = base_cfg.copy()

    # Update values
    cfg["attack"]["type"] = attack
    cfg["defense"]["enabled"] = defense
    cfg["data"]["alpha"] = alpha

    print(f"\nRunning: attack={attack}, defense={defense}, alpha={alpha}")

    # Create temp config file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tmp:
        yaml.dump(cfg, tmp)
        temp_config_path = tmp.name

    # Run main.py
    subprocess.run([
        "python", "main.py",
        "--config", temp_config_path
    ])

    # Remove temp config
    os.remove(temp_config_path)

print("\nAll experiments completed!")