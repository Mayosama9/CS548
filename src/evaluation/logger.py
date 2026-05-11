import os
import torch
import torchvision.utils as vutils


def save_reconstruction(img, rnd, output_dir, cid):
    path = os.path.join(
        output_dir,
        "reconstructions",
        f"round_{rnd}_client_{cid}.png"
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(img, path)


def save_model(model, rnd, output_dir):
    path = os.path.join(
        output_dir,
        "models",
        f"round_{rnd}.pt"
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)