import torch
import torch.nn as nn


# =========================
# GRADIENT DISTANCE (ROBUST)
# =========================
def gradient_distance(g1, g2):
    loss = 0
    for a, b in zip(g1, g2):
        loss += ((a - b) ** 2).mean()
    return loss


# =========================
# LABEL INFERENCE (KEEP)
# =========================
def infer_label(gradients):
    last_grad = gradients[-1]

    if last_grad.dim() == 1:
        return torch.argmin(last_grad).unsqueeze(0)
    elif last_grad.dim() == 2:
        return torch.argmin(last_grad.sum(dim=1)).unsqueeze(0)
    else:
        raise ValueError("Unexpected gradient shape")


# =========================
# TOTAL VARIATION (LIGHT)
# =========================
def total_variation(x):
    return (
        torch.mean(torch.abs(x[:, :, :-1] - x[:, :, 1:])) +
        torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    )


# =========================
# MAIN DLG (ROBUST VERSION)
# =========================
def run_dlg(model, gradients, config):
    device = next(model.parameters()).device
    model.eval()

    target_gradients = [g.detach().clone().to(device) for g in gradients]

    shape = config["dataset"]["shape"]
    dummy_data = torch.randn(shape, device=device, requires_grad=True)

    dummy_label = infer_label(target_gradients).to(device)

    # 🔥 USE ADAM (NOT LBFGS)
    optimizer = torch.optim.Adam([dummy_data], lr=config["attack"]["lr"])

    loss_fn = nn.CrossEntropyLoss()

    tv_weight = config["attack"].get("tv_weight", 1e-4)

    history = []

    for _ in range(config["attack"]["iters"]):
        optimizer.zero_grad()

        output = model(dummy_data)
        loss = loss_fn(output, dummy_label)

        dummy_grads = torch.autograd.grad(
            loss, model.parameters(), create_graph=True
        )

        # =========================
        # GRADIENT MATCHING
        # =========================
        grad_loss = gradient_distance(dummy_grads, target_gradients)

        # =========================
        # TV REGULARIZATION
        # =========================
        tv_loss = total_variation(dummy_data)

        # 🔥 BALANCED LOSS (IMPORTANT)
        total_loss = grad_loss + tv_weight * tv_loss

        total_loss.backward()

        # 🔥 CLIP FOR STABILITY
        torch.nn.utils.clip_grad_norm_([dummy_data], 1.0)

        optimizer.step()

        # clamp image
        dummy_data.data.clamp_(0, 1)

        history.append(grad_loss.item())

    return dummy_data.detach(), dummy_label.detach()