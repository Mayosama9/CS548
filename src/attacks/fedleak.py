import torch
import torch.nn as nn


def gradient_distance(g1, g2, top_k_ratio=1.0):
    total_loss = 0

    for grad1, grad2 in zip(g1, g2):
        g1_flat = grad1.view(-1)
        g2_flat = grad2.view(-1)

        if top_k_ratio < 1.0:
            k = max(1, int(top_k_ratio * g1_flat.numel()))
            _, idx = torch.topk(torch.abs(g2_flat), k)
            g1_flat = g1_flat[idx]
            g2_flat = g2_flat[idx]

        total_loss += ((g1_flat - g2_flat) ** 2).sum()

    return total_loss


def infer_label(gradients):
    last_grad = gradients[-1]

    if last_grad.dim() == 1:
        return torch.argmin(last_grad).unsqueeze(0)

    elif last_grad.dim() == 2:
        return torch.argmin(last_grad.sum(dim=1)).unsqueeze(0)

    else:
        raise ValueError("Unexpected gradient shape")


def run_fedleak(model, target_gradients, config):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()

    shape = config["dataset"]["shape"]
    dummy_x = torch.randn(shape, device=device, requires_grad=True)

    # ✅ robust label inference
    dummy_y = infer_label(target_gradients).to(device)

    optimizer = torch.optim.Adam([dummy_x], lr=config["attack"]["lr"])

    lambda_ = config["attack"].get("lambda", 0.5)
    top_k_ratio = config["attack"].get("top_k_ratio", 1.0)

    history = []

    for _ in range(config["attack"]["iterations"]):
        optimizer.zero_grad()

        out = model(dummy_x)
        loss = loss_fn(out, dummy_y)

        dummy_grads = torch.autograd.grad(
            loss, model.parameters(), create_graph=True
        )

        # ===== PARTIAL MATCHING =====
        dist = gradient_distance(dummy_grads, target_gradients, top_k_ratio)

        # ===== GRADIENT =====
        grad = torch.autograd.grad(dist, dummy_x, create_graph=True)[0]

        # ===== STABLE PHI =====
        phi = grad / (grad.norm() + 1e-8)
        phi = phi.clamp(-1, 1)

        # ===== SECOND FORWARD =====
        out_phi = model(dummy_x + phi)
        loss_phi = loss_fn(out_phi, dummy_y)

        dummy_grads_phi = torch.autograd.grad(
            loss_phi, model.parameters(), create_graph=True
        )

        dist_phi = gradient_distance(dummy_grads_phi, target_gradients, top_k_ratio)

        grad_phi = torch.autograd.grad(dist_phi, dummy_x)[0]

        # ===== BLENDED UPDATE =====
        final_grad = (1 - lambda_) * grad + lambda_ * grad_phi

        dummy_x.grad = final_grad

        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([dummy_x], 1.0)

        optimizer.step()

        dummy_x.data.clamp_(0, 1)

        history.append(dist.item())

    return dummy_x.detach(), history