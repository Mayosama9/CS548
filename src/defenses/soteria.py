import torch


def apply_soteria(model, x, config):
    """
    Optimized Soteria:
    - Gradient-based sensitivity (paper idea)
    - Feature sampling (speed)
    - Designed to be used on first batch only
    """

    device = next(model.parameters()).device
    x = x.clone().detach().to(device)
    x.requires_grad = True

    # =========================
    # FORWARD PASS
    # =========================
    _ = model(x)
    r = model.get_representation()  # [B, D]

    B, D = r.shape

    # =========================
    # FEATURE SAMPLING (OPT 1)
    # =========================
    sample_size = min(64, D)   # you can tune: 16–64
    sampled_idx = torch.randperm(D)[:sample_size]

    sensitivity = torch.zeros_like(r)

    for i in sampled_idx:
        model.zero_grad()

        grad_outputs = torch.zeros_like(r)
        grad_outputs[:, i] = 1.0

        grads = torch.autograd.grad(
            outputs=r,
            inputs=x,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False
        )[0]

        sensitivity[:, i] = grads.view(B, -1).norm(dim=1)
    sensitivity = sensitivity / (sensitivity.max(dim=1, keepdim=True)[0] + 1e-8)
    # =========================
    # SELECT TOP-K SENSITIVE FEATURES
    # =========================
    ratio = config["defense"].get("top_k_ratio", 0.01)
    k = max(1, int(ratio * sample_size))

    sampled_sens = sensitivity[:, sampled_idx]

    topk_vals = torch.topk(sampled_sens, k, dim=1)[0]
    threshold = topk_vals[:, -1].unsqueeze(1)

    mask_sampled = (sampled_sens >= threshold).float()

    # full mask
    mask = torch.zeros_like(r)
    mask[:, sampled_idx] = mask_sampled

    # =========================
    # SUPPRESS SENSITIVE FEATURES
    # =========================
    r_perturbed = r * (1 - mask)

    # optional noise
    noise_scale = config["defense"].get("noise_scale", 0.1)
    noise = torch.randn_like(r).to(device) * noise_scale

    r_perturbed = r_perturbed + noise * mask

    # stability
    r_perturbed = torch.clamp(r_perturbed, -1.0, 1.0)

    # =========================
    # FINAL FORWARD
    # =========================
    out = model.fc2(r_perturbed)

    return out