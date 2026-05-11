import torch


def train(model, loader, optimizer, loss_fn, epochs, defense_fn=None):
    model.train()
    device = next(model.parameters()).device

    total_loss = 0
    first_sample = None

    for _ in range(epochs):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # =========================
            # APPLY DEFENSE ONLY FIRST BATCH (OPT 2)
            # =========================
            if defense_fn is not None and i <3:
                out = defense_fn(model, x)
            else:
                out = model(x)

            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if first_sample is None:
                first_sample = (x[0].detach().cpu(), y[0].detach().cpu())

    return total_loss / len(loader), first_sample