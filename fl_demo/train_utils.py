from typing import Dict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from opacus import PrivacyEngine


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: data.DataLoader,
    NUM_EPOCHS: int,
    device: torch.device,
) -> None:
    for epoch in range(NUM_EPOCHS):

        model.train()

        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()


def dp_train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: data.DataLoader,
    NUM_EPOCHS: int,
    device: torch.device,
    privacy_engine: PrivacyEngine,
    dp_config: Dict,
) -> Dict:

    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        NUM_EPOCHS=NUM_EPOCHS,
        device=device,
    )

    epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
        delta=dp_config["delta"],
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
    )
    # debug print
    print(f"(ε = {epsilon:.2f}, δ = {dp_config['delta']}) for α = {best_alpha}")

    dp_stats = {
        "epsilon": f"{epsilon:.2f}",
        "delta": dp_config["delta"],
        "alpha": best_alpha,
    }

    return dp_stats
