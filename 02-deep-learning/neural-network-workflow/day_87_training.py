import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from day_85_dataset import build_dataloaders
from day_86_architecture import RetailDemandMLP


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Runs one full pass over training batches and returns (mean_mse, mean_mae)."""
    model.train()
    total_mse_sum = 0.0
    total_mae_sum = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_size = batch_x.shape[0]

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mae_batch = torch.abs(preds - batch_y).sum().item()
            total_mse_sum += loss.item() * batch_size
            total_mae_sum += mae_batch
            total_samples += batch_size

    return total_mse_sum / total_samples, total_mae_sum / total_samples


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 25,
    log_every: int = 5,
) -> list[dict[str, float]]:
    """Trains the model across multiple epochs and records training history."""
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_mse, train_mae = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        train_rmse = float(np.sqrt(train_mse))
        record = {
            "epoch": float(epoch),
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
        }
        history.append(record)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Train MSE: {train_mse:8.2f} | "
                f"RMSE: {train_rmse:6.2f} units | "
                f"MAE: {train_mae:6.2f} units"
            )

    return history


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, num_features, source = build_dataloaders(batch_size=32)
    model = RetailDemandMLP(
        in_features=num_features,
        hidden_dims=(64, 32),
        dropout=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    print("--- Day 87: Neural Network Training Loop ---")
    print(f"Data source: {source} | Device: {device} | Features: {num_features}\n")

    history = fit_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=25,
        log_every=5,
    )