from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from day_85_dataset import build_dataloaders
from day_86_architecture import RetailDemandMLP
from day_87_training import set_seed, train_one_epoch


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluates model on a DataLoader without gradient tracking."""
    model.eval()
    total_mse_sum = 0.0
    total_mae_sum = 0.0
    total_samples = 0

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size = batch_x.shape[0]

            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            total_mse_sum += loss.item() * batch_size
            total_mae_sum += torch.abs(preds - batch_y).sum().item()
            total_samples += batch_size

            all_preds.append(preds.squeeze(-1).cpu().numpy())
            all_targets.append(batch_y.squeeze(-1).cpu().numpy())

    mse = total_mse_sum / total_samples
    rmse = float(np.sqrt(mse))
    mae = total_mae_sum / total_samples

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def fit_with_evaluation(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 30,
    log_every: int = 5,
    checkpoint_path: str | None = None,
) -> list[dict[str, float]]:
    """Trains across epochs and tracks held-out metrics each epoch."""
    history: list[dict[str, float]] = []
    best_val_rmse = float("inf")

    for epoch in range(1, epochs + 1):
        train_mse, train_mae = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        train_rmse = float(np.sqrt(train_mse))

        val_metrics = evaluate(
            model=model,
            dataloader=eval_loader,
            criterion=criterion,
            device=device,
        )

        if val_metrics["rmse"] < best_val_rmse and checkpoint_path is not None:
            best_val_rmse = val_metrics["rmse"]
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        row = {
            "epoch": float(epoch),
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
        }
        history.append(row)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Train RMSE: {train_rmse:6.2f} | "
                f"Val RMSE: {val_metrics['rmse']:6.2f} | "
                f"Val MAE: {val_metrics['mae']:6.2f} | "
                f"Val R2: {val_metrics['r2']:6.4f}"
            )

    return history


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, num_features, source = build_dataloaders(batch_size=32)
    model = RetailDemandMLP(
        in_features=num_features,
        hidden_dims=(64, 32),
        dropout=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    print("--- Day 88: Training with Held-Out Evaluation ---")
    print(f"Data source: {source} | Device: {device} | Features: {num_features}\n")

    history = fit_with_evaluation(
        model=model,
        train_loader=train_loader,
        eval_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=30,
        log_every=5,
        checkpoint_path="checkpoints/retail_demand_mlp_best.pt",
    )