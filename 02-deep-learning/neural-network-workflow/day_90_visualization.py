from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from day_85_dataset import build_dataloaders
from day_86_architecture import RetailDemandMLP
from day_87_training import set_seed
from day_88_validation import fit_with_evaluation


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs inference on a DataLoader and returns (y_true, y_pred) as 1D arrays."""
    model.eval()
    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_preds = model(batch_x).squeeze(-1).cpu().numpy()
            batch_targets = batch_y.squeeze(-1).cpu().numpy()

            preds_list.append(batch_preds)
            targets_list.append(batch_targets)

    return np.concatenate(targets_list), np.concatenate(preds_list)


def plot_learning_curves(
    history: list[dict[str, float]],
    output_path: str = "artifacts/day_90_learning_curves.png",
) -> Path:
    """Plots training and validation RMSE across epochs."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]
    train_rmse = [row["train_rmse"] for row in history]
    val_rmse = [row["val_rmse"] for row in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_rmse, marker="o", markersize=4, label="Train RMSE")
    ax.plot(epochs, val_rmse, marker="s", markersize=4, label="Validation RMSE")
    ax.set_title("Training vs. Validation RMSE Across Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (demand_units)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def plot_prediction_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "artifacts/day_90_diagnostics.png",
) -> Path:
    """Creates side-by-side predicted-vs-actual and residual diagnostic plots."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Left panel: Predicted vs. Actual with ideal diagonal line
    axes[0].scatter(y_true, y_pred, alpha=0.65, edgecolors="none", s=28)
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal (y_pred = y_true)")
    axes[0].set_title("Predicted vs. Actual Demand")
    axes[0].set_xlabel("Actual demand_units")
    axes[0].set_ylabel("Predicted demand_units")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # Right panel: Residual histogram centered at zero error
    axes[1].hist(residuals, bins=25, edgecolor="black", alpha=0.75)
    axes[1].axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Zero Residual")
    axes[1].axvline(float(np.mean(residuals)), color="orange", linestyle="-", linewidth=1.5, label=f"Mean ({np.mean(residuals):.2f})")
    axes[1].set_title("Residual Distribution (Actual - Predicted)")
    axes[1].set_xlabel("Residual (demand_units)")
    axes[1].set_ylabel("Sample Count")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


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

    history = fit_with_evaluation(
        model=model,
        train_loader=train_loader,
        eval_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=30,
        log_every=10,
    )

    y_true, y_pred = collect_predictions(model=model, dataloader=test_loader, device=device)
    curves_path = plot_learning_curves(history)
    diag_path = plot_prediction_diagnostics(y_true, y_pred)

    print("\n--- Day 90: Diagnostic Plots Saved ---")
    print(f"Data source:       {source}")
    print(f"Learning curves:   {curves_path}")
    print(f"Error diagnostics: {diag_path}")
    print(f"Residual mean:     {np.mean(y_true - y_pred):.3f} units")
    print(f"Residual std:      {np.std(y_true - y_pred):.3f} units")