import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from day_85_dataset import build_dataloaders
from day_86_architecture import RetailDemandMLP
from day_87_training import set_seed


def check_tensor_health(tensor: torch.Tensor, name: str) -> bool:
    """Checks a tensor for NaN or Inf values and prints shape/range diagnostics."""
    has_nan = bool(torch.isnan(tensor).any().item())
    has_inf = bool(torch.isinf(tensor).any().item())
    ok = not (has_nan or has_inf)

    t_min = tensor.min().item() if tensor.numel() > 0 else float("nan")
    t_max = tensor.max().item() if tensor.numel() > 0 else float("nan")

    print(
        f"[Health] {name:18s} | shape={str(tuple(tensor.shape)):12s} | "
        f"dtype={str(tensor.dtype):14s} | min={t_min:8.2f} | max={t_max:8.2f} | "
        f"nan={has_nan} | inf={has_inf}"
    )
    return ok


def compute_total_grad_norm(model: nn.Module) -> float:
    """Computes the global L2 norm across all parameter gradients."""
    sq_sum = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.detach().data.norm(2).item()
            sq_sum += grad_norm ** 2
    return float(np.sqrt(sq_sum))


def run_single_batch_overfit_test(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    steps: int = 200,
    lr: float = 1e-2,
) -> dict[str, float]:
    """Verifies that the model and loss can memorize a single mini-batch."""
    model.train()
    batch_x, batch_y = next(iter(dataloader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    check_tensor_health(batch_x, "single_batch_x")
    check_tensor_health(batch_y, "single_batch_y")

    # Fresh optimizer isolated to this sanity check
    probe_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    initial_loss = None
    final_loss = None
    final_grad_norm = 0.0

    for step in range(1, steps + 1):
        probe_optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"Non-finite loss detected at step {step}: {loss.item()}")

        loss.backward()
        final_grad_norm = compute_total_grad_norm(model)
        probe_optimizer.step()

        loss_val = float(loss.item())
        if initial_loss is None:
            initial_loss = loss_val
        final_loss = loss_val

    return {
        "initial_mse": initial_loss if initial_loss is not None else float("nan"),
        "final_mse": final_loss if final_loss is not None else float("nan"),
        "final_rmse": float(np.sqrt(final_loss)) if final_loss is not None else float("nan"),
        "final_grad_norm": final_grad_norm,
    }


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, num_features, source = build_dataloaders(batch_size=16)
    model = RetailDemandMLP(
        in_features=num_features,
        hidden_dims=(64, 32),
        dropout=0.0,  # Keep dropout off during a single-batch memorization check
    ).to(device)

    criterion = nn.MSELoss()

    print("--- Day 89: Neural Network Debugging & Sanity Checks ---")
    print(f"Data source: {source} | Device: {device} | Features: {num_features}\n")

    stats = run_single_batch_overfit_test(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        device=device,
        steps=250,
        lr=2e-2,
    )

    print("\n--- Single-Batch Overfit Summary ---")
    print(f"Initial MSE:      {stats['initial_mse']:10.2f}")
    print(f"Final MSE:        {stats['final_mse']:10.4f}")
    print(f"Final RMSE:       {stats['final_rmse']:8.4f} units")
    print(f"Final Grad Norm:  {stats['final_grad_norm']:8.4f}")