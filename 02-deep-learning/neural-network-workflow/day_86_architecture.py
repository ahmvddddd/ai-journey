import torch
import torch.nn as nn

from day_85_dataset import build_dataloaders


class RetailDemandMLP(nn.Module):
    """Multi-Layer Perceptron for tabular retail demand regression."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer width.")

        layers: list[nn.Module] = []
        prev_dim = in_features

        for width in hidden_dims:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = width

        # Final linear head predicts continuous demand_units with no output activation
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, num_features, source = build_dataloaders(batch_size=32)

    model = RetailDemandMLP(
        in_features=num_features,
        hidden_dims=(64, 32),
        dropout=0.1,
    ).to(device)

    batch_x, batch_y = next(iter(train_loader))
    batch_x = batch_x.to(device)

    with torch.no_grad():
        preds = model(batch_x)

    total_params, trainable_params = count_parameters(model)

    print("--- Day 86: RetailDemandMLP Architecture ---")
    print(f"Data source:       {source}")
    print(f"Device:            {device}")
    print(f"Input features:    {num_features}")
    print(f"Total params:      {total_params:,} (trainable: {trainable_params:,})")
    print(f"Batch X shape:     {tuple(batch_x.shape)}")
    print(f"Predictions shape: {tuple(preds.shape)}")
    print(f"Target y shape:    {tuple(batch_y.shape)}\n")
    print("Layer breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name:28s} -> {tuple(param.shape)}")