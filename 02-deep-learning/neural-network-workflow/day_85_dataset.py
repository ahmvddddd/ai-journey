from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


FEATURE_COLUMNS = [
    "lag_7_sales",
    "lag_14_sales",
    "unit_price_ngn",
    "promo_flag",
    "fuel_transport_cost_ngn",
    "is_holiday_season",
    "region_lagos",
    "region_kano",
    "region_abuja",
    "region_rivers",
    "category_groceries",
    "category_fmcg",
]
TARGET_COLUMN = "demand_units"


def generate_synthetic_retail_splits(
    n_train: int = 800, n_test: int = 200, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback generator matching the preprocessed retail demand schema."""
    rng = np.random.default_rng(seed)
    total = n_train + n_test

    lag_7 = rng.normal(150.0, 35.0, size=total).clip(20.0, None)
    lag_14 = rng.normal(145.0, 32.0, size=total).clip(20.0, None)
    unit_price = rng.normal(2400.0, 450.0, size=total).clip(500.0, None)
    promo = rng.binomial(1, 0.25, size=total).astype(float)
    fuel_cost = rng.normal(650.0, 140.0, size=total).clip(150.0, None)
    holiday = rng.binomial(1, 0.20, size=total).astype(float)

    regions = rng.choice(["lagos", "kano", "abuja", "rivers"], size=total, p=[0.35, 0.25, 0.20, 0.20])
    categories = rng.choice(["groceries", "fmcg"], size=total, p=[0.60, 0.40])

    # Standardize continuous features for neural network stability
    def zscore(arr: np.ndarray) -> np.ndarray:
        return (arr - arr.mean()) / (arr.std() + 1e-8)

    demand = (
        40.0
        + 0.45 * lag_7
        + 0.25 * lag_14
        - 0.015 * (unit_price - 2400.0)
        + 28.0 * promo
        - 0.02 * (fuel_cost - 650.0)
        + 35.0 * holiday
        + np.where(regions == "lagos", 22.0, 0.0)
        + np.where(regions == "kano", 12.0, 0.0)
        + rng.normal(0.0, 14.0, size=total)
    ).clip(10.0, None)

    df = pd.DataFrame({
        "lag_7_sales": zscore(lag_7),
        "lag_14_sales": zscore(lag_14),
        "unit_price_ngn": zscore(unit_price),
        "promo_flag": promo,
        "fuel_transport_cost_ngn": zscore(fuel_cost),
        "is_holiday_season": holiday,
        "region_lagos": (regions == "lagos").astype(float),
        "region_kano": (regions == "kano").astype(float),
        "region_abuja": (regions == "abuja").astype(float),
        "region_rivers": (regions == "rivers").astype(float),
        "category_groceries": (categories == "groceries").astype(float),
        "category_fmcg": (categories == "fmcg").astype(float),
        TARGET_COLUMN: np.round(demand, 2),
    })

    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def load_retail_dataframes(
    train_path: str = "retail_demand_train_processed.csv",
    test_path: str = "retail_demand_test_processed.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Loads processed CSV splits if present, otherwise uses synthetic fallback."""
    t_file = Path(train_path)
    v_file = Path(test_path)
    if t_file.exists() and v_file.exists():
        return pd.read_csv(t_file), pd.read_csv(v_file), "disk_csv"
    train_df, test_df = generate_synthetic_retail_splits()
    return train_df, test_df, "synthetic_fallback"


class RetailDemandDataset(Dataset):
    """PyTorch Dataset wrapping tabular features and continuous demand targets."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        if features.shape[0] != targets.shape[0]:
            raise ValueError("Features and targets must have the same number of rows.")
        if np.isnan(features).any() or np.isnan(targets).any():
            raise ValueError("Input features or targets contain NaN values.")

        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_dataloaders(
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, int, str]:
    train_df, test_df, source = load_retail_dataframes()

    X_train = train_df.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    X_test = test_df.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    train_ds = RetailDemandDataset(X_train, y_train)
    test_ds = RetailDemandDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, X_train.shape[1], source


if __name__ == "__main__":
    train_loader, test_loader, num_features, source = build_dataloaders(batch_size=32)
    batch_x, batch_y = next(iter(train_loader))

    print("--- Day 85: Retail Demand Dataset & DataLoaders ---")
    print(f"Data source:        {source}")
    print(f"Feature count:      {num_features}")
    print(f"Train dataset rows: {len(train_loader.dataset)} across {len(train_loader)} batches")
    print(f"Test dataset rows:  {len(test_loader.dataset)} across {len(test_loader)} batches")
    print(f"First batch X:      shape={tuple(batch_x.shape)}, dtype={batch_x.dtype}")
    print(f"First batch y:      shape={tuple(batch_y.shape)}, dtype={batch_y.dtype}")