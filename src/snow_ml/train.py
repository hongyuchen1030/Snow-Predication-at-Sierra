from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(32, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(32, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(x)
        skip2 = self.down2(self.pool1(skip1))
        bottleneck = self.bottleneck(self.pool2(skip2))

        up1 = self.up1(bottleneck)
        up1 = _match_spatial_shape(up1, skip2)
        dec1 = self.dec1(torch.cat([up1, skip2], dim=1))

        up2 = self.up2(dec1)
        up2 = _match_spatial_shape(up2, skip1)
        dec2 = self.dec2(torch.cat([up2, skip1], dim=1))
        return self.head(dec2)


class PreprocessedSnowDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]]
):
    def __init__(self, sample_paths: list[Path]) -> None:
        self.sample_paths = sample_paths

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        path = self.sample_paths[index]
        with np.load(path, allow_pickle=False) as data:
            inputs = torch.from_numpy(data["inputs"]).float()
            target = torch.from_numpy(data["target"]).float()
            valid_mask = torch.from_numpy(data["valid_mask"]).float()
            metadata = json.loads(str(data["metadata_json"].item()))
        return inputs, target, valid_mask, metadata


def build_data_loader(
    sample_paths: list[Path],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]]:
    dataset = PreprocessedSnowDataset(sample_paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def sample_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"].item()))


def select_sample_paths(
    sample_paths: list[Path],
    *,
    years: list[int] | None = None,
    target_month_day: str | None = None,
) -> list[Path]:
    selected: list[Path] = []
    allowed_years = {int(year) for year in years} if years is not None else None
    for path in sorted(sample_paths):
        metadata = sample_metadata(path)
        water_year = int(metadata["water_year"])
        sample_target_month_day = str(metadata["target_month_day"])
        if allowed_years is not None and water_year not in allowed_years:
            continue
        if target_month_day is not None and sample_target_month_day != target_month_day:
            continue
        selected.append(path)
    return selected


def train_model(
    *,
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]] | None,
    epochs: int,
    learning_rate: float,
    underpredict_weight: float,
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, float]]:
    if underpredict_weight <= 1.0:
        raise ValueError(f"underpredict_weight must be greater than 1.0, got {underpredict_weight}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for batch_index, (inputs, targets, valid_mask, _) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            valid_mask = valid_mask.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = masked_asymmetric_mse_loss(
                outputs,
                targets,
                valid_mask,
                underpredict_weight=underpredict_weight,
            )
            loss.backward()
            optimizer.step()

            train_loss_total += float(loss.item())
            train_batches += 1

            if epoch == 1 and batch_index == 1:
                print(f"train batch input shape: {tuple(inputs.shape)}")
                print(f"train batch target shape: {tuple(targets.shape)}")
                print(f"train batch valid-mask shape: {tuple(valid_mask.shape)}")
                print(f"train batch output shape: {tuple(outputs.shape)}")
                print(f"train underprediction weight: {underpredict_weight:.6f}")
                print(
                    f"train batch valid-mask fraction: "
                    f"{float(valid_mask.sum().item()) / max(float(valid_mask.numel()), 1.0):.6f}"
                )

        train_loss = train_loss_total / max(train_batches, 1)
        val_loss = evaluate_model(model, val_loader, device) if val_loader else float("nan")
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(row)
        print(f"epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / "last_checkpoint.pt")

    write_loss_history(history, output_dir / "loss_history.csv")
    return history


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]] | None,
    device: torch.device,
) -> float:
    if loader is None:
        return float("nan")

    model.eval()
    loss_total = 0.0
    batches = 0
    with torch.no_grad():
        for inputs, targets, valid_mask, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            valid_mask = valid_mask.to(device)
            outputs = model(inputs)
            loss = masked_mse_loss(outputs, targets, valid_mask)
            loss_total += float(loss.item())
            batches += 1
    return loss_total / max(batches, 1)


def run_one_step_from_raw() -> None:
    from snow_ml.features import build_training_sample, forecast_config_from_env

    config = forecast_config_from_env()
    inputs_np, targets_np, valid_mask_np, sierra_mask_np, metadata = build_training_sample(config)
    inputs = torch.from_numpy(inputs_np).unsqueeze(0)
    targets = torch.from_numpy(targets_np).unsqueeze(0)
    valid_mask = torch.from_numpy(valid_mask_np).unsqueeze(0)
    model = SmallUNet(in_channels=inputs.shape[1], out_channels=targets.shape[1])

    print(f"raw sample input shape: {tuple(inputs.shape)}")
    print(f"raw sample target shape: {tuple(targets.shape)}")
    print(f"raw sample valid-mask shape: {tuple(valid_mask.shape)}")
    print(f"raw sample sierra-mask shape: {tuple(torch.from_numpy(sierra_mask_np).unsqueeze(0).shape)}")
    with torch.no_grad():
        outputs = model(inputs)
    print(f"raw sample output shape: {tuple(outputs.shape)}")
    print(f"metadata target month-day: {metadata['target_month_day']}")
    print(f"metadata target date: {metadata['target_date_iso']}")


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    squared_error = (prediction - target) ** 2
    weighted = squared_error * valid_mask
    return weighted.sum() / valid_mask.sum().clamp_min(1.0)


def masked_asymmetric_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    underpredict_weight: float,
) -> torch.Tensor:
    if underpredict_weight <= 1.0:
        raise ValueError(f"underpredict_weight must be greater than 1.0, got {underpredict_weight}")

    error = prediction - target
    squared_error = error ** 2
    weighted_squared_error = torch.where(
        error < 0,
        underpredict_weight * squared_error,
        squared_error,
    )
    masked_weighted_error = weighted_squared_error * valid_mask
    return masked_weighted_error.sum() / valid_mask.sum().clamp_min(1.0)


def split_sample_paths(
    sample_paths: list[Path],
    *,
    validation_fraction: float,
) -> tuple[list[Path], list[Path]]:
    sorted_paths = sorted(sample_paths)
    if not sorted_paths:
        raise ValueError("No preprocessed sample files were provided.")
    validation_count = int(round(len(sorted_paths) * validation_fraction))
    validation_count = min(max(validation_count, 1), len(sorted_paths) - 1) if len(sorted_paths) > 1 else 0
    if validation_count == 0:
        return sorted_paths, []
    return sorted_paths[:-validation_count], sorted_paths[-validation_count:]


def save_split_description(
    *,
    train_paths: list[Path],
    val_paths: list[Path],
    output_path: Path,
) -> None:
    payload = {
        "train_files": [str(path) for path in train_paths],
        "validation_files": [str(path) for path in val_paths],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_loss_history(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _match_spatial_shape(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.shape[-2:] == reference.shape[-2:]:
        return values
    return torch.nn.functional.interpolate(
        values,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
