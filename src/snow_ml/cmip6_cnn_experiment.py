from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
import warnings


MONTH_LABELS = ("Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar")
LATENT_K = 8
LATENT_D = 16
FEATURE_Z_CLIP = 20.0


@dataclass(frozen=True)
class ExperimentConfig:
    architecture: str
    seed: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    amp: bool
    predictor_cache_dir: str
    output_dir: str
    split_json_path: str | None = None
    split_manifest_path: str | None = None
    historical_batch_order_audit_path: str | None = None


def architecture_uses_auxiliary(architecture: str) -> bool:
    return architecture in {
        "M1_static_cnn",
        "D1_static_cnn_cpm_aqm_diag",
        "D2_static_cnn_cpm_only",
        "K1_static_cnn_kendall_aux",
        "P1_static_cnn_pcgrad",
        "P2_static_cnn_gradsim",
        "U1_stage2_cpm_aqm",
        "M2_temporal_conv",
        "M3_convgru",
        "M4_attention_swe",
        "M6_static_attention_with_aux",
    }


def architecture_uses_swe_only_loss(architecture: str) -> bool:
    return architecture in {
        "M5_static_attention_swe_only",
        "M7_static_cnn_swe_only",
        "S0_static_cnn_swe_only",
        "S1_static_latent_self_attention",
        "S2_static_swe_token",
        "S3_static_residual_gated_attention",
    }


def architecture_uses_eval_mode_train_metrics(architecture: str) -> bool:
    """Use end-of-epoch eval predictions for the S0-S3 comparison metrics."""
    return architecture in {
        "S0_static_cnn_swe_only",
        "S1_static_latent_self_attention",
        "S2_static_swe_token",
        "S3_static_residual_gated_attention",
    }


class HistoricalBatchOrderSampler(Sampler[list[int]]):
    """Replay precomputed dataset-position batches for a selected epoch."""

    def __init__(self, batches_by_epoch: list[list[list[int]]]) -> None:
        self.batches_by_epoch = batches_by_epoch
        self.epoch_index = 0

    def set_epoch(self, epoch_index: int) -> None:
        if epoch_index < 1 or epoch_index > len(self.batches_by_epoch):
            raise ValueError(f"Historical batch order is unavailable for epoch {epoch_index}")
        self.epoch_index = epoch_index - 1

    def __iter__(self):
        if self.epoch_index < 0 or self.epoch_index >= len(self.batches_by_epoch):
            raise RuntimeError("Call set_epoch() before iterating the historical batch sampler")
        yield from self.batches_by_epoch[self.epoch_index]

    def __len__(self) -> int:
        if self.epoch_index < 0 or self.epoch_index >= len(self.batches_by_epoch):
            return 0
        return len(self.batches_by_epoch[self.epoch_index])


def architecture_active_tasks(architecture: str) -> tuple[str, ...]:
    if architecture in {"D2_static_cnn_cpm_only"}:
        return ("swe", "cpm")
    if architecture in {
        "M1_static_cnn",
        "D1_static_cnn_cpm_aqm_diag",
        "K1_static_cnn_kendall_aux",
        "P1_static_cnn_pcgrad",
        "P2_static_cnn_gradsim",
        "U1_stage2_cpm_aqm",
        "M2_temporal_conv",
        "M3_convgru",
        "M4_attention_swe",
        "M6_static_attention_with_aux",
    }:
        return ("swe", "cpm", "aqm")
    return ("swe",)


def architecture_records_gradient_diagnostics(architecture: str) -> bool:
    return architecture in {
        "D1_static_cnn_cpm_aqm_diag",
        "D2_static_cnn_cpm_only",
        "K1_static_cnn_kendall_aux",
        "P1_static_cnn_pcgrad",
        "P2_static_cnn_gradsim",
        "U1_stage2_cpm_aqm",
    }


def architecture_uses_kendall_weighting(architecture: str) -> bool:
    return architecture in {"K1_static_cnn_kendall_aux"}


def architecture_uses_pcgrad(architecture: str) -> bool:
    return architecture in {"P1_static_cnn_pcgrad"}


def architecture_uses_gradsim(architecture: str) -> bool:
    return architecture in {"P2_static_cnn_gradsim"}


def architecture_uses_upstream_aux(architecture: str) -> bool:
    return architecture in {"U1_stage2_cpm_aqm"}


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{timestamp}] {message}", flush=True)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def _batch_order_digest(batches: list[list[int]]) -> str:
    payload = json.dumps(batches, separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build_historical_batch_orders(
    *,
    architecture: str,
    audit_path: Path,
    train_source_indices: list[int],
    val_source_indices: list[int],
    batch_size: int,
    max_epochs: int,
    rng_state_after_model_init: torch.Tensor,
) -> tuple[list[list[list[int]]], dict[str, Any]]:
    """Generate and verify an architecture's original shuffle schedule."""
    audit = load_json(audit_path)
    recorded = audit["architectures"][architecture]["original"]
    if int(audit["seed"]) != 20260813:
        raise ValueError(f"Unexpected historical-replay audit seed: {audit['seed']}")
    if int(audit["batch_size"]) != batch_size:
        raise ValueError(f"Audit batch size {audit['batch_size']} does not match config batch size {batch_size}")
    if int(audit["train_row_count"]) != len(train_source_indices):
        raise ValueError("Audit train-row count does not match the current split")
    if int(audit["val_row_count"]) != len(val_source_indices):
        raise ValueError("Audit validation-row count does not match the current split")

    source_to_dataset_position = {source_index: position for position, source_index in enumerate(train_source_indices)}
    if len(source_to_dataset_position) != len(train_source_indices):
        raise ValueError("Current train split contains duplicate source indices")

    # RandomSampler(generator=None) first draws a seed from the global CPU RNG,
    # then uses a local generator. Preserve that exact historical path in a fork.
    position_batches_by_epoch: list[list[list[int]]] = []
    source_batches_by_epoch: list[list[list[int]]] = []
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng_state_after_model_init)
        schedule_train_loader = DataLoader(
            list(range(len(train_source_indices))),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        schedule_val_loader = DataLoader(
            list(range(len(val_source_indices))),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for _ in range(max_epochs):
            position_batches = [list(map(int, batch.tolist())) for batch in schedule_train_loader]
            position_batches_by_epoch.append(position_batches)
            source_batches_by_epoch.append(
                [[train_source_indices[position] for position in batch] for batch in position_batches]
            )
            # The original loop iterated validation after each optimization epoch.
            for _ in schedule_val_loader:
                pass

    verification_epochs = min(len(recorded), max_epochs)
    recorded_digests = [_batch_order_digest(recorded[index]) for index in range(verification_epochs)]
    generated_digests = [_batch_order_digest(source_batches_by_epoch[index]) for index in range(verification_epochs)]
    if recorded_digests != generated_digests:
        raise RuntimeError(
            "Generated batches do not match the recorded original audit for epochs "
            f"1-{verification_epochs}: recorded={recorded_digests} generated={generated_digests}"
        )

    # Confirm every audited manifest index maps to the dataset position used for replay.
    for epoch_batches in recorded:
        for batch in epoch_batches:
            if any(source_index not in source_to_dataset_position for source_index in batch):
                raise ValueError("Audit includes a source index absent from the current train split")
    return position_batches_by_epoch, {
        "architecture": architecture,
        "audit_path": str(audit_path),
        "recorded_epoch_count": len(recorded),
        "verified_epoch_count": verification_epochs,
        "recorded_source_batch_digests": recorded_digests,
        "generated_source_batch_digests": generated_digests,
        "full_schedule_epoch_count": len(position_batches_by_epoch),
    }


def manifest_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["model_member"]),
        str(row["experiment"]),
        str(row["row_year"]),
        str(row["water_year"]),
    )


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()
    denom = math.sqrt(float((y_true_centered**2).sum() * (y_pred_centered**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((y_true_centered * y_pred_centered).sum() / denom)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1
    return ranks


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return pearson_r(rankdata_average_ties(y_true), rankdata_average_ties(y_pred))


def wet_dry_accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true >= threshold) == (y_pred >= threshold)))


def build_grouped_split(
    manifest_rows: list[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float = 0.8,
) -> dict[str, list[int]]:
    unique_water_years = sorted({int(row["water_year"]) for row in manifest_rows})
    rng = np.random.default_rng(seed)
    shuffled = list(unique_water_years)
    rng.shuffle(shuffled)
    train_group_count = max(1, int(round(train_fraction * len(shuffled))))
    train_years = set(shuffled[:train_group_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for index, row in enumerate(manifest_rows):
        if int(row["water_year"]) in train_years:
            train_indices.append(index)
        else:
            val_indices.append(index)
    return {"train": train_indices, "val": val_indices}


def load_or_build_split(
    manifest_rows: list[dict[str, Any]],
    *,
    seed: int,
    split_json_path: str | None,
    split_manifest_path: str | None,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    if split_json_path is None:
        split = build_grouped_split(manifest_rows, seed=seed)
        return (
            split,
            {
                "split_source": "generated_from_seed",
                "seed": seed,
                "train_water_years": sorted({int(manifest_rows[index]["water_year"]) for index in split["train"]}),
                "val_water_years": sorted({int(manifest_rows[index]["water_year"]) for index in split["val"]}),
            },
        )

    split = load_json(Path(split_json_path))
    if not isinstance(split, dict) or "train" not in split or "val" not in split:
        raise ValueError(f"Invalid split payload at {split_json_path}")

    n_rows = len(manifest_rows)
    raw_split: dict[str, list[int]] = {}
    for key in ("train", "val"):
        values = split[key]
        if not isinstance(values, list):
            raise ValueError(f"Split key {key} must contain a list of indices")
        normalized_values = [int(value) for value in values]
        if len(set(normalized_values)) != len(normalized_values):
            raise ValueError(f"Split key {key} contains duplicate indices")
        raw_split[key] = normalized_values

    if split_manifest_path is not None:
        reference_manifest = read_manifest(Path(split_manifest_path))
        current_index_by_identity = {manifest_identity(row): index for index, row in enumerate(manifest_rows)}
        reference_identities = [manifest_identity(row) for row in reference_manifest]
        missing_reference_ids = [identity for identity in reference_identities if identity not in current_index_by_identity]
        if missing_reference_ids:
            raise ValueError(
                "Current manifest is missing rows from the reference manifest. "
                f"First missing identities: {missing_reference_ids[:10]}"
            )

        normalized_split = {"train": [], "val": []}
        train_water_years: set[int] = set()
        val_water_years: set[int] = set()
        for key in ("train", "val"):
            target_list = normalized_split[key]
            for ref_index in raw_split[key]:
                if ref_index < 0 or ref_index >= len(reference_manifest):
                    raise ValueError(
                        f"Split key {key} contains out-of-range reference index {ref_index} "
                        f"for manifest size {len(reference_manifest)}"
                    )
                identity = reference_identities[ref_index]
                current_index = current_index_by_identity[identity]
                target_list.append(current_index)
                water_year = int(manifest_rows[current_index]["water_year"])
                if key == "train":
                    train_water_years.add(water_year)
                else:
                    val_water_years.add(water_year)

        extra_rows = []
        assigned_indices = set(normalized_split["train"]) | set(normalized_split["val"])
        for current_index, row in enumerate(manifest_rows):
            if current_index in assigned_indices:
                continue
            water_year = int(row["water_year"])
            if water_year in train_water_years and water_year in val_water_years:
                raise ValueError(f"Water year {water_year} appears in both train and val reference partitions")
            if water_year in train_water_years:
                normalized_split["train"].append(current_index)
                extra_rows.append({"index": current_index, "identity": manifest_identity(row), "assigned_to": "train"})
            elif water_year in val_water_years:
                normalized_split["val"].append(current_index)
                extra_rows.append({"index": current_index, "identity": manifest_identity(row), "assigned_to": "val"})
            else:
                raise ValueError(
                    "Encountered new manifest row whose water year is absent from the reference split: "
                    f"{manifest_identity(row)}"
                )

        verification = {
            "split_source": str(split_json_path),
            "reference_manifest": str(split_manifest_path),
            "reference_manifest_rows": len(reference_manifest),
            "current_manifest_rows": n_rows,
            "split_mapping_mode": "reference_identity_with_water_year_assignment_for_extra_rows",
            "extra_rows_assigned_by_water_year": extra_rows,
            "manifest_identity_match": len(extra_rows) == 0 and len(reference_manifest) == n_rows,
            "train_count": len(normalized_split["train"]),
            "val_count": len(normalized_split["val"]),
            "train_water_years": sorted({int(manifest_rows[index]["water_year"]) for index in normalized_split["train"]}),
            "val_water_years": sorted({int(manifest_rows[index]["water_year"]) for index in normalized_split["val"]}),
        }
    else:
        normalized_split = {}
        for key in ("train", "val"):
            normalized_values = raw_split[key]
            if any(index < 0 or index >= n_rows for index in normalized_values):
                raise ValueError(f"Split key {key} contains out-of-range indices for manifest size {n_rows}")
            normalized_split[key] = normalized_values
        verification = {
            "split_source": str(split_json_path),
            "train_count": len(normalized_split["train"]),
            "val_count": len(normalized_split["val"]),
            "train_water_years": sorted({int(manifest_rows[index]["water_year"]) for index in normalized_split["train"]}),
            "val_water_years": sorted({int(manifest_rows[index]["water_year"]) for index in normalized_split["val"]}),
        }

    overlap = set(normalized_split["train"]) & set(normalized_split["val"])
    if overlap:
        raise ValueError(f"Split contains overlapping train/val indices: {sorted(overlap)[:10]}")
    return normalized_split, verification


def compute_target_stats(targets: np.ndarray, train_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    train_targets = targets[np.asarray(train_indices, dtype=np.int64)]
    mu = train_targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    sigma = train_targets.std(axis=0, dtype=np.float64).astype(np.float32)
    sigma = np.where(sigma < 1e-6, 1.0, sigma).astype(np.float32)
    return mu, sigma


def compute_feature_stats(
    physical_data: np.memmap,
    train_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_data = np.asarray(physical_data[np.asarray(train_indices, dtype=np.int64)], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", category=RuntimeWarning)
        mu = np.nanmean(train_data, axis=0, dtype=np.float64).astype(np.float32)
        sigma = np.nanstd(train_data, axis=0, dtype=np.float64).astype(np.float32)
    valid_count = np.isfinite(train_data).sum(axis=0, dtype=np.int32)
    all_nan_mask = valid_count == 0
    zero_variance_mask = np.isfinite(sigma) & (sigma < 1e-6)
    mu = np.where(all_nan_mask, 0.0, mu).astype(np.float32)
    sigma = np.where(all_nan_mask, 1.0, sigma).astype(np.float32)
    sigma = np.where(sigma < 1e-6, 1.0, sigma).astype(np.float32)
    if caught_warnings:
        _log(
            "NORMALIZATION WARNING SOURCE compute_feature_stats: "
            f"{len(caught_warnings)} runtime warnings while computing train-only feature statistics"
        )
    _log(
        "NORMALIZATION COLUMN SUMMARY "
        f"all_nan_columns={int(all_nan_mask.sum())} "
        f"zero_variance_columns={int(zero_variance_mask.sum())}"
    )
    return mu, sigma, valid_count


class CMIP6TensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    def __init__(
        self,
        *,
        physical_data: np.memmap,
        valid_mask: np.memmap,
        targets: np.ndarray,
        metadata: list[dict[str, Any]],
        indices: list[int],
        feature_mu: np.ndarray,
        feature_sigma: np.ndarray,
        target_mu: np.ndarray,
        target_sigma: np.ndarray,
    ) -> None:
        self.physical_data = physical_data
        self.valid_mask = valid_mask
        self.targets = targets
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=np.int64)
        self.feature_mu = feature_mu
        self.feature_sigma = feature_sigma
        self.target_mu = target_mu
        self.target_sigma = target_sigma

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        source_index = int(self.indices[item])
        x = np.asarray(self.physical_data[source_index], dtype=np.float32)
        mask = np.asarray(self.valid_mask[source_index], dtype=np.float32)
        x = (x - self.feature_mu) / self.feature_sigma
        x = np.clip(x, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
        x = np.where(mask > 0.5, x, np.nan).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        stacked = np.concatenate([x, mask], axis=1)
        target = ((self.targets[source_index] - self.target_mu) / self.target_sigma).astype(np.float32)
        return torch.from_numpy(stacked), torch.from_numpy(target), self.metadata[source_index]


def collate_with_metadata(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    inputs = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    metadata = [item[2] for item in batch]
    return inputs, targets, metadata


def periodic_pad_2d(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_w > 0:
        x = torch.cat([x[..., -pad_w:], x, x[..., :pad_w]], dim=-1)
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="replicate")
    return x


class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.pad_h = padding
        self.pad_w = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(periodic_pad_2d(x, self.pad_h, self.pad_w))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels >= 8 else 1
        self.conv1 = PeriodicConv2d(channels, channels, kernel_size=3)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = PeriodicConv2d(channels, channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        return self.act(out)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 8 if out_channels >= 8 else 1
        self.conv = PeriodicConv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SpatialBackbone(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            PeriodicConv2d(in_channels, 32, kernel_size=5, stride=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(ResidualBlock(32), ResidualBlock(32))
        self.down1 = DownsampleBlock(32, 64)
        self.stage2 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down2 = DownsampleBlock(64, 128)
        self.stage3 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_feature_dict(x)
        return features["stage3"]

    def forward_feature_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        stage1 = self.stage1(x)
        down1 = self.down1(stage1)
        stage2 = self.stage2(down1)
        down2 = self.down2(stage2)
        stage3 = self.stage3(down2)
        return {
            "stem": x,
            "stage1": stage1,
            "down1": down1,
            "stage2": stage2,
            "down2": down2,
            "stage3": stage3,
        }


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gates = PeriodicConv2d(channels * 2, channels * 2, kernel_size=3)
        self.candidate = PeriodicConv2d(channels * 2, channels, kernel_size=3)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        gate_inputs = torch.cat([x, h_prev], dim=1)
        z_gate, r_gate = torch.chunk(torch.sigmoid(self.gates(gate_inputs)), 2, dim=1)
        candidate_inputs = torch.cat([x, r_gate * h_prev], dim=1)
        h_tilde = torch.tanh(self.candidate(candidate_inputs))
        return (1.0 - z_gate) * h_prev + z_gate * h_tilde


class AttentionSWEHead(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.score_hidden = nn.Linear(latent_dim, latent_dim)
        self.score_out = nn.Linear(latent_dim, 1)
        self.mlp = nn.Sequential(nn.Linear(latent_dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scores = self.score_out(torch.tanh(self.score_hidden(z))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        return self.mlp(pooled).squeeze(-1)


class LatentSelfAttentionBlock(nn.Module):
    def __init__(self, latent_dim: int, *, num_heads: int = 2, ff_width: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ff_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_width, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class AuxiliaryHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cpm = nn.Linear(LATENT_K * LATENT_D, 1)
        self.aqm = nn.Linear(LATENT_K * LATENT_D, 1)

    def forward(self, z_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cpm(z_flat).squeeze(-1), self.aqm(z_flat).squeeze(-1)


class BaseArchitecture(nn.Module):
    def __init__(self, *, include_auxiliary_heads: bool = True) -> None:
        super().__init__()
        self.include_auxiliary_heads = include_auxiliary_heads
        self.aux_heads = AuxiliaryHeads() if include_auxiliary_heads else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        z_flat = z.reshape(z.shape[0], -1)
        outputs = {
            "z": z,
            "swe_hat": self.predict_swe(z, z_flat),
        }
        if self.aux_heads is not None:
            cpm_hat, aqm_hat = self.aux_heads(z_flat)
            outputs["cpm_hat"] = cpm_hat
            outputs["aqm_hat"] = aqm_hat
        return outputs

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_swe(self, z: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_history_metrics(self) -> dict[str, float]:
        return {}

    def current_task_weights(self) -> dict[str, float]:
        return {}

    def named_encoder_parameters(self) -> list[tuple[str, nn.Parameter]]:
        named_params: list[tuple[str, nn.Parameter]] = []
        for module_name in ("backbone", "project", "temporal", "conv_gru"):
            module = getattr(self, module_name, None)
            if module is not None:
                named_params.extend(
                    (f"{module_name}.{name}", parameter)
                    for name, parameter in module.named_parameters()
                    if parameter.requires_grad
                )
        return named_params

    def named_swe_head_parameters(self) -> list[tuple[str, nn.Parameter]]:
        swe_head = getattr(self, "swe_head", None)
        if swe_head is None:
            return []
        return [
            (f"swe_head.{name}", parameter)
            for name, parameter in swe_head.named_parameters()
            if parameter.requires_grad
        ]

    def named_cpm_head_parameters(self) -> list[tuple[str, nn.Parameter]]:
        if self.aux_heads is None or not hasattr(self.aux_heads, "cpm"):
            return []
        return [
            (f"aux_heads.cpm.{name}", parameter)
            for name, parameter in self.aux_heads.cpm.named_parameters()
            if parameter.requires_grad
        ]

    def named_aqm_head_parameters(self) -> list[tuple[str, nn.Parameter]]:
        if self.aux_heads is None or not hasattr(self.aux_heads, "aqm"):
            return []
        return [
            (f"aux_heads.aqm.{name}", parameter)
            for name, parameter in self.aux_heads.aqm.named_parameters()
            if parameter.requires_grad
        ]

    def shared_encoder_parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.named_encoder_parameters()]

    def named_early_encoder_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return self.named_encoder_parameters()

    def named_late_encoder_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return []


class M1StaticCNN(BaseArchitecture):
    def __init__(self, in_channels_per_month: int, *, include_auxiliary_heads: bool = True) -> None:
        super().__init__(include_auxiliary_heads=include_auxiliary_heads)
        self.backbone = SpatialBackbone(in_channels_per_month * len(MONTH_LABELS))
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D),
        )
        self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        x = x.reshape(batch_size, months * channels, height, width)
        features = self.backbone(x)
        latent = self.project(features)
        return latent.reshape(batch_size, LATENT_K, LATENT_D)

    def predict_swe(self, z: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        return self.swe_head(z_flat).squeeze(-1)


class StaticAttentionCNN(BaseArchitecture):
    def __init__(self, in_channels_per_month: int, *, include_auxiliary_heads: bool) -> None:
        super().__init__(include_auxiliary_heads=include_auxiliary_heads)
        self.backbone = SpatialBackbone(in_channels_per_month * len(MONTH_LABELS))
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D),
        )
        self.swe_head = AttentionSWEHead(LATENT_D)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        x = x.reshape(batch_size, months * channels, height, width)
        features = self.backbone(x)
        latent = self.project(features)
        return latent.reshape(batch_size, LATENT_K, LATENT_D)

    def predict_swe(self, z: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        return self.swe_head(z)


class StaticLatentSelfAttentionCNN(BaseArchitecture):
    def __init__(self, in_channels_per_month: int) -> None:
        super().__init__(include_auxiliary_heads=False)
        self.backbone = SpatialBackbone(in_channels_per_month * len(MONTH_LABELS))
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D),
        )
        self.latent_block = LatentSelfAttentionBlock(LATENT_D, num_heads=2, ff_width=32, dropout=0.1)
        self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        x = x.reshape(batch_size, months * channels, height, width)
        features = self.backbone(x)
        latent = self.project(features).reshape(batch_size, LATENT_K, LATENT_D)
        return self.latent_block(latent)

    def predict_swe(self, z: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        return self.swe_head(z_flat).squeeze(-1)


class StaticSWETokenAttentionCNN(BaseArchitecture):
    def __init__(self, in_channels_per_month: int) -> None:
        super().__init__(include_auxiliary_heads=False)
        self.backbone = SpatialBackbone(in_channels_per_month * len(MONTH_LABELS))
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D),
        )
        self.swe_token = nn.Parameter(torch.zeros(1, 1, LATENT_D))
        nn.init.normal_(self.swe_token, mean=0.0, std=0.02)
        self.latent_block = LatentSelfAttentionBlock(LATENT_D, num_heads=2, ff_width=32, dropout=0.1)
        self.swe_head = nn.Sequential(nn.Linear(LATENT_D, 32), nn.GELU(), nn.Linear(32, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        x = x.reshape(batch_size, months * channels, height, width)
        features = self.backbone(x)
        latent = self.project(features).reshape(batch_size, LATENT_K, LATENT_D)
        swe_token = self.swe_token.expand(batch_size, -1, -1)
        tokenized = torch.cat([swe_token, latent], dim=1)
        return self.latent_block(tokenized)

    def predict_swe(self, z: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        swe_token = z[:, 0, :]
        return self.swe_head(swe_token).squeeze(-1)


class StaticResidualGatedAttentionCNN(BaseArchitecture):
    def __init__(self, in_channels_per_month: int) -> None:
        super().__init__(include_auxiliary_heads=False)
        self.backbone = SpatialBackbone(in_channels_per_month * len(MONTH_LABELS))
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D),
        )
        self.base_swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))
        self.latent_block = LatentSelfAttentionBlock(LATENT_D, num_heads=2, ff_width=32, dropout=0.1)
        self.attn_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 32), nn.GELU(), nn.Linear(32, 1))
        self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        x = x.reshape(batch_size, months * channels, height, width)
        features = self.backbone(x)
        latent = self.project(features)
        return latent.reshape(batch_size, LATENT_K, LATENT_D)

    def predict_swe(self, z: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        y_base = self.base_swe_head(z_flat).squeeze(-1)
        z_attn = self.latent_block(z)
        y_attn = self.attn_head(z_attn.reshape(z_attn.shape[0], -1)).squeeze(-1)
        return y_base + (self.alpha * y_attn)

    def extra_history_metrics(self) -> dict[str, float]:
        return {"alpha": float(self.alpha.detach().cpu().item())}


class StaticCNNCPMOnly(BaseArchitecture):
    def __init__(self, in_channels_per_month: int) -> None:
        super().__init__(include_auxiliary_heads=False)
        self.backbone = SpatialBackbone(in_channels_per_month * len(MONTH_LABELS))
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D),
        )
        self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))
        self.cpm_head = nn.Linear(LATENT_K * LATENT_D, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        x = x.reshape(batch_size, months * channels, height, width)
        features = self.backbone(x)
        latent = self.project(features)
        return latent.reshape(batch_size, LATENT_K, LATENT_D)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        z_flat = z.reshape(z.shape[0], -1)
        return {
            "z": z,
            "swe_hat": self.predict_swe(z, z_flat),
            "cpm_hat": self.cpm_head(z_flat).squeeze(-1),
        }

    def predict_swe(self, z: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        return self.swe_head(z_flat).squeeze(-1)


class StaticCNNKendallAux(M1StaticCNN):
    def __init__(self, in_channels_per_month: int) -> None:
        super().__init__(in_channels_per_month=in_channels_per_month, include_auxiliary_heads=True)
        self.log_var_swe = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.log_var_cpm = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.log_var_aqm = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def current_task_weights(self) -> dict[str, float]:
        s_swe = float(self.log_var_swe.detach().cpu().item())
        s_cpm = float(self.log_var_cpm.detach().cpu().item())
        s_aqm = float(self.log_var_aqm.detach().cpu().item())
        w_swe = 0.5 * math.exp(-s_swe)
        w_cpm = 0.5 * math.exp(-s_cpm)
        w_aqm = 0.5 * math.exp(-s_aqm)
        total = w_swe + w_cpm + w_aqm
        return {
            "s_swe": s_swe,
            "s_cpm": s_cpm,
            "s_aqm": s_aqm,
            "w_swe": w_swe,
            "w_cpm": w_cpm,
            "w_aqm": w_aqm,
            "w_rel_swe": w_swe / total,
            "w_rel_cpm": w_cpm / total,
            "w_rel_aqm": w_aqm / total,
        }

    def extra_history_metrics(self) -> dict[str, float]:
        return self.current_task_weights()


class UpstreamStage2AuxCNN(BaseArchitecture):
    def __init__(self, in_channels_per_month: int) -> None:
        super().__init__(include_auxiliary_heads=False)
        self.backbone = SpatialBackbone(in_channels_per_month * len(MONTH_LABELS))
        self.stage2_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cpm_stage2_head = nn.Linear(64, 1)
        self.aqm_stage2_head = nn.Linear(64, 1)
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D),
        )
        self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))

    def _forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, months, channels, height, width = x.shape
        x = x.reshape(batch_size, months * channels, height, width)
        feature_dict = self.backbone.forward_feature_dict(x)
        stage2 = feature_dict["stage2"]
        stage3 = feature_dict["stage3"]
        stage2_pooled = self.stage2_pool(stage2).reshape(batch_size, -1)
        return stage2_pooled, stage3

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, stage3 = self._forward_features(x)
        latent = self.project(stage3)
        return latent.reshape(latent.shape[0], LATENT_K, LATENT_D)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stage2_pooled, stage3 = self._forward_features(x)
        latent = self.project(stage3).reshape(x.shape[0], LATENT_K, LATENT_D)
        z_flat = latent.reshape(latent.shape[0], -1)
        return {
            "z": latent,
            "stage2_pooled": stage2_pooled,
            "swe_hat": self.predict_swe(latent, z_flat),
            "cpm_hat": self.cpm_stage2_head(stage2_pooled).squeeze(-1),
            "aqm_hat": self.aqm_stage2_head(stage2_pooled).squeeze(-1),
        }

    def predict_swe(self, z: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        return self.swe_head(z_flat).squeeze(-1)

    def named_cpm_head_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (f"cpm_stage2_head.{name}", parameter)
            for name, parameter in self.cpm_stage2_head.named_parameters()
            if parameter.requires_grad
        ]

    def named_aqm_head_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (f"aqm_stage2_head.{name}", parameter)
            for name, parameter in self.aqm_stage2_head.named_parameters()
            if parameter.requires_grad
        ]

    def named_early_encoder_parameters(self) -> list[tuple[str, nn.Parameter]]:
        named_params: list[tuple[str, nn.Parameter]] = []
        for module_name in ("stem", "stage1", "down1", "stage2"):
            module = getattr(self.backbone, module_name)
            named_params.extend(
                (f"backbone.{module_name}.{name}", parameter)
                for name, parameter in module.named_parameters()
                if parameter.requires_grad
            )
        return named_params

    def named_late_encoder_parameters(self) -> list[tuple[str, nn.Parameter]]:
        named_params: list[tuple[str, nn.Parameter]] = []
        for module_name in ("down2", "stage3"):
            module = getattr(self.backbone, module_name)
            named_params.extend(
                (f"backbone.{module_name}.{name}", parameter)
                for name, parameter in module.named_parameters()
                if parameter.requires_grad
            )
        named_params.extend(
            (f"project.{name}", parameter)
            for name, parameter in self.project.named_parameters()
            if parameter.requires_grad
        )
        return named_params


class TemporalConvEncoder(BaseArchitecture):
    def __init__(self, in_channels_per_month: int, *, attention_swe_head: bool = False) -> None:
        super().__init__()
        self.backbone = SpatialBackbone(in_channels_per_month)
        self.temporal = nn.Sequential(
            nn.Conv3d(self.backbone.out_channels, self.backbone.out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(8, self.backbone.out_channels),
            nn.GELU(),
        )
        self.project = nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D)
        if attention_swe_head:
            self.swe_head = AttentionSWEHead(LATENT_D)
        else:
            self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))
        self.attention_swe_head = attention_swe_head

    def _monthly_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        x = x.reshape(batch_size * months, channels, height, width)
        features = self.backbone(x)
        _, feat_channels, feat_h, feat_w = features.shape
        return features.reshape(batch_size, months, feat_channels, feat_h, feat_w)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self._monthly_features(x)
        features = features.permute(0, 2, 1, 3, 4)
        features = self.temporal(features)
        features = features.mean(dim=2)
        pooled = F.adaptive_avg_pool2d(features, output_size=(1, 1)).reshape(features.shape[0], -1)
        latent = self.project(pooled)
        return latent.reshape(features.shape[0], LATENT_K, LATENT_D)

    def predict_swe(self, z: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        if self.attention_swe_head:
            return self.swe_head(z)
        return self.swe_head(z_flat).squeeze(-1)


class ConvGRUEncoder(BaseArchitecture):
    def __init__(self, in_channels_per_month: int) -> None:
        super().__init__()
        self.backbone = SpatialBackbone(in_channels_per_month)
        self.conv_gru = ConvGRUCell(self.backbone.out_channels)
        self.project = nn.Linear(self.backbone.out_channels, LATENT_K * LATENT_D)
        self.swe_head = nn.Sequential(nn.Linear(LATENT_K * LATENT_D, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, months, channels, height, width = x.shape
        x = x.reshape(batch_size * months, channels, height, width)
        features = self.backbone(x)
        _, feat_channels, feat_h, feat_w = features.shape
        features = features.reshape(batch_size, months, feat_channels, feat_h, feat_w)
        hidden = torch.zeros(
            batch_size,
            feat_channels,
            feat_h,
            feat_w,
            dtype=features.dtype,
            device=features.device,
        )
        for month_index in range(months):
            hidden = self.conv_gru(features[:, month_index], hidden)
        pooled = F.adaptive_avg_pool2d(hidden, output_size=(1, 1)).reshape(batch_size, -1)
        latent = self.project(pooled)
        return latent.reshape(batch_size, LATENT_K, LATENT_D)

    def predict_swe(self, z: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        return self.swe_head(z_flat).squeeze(-1)


def build_model(architecture: str, in_channels_per_month: int) -> BaseArchitecture:
    if architecture == "M1_static_cnn":
        return M1StaticCNN(in_channels_per_month=in_channels_per_month)
    if architecture == "D1_static_cnn_cpm_aqm_diag":
        return M1StaticCNN(in_channels_per_month=in_channels_per_month)
    if architecture == "D2_static_cnn_cpm_only":
        return StaticCNNCPMOnly(in_channels_per_month=in_channels_per_month)
    if architecture == "K1_static_cnn_kendall_aux":
        return StaticCNNKendallAux(in_channels_per_month=in_channels_per_month)
    if architecture == "P1_static_cnn_pcgrad":
        return M1StaticCNN(in_channels_per_month=in_channels_per_month)
    if architecture == "P2_static_cnn_gradsim":
        return M1StaticCNN(in_channels_per_month=in_channels_per_month)
    if architecture == "U1_stage2_cpm_aqm":
        return UpstreamStage2AuxCNN(in_channels_per_month=in_channels_per_month)
    if architecture == "M7_static_cnn_swe_only":
        return M1StaticCNN(in_channels_per_month=in_channels_per_month, include_auxiliary_heads=False)
    if architecture == "S0_static_cnn_swe_only":
        return M1StaticCNN(in_channels_per_month=in_channels_per_month, include_auxiliary_heads=False)
    if architecture == "M2_temporal_conv":
        return TemporalConvEncoder(in_channels_per_month=in_channels_per_month, attention_swe_head=False)
    if architecture == "M3_convgru":
        return ConvGRUEncoder(in_channels_per_month=in_channels_per_month)
    if architecture == "M4_attention_swe":
        return TemporalConvEncoder(in_channels_per_month=in_channels_per_month, attention_swe_head=True)
    if architecture == "M5_static_attention_swe_only":
        return StaticAttentionCNN(in_channels_per_month=in_channels_per_month, include_auxiliary_heads=False)
    if architecture == "M6_static_attention_with_aux":
        return StaticAttentionCNN(in_channels_per_month=in_channels_per_month, include_auxiliary_heads=True)
    if architecture == "S1_static_latent_self_attention":
        return StaticLatentSelfAttentionCNN(in_channels_per_month=in_channels_per_month)
    if architecture == "S2_static_swe_token":
        return StaticSWETokenAttentionCNN(in_channels_per_month=in_channels_per_month)
    if architecture == "S3_static_residual_gated_attention":
        return StaticResidualGatedAttentionCNN(in_channels_per_month=in_channels_per_month)
    raise ValueError(f"Unsupported architecture: {architecture}")


def compute_losses(
    predictions: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    swe_only: bool = False,
    kendall_model: BaseArchitecture | None = None,
) -> dict[str, torch.Tensor]:
    swe_loss = F.mse_loss(predictions["swe_hat"], targets[:, 0])
    zero = swe_loss.new_tensor(0.0)
    cpm_loss = zero if swe_only else (F.mse_loss(predictions["cpm_hat"], targets[:, 1]) if "cpm_hat" in predictions else zero)
    aqm_loss = zero if swe_only else (F.mse_loss(predictions["aqm_hat"], targets[:, 2]) if "aqm_hat" in predictions else zero)
    total_loss = swe_loss + cpm_loss + aqm_loss
    kendall_total_loss = total_loss
    if kendall_model is not None:
        assert isinstance(kendall_model, StaticCNNKendallAux)
        w_swe = 0.5 * torch.exp(-kendall_model.log_var_swe)
        w_cpm = 0.5 * torch.exp(-kendall_model.log_var_cpm)
        w_aqm = 0.5 * torch.exp(-kendall_model.log_var_aqm)
        kendall_total_loss = (
            w_swe * swe_loss
            + (0.5 * kendall_model.log_var_swe)
            + w_cpm * cpm_loss
            + (0.5 * kendall_model.log_var_cpm)
            + w_aqm * aqm_loss
            + (0.5 * kendall_model.log_var_aqm)
        )
        total_loss = kendall_total_loss
    return {
        "total_loss": total_loss,
        "kendall_total_loss": kendall_total_loss,
        "swe_loss": swe_loss,
        "cpm_loss": cpm_loss,
        "aqm_loss": aqm_loss,
    }


def _task_loss_key(task_name: str) -> str:
    return {
        "swe": "swe_loss",
        "cpm": "cpm_loss",
        "aqm": "aqm_loss",
    }[task_name]


def _gradient_norm(gradients: list[torch.Tensor | None]) -> float:
    total = 0.0
    for grad in gradients:
        if grad is not None:
            total += float(torch.sum(grad.detach().float() ** 2).item())
    return math.sqrt(total)


def _gradient_cosine(gradients_a: list[torch.Tensor | None], gradients_b: list[torch.Tensor | None]) -> float:
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for grad_a, grad_b in zip(gradients_a, gradients_b, strict=False):
        if grad_a is not None:
            grad_a_f = grad_a.detach().float()
            norm_a_sq += float(torch.sum(grad_a_f * grad_a_f).item())
        if grad_b is not None:
            grad_b_f = grad_b.detach().float()
            norm_b_sq += float(torch.sum(grad_b_f * grad_b_f).item())
        if grad_a is not None and grad_b is not None:
            dot += float(torch.sum(grad_a_f * grad_b_f).item())
    if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
        return float("nan")
    return dot / math.sqrt(norm_a_sq * norm_b_sq)


def _flatten_gradient_list(gradients: list[torch.Tensor | None]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    device: torch.device | None = None
    for grad in gradients:
        if grad is None:
            continue
        grad_f = grad.detach().reshape(-1).float()
        device = grad_f.device
        chunks.append(grad_f)
    if not chunks:
        return torch.zeros(0, dtype=torch.float32, device=device or torch.device("cpu"))
    return torch.cat(chunks, dim=0)


def _split_flat_gradient_like(flat_gradient: torch.Tensor, reference_gradients: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
    pieces: list[torch.Tensor | None] = []
    start = 0
    for grad in reference_gradients:
        if grad is None:
            pieces.append(None)
            continue
        numel = grad.numel()
        chunk = flat_gradient[start : start + numel].reshape(grad.shape).to(dtype=grad.dtype, device=grad.device)
        pieces.append(chunk)
        start += numel
    return pieces


def _combine_nonencoder_task_grads(
    *,
    task_grads_all_params: dict[str, list[torch.Tensor | None]],
    all_params: list[nn.Parameter],
    encoder_param_ids: set[int],
) -> list[torch.Tensor | None]:
    combined: list[torch.Tensor | None] = []
    task_names = list(task_grads_all_params.keys())
    for param_index, parameter in enumerate(all_params):
        if id(parameter) in encoder_param_ids:
            combined.append(None)
            continue
        grad_sum: torch.Tensor | None = None
        for task_name in task_names:
            grad = task_grads_all_params[task_name][param_index]
            if grad is None:
                continue
            grad_sum = grad.detach().clone() if grad_sum is None else grad_sum + grad.detach()
        combined.append(grad_sum)
    return combined


def _collect_task_gradients_all_params(
    *,
    losses: dict[str, torch.Tensor],
    active_tasks: tuple[str, ...],
    all_params: list[nn.Parameter],
) -> dict[str, list[torch.Tensor | None]]:
    gradients_by_task: dict[str, list[torch.Tensor | None]] = {}
    for task_name in active_tasks:
        gradients = torch.autograd.grad(
            losses[_task_loss_key(task_name)],
            all_params,
            retain_graph=True,
            allow_unused=True,
        )
        gradients_by_task[task_name] = [gradient.detach() if gradient is not None else None for gradient in gradients]
    return gradients_by_task


def _gradients_for_named_parameters(
    *,
    loss: torch.Tensor,
    named_parameters: list[tuple[str, nn.Parameter]],
) -> tuple[list[torch.Tensor], int]:
    parameters = [parameter for _, parameter in named_parameters]
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    filled: list[torch.Tensor] = []
    unused_count = 0
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        if gradient is None:
            filled_gradient = torch.zeros_like(parameter)
            unused_count += 1
        else:
            filled_gradient = gradient.detach()
        filled.append(filled_gradient)
    return filled, unused_count


def _assert_gradient_list_invariants(
    *,
    group_name: str,
    named_parameters: list[tuple[str, nn.Parameter]],
    gradients: list[torch.Tensor],
) -> None:
    assert len(named_parameters) > 0, f"{group_name} parameter list is empty"
    assert len(named_parameters) == len(gradients), (
        f"{group_name} gradient length mismatch: params={len(named_parameters)} grads={len(gradients)}"
    )
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        assert gradient is not None, f"{group_name} gradient unexpectedly None for {name}"
        assert gradient.shape == parameter.shape, (
            f"{group_name} gradient shape mismatch for {name}: grad={tuple(gradient.shape)} param={tuple(parameter.shape)}"
        )
        assert torch.isfinite(gradient).all().item(), f"{group_name} gradient non-finite for {name}"
        assert gradient.device == parameter.device, f"{group_name} gradient device mismatch for {name}"
        assert gradient.dtype == parameter.dtype, f"{group_name} gradient dtype mismatch for {name}"


def _log_parameter_group_summary(
    *,
    model: BaseArchitecture,
    architecture: str,
) -> None:
    group_specs = [
        ("encoder", model.named_encoder_parameters()),
        ("swe_head", model.named_swe_head_parameters()),
        ("cpm_head", model.named_cpm_head_parameters()),
        ("aqm_head", model.named_aqm_head_parameters()),
    ]
    _log(f"PARAMETER GROUP SUMMARY architecture={architecture}")
    for group_name, named_parameters in group_specs:
        _log(f"PARAMETER GROUP {group_name} n_params={len(named_parameters)}")
        for name, parameter in named_parameters[:5]:
            _log(f"PARAMETER DETAIL group={group_name} name={name} shape={tuple(parameter.shape)}")


def _log_gradient_group_summary(
    *,
    group_name: str,
    named_parameters: list[tuple[str, nn.Parameter]],
    gradients: list[torch.Tensor],
) -> None:
    _log(
        f"GRADIENT GROUP group={group_name} n_params={len(named_parameters)} grad_list_len={len(gradients)}"
    )
    for (name, parameter), gradient in list(zip(named_parameters, gradients, strict=True))[:5]:
        _log(
            f"GRADIENT DETAIL group={group_name} name={name} param_shape={tuple(parameter.shape)} grad_shape={tuple(gradient.shape)}"
        )


def _pcgrad_project_encoder_gradients(
    *,
    encoder_gradients_by_task: dict[str, list[torch.Tensor | None]],
    rng: np.random.Generator,
) -> tuple[dict[str, list[torch.Tensor | None]], dict[str, float]]:
    task_names = list(encoder_gradients_by_task.keys())
    flat_by_task = {task_name: _flatten_gradient_list(encoder_gradients_by_task[task_name]) for task_name in task_names}
    projected_flat: dict[str, torch.Tensor] = {}
    shuffled = list(task_names)
    rng.shuffle(shuffled)
    projection_triggers = 0
    projection_checks = 0
    for task_name in task_names:
        grad_task = flat_by_task[task_name].clone()
        for other_name in shuffled:
            if other_name == task_name:
                continue
            other_grad = flat_by_task[other_name]
            denom = float(torch.sum(other_grad * other_grad).item())
            if denom <= 0.0:
                continue
            inner_product = float(torch.sum(grad_task * other_grad).item())
            projection_checks += 1
            if inner_product < 0.0:
                grad_task = grad_task - ((inner_product / denom) * other_grad)
                projection_triggers += 1
        projected_flat[task_name] = grad_task
    projected = {
        task_name: _split_flat_gradient_like(projected_flat[task_name], encoder_gradients_by_task[task_name])
        for task_name in task_names
    }
    diagnostics: dict[str, float] = {
        "pcgrad_projection_trigger_fraction": (
            float(projection_triggers) / float(projection_checks) if projection_checks > 0 else 0.0
        )
    }
    for task_a, task_b in (("swe", "cpm"), ("swe", "aqm"), ("cpm", "aqm")):
        if task_a in projected and task_b in projected:
            diagnostics[f"post_grad_cos_{task_a}_{task_b}"] = _gradient_cosine(projected[task_a], projected[task_b])
    return projected, diagnostics


def _gradsim_encoder_combination(
    *,
    encoder_gradients_by_task: dict[str, list[torch.Tensor | None]],
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    swe_grads = encoder_gradients_by_task["swe"]
    diagnostics: dict[str, float] = {}
    coefficients = {"swe": 1.0}
    combined = [grad.detach().clone() if grad is not None else None for grad in swe_grads]
    for aux_name in ("cpm", "aqm"):
        if aux_name not in encoder_gradients_by_task:
            continue
        cosine = _gradient_cosine(swe_grads, encoder_gradients_by_task[aux_name])
        coefficient = max(0.0, cosine) if not math.isnan(cosine) else 0.0
        coefficients[aux_name] = coefficient
        diagnostics[f"gradsim_coeff_{aux_name}"] = coefficient
        diagnostics[f"gradsim_contributed_{aux_name}"] = 1.0 if coefficient > 0.0 else 0.0
        if coefficient <= 0.0:
            continue
        for index, grad in enumerate(encoder_gradients_by_task[aux_name]):
            if grad is None:
                continue
            if combined[index] is None:
                combined[index] = coefficient * grad.detach()
            else:
                combined[index] = combined[index] + (coefficient * grad.detach())
    return combined, diagnostics


def compute_gradient_diagnostics(
    *,
    model: BaseArchitecture,
    losses: dict[str, torch.Tensor],
    active_tasks: tuple[str, ...],
    task_weight_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    encoder_params = [parameter for parameter in model.shared_encoder_parameters() if parameter.requires_grad]
    grads_by_task: dict[str, list[torch.Tensor | None]] = {}
    diagnostics: dict[str, float] = {}
    for task_name in active_tasks:
        gradients = torch.autograd.grad(
            losses[_task_loss_key(task_name)],
            encoder_params,
            retain_graph=True,
            allow_unused=True,
        )
        grads_by_task[task_name] = [gradient.detach() if gradient is not None else None for gradient in gradients]
        diagnostics[f"grad_norm_{task_name}"] = _gradient_norm(grads_by_task[task_name])
        if task_weight_metrics is not None and f"w_{task_name}" in task_weight_metrics:
            diagnostics[f"weighted_grad_norm_{task_name}"] = (
                float(task_weight_metrics[f"w_{task_name}"]) * diagnostics[f"grad_norm_{task_name}"]
            )
    for task_a, task_b in (("swe", "cpm"), ("swe", "aqm"), ("cpm", "aqm")):
        if task_a in grads_by_task and task_b in grads_by_task:
            diagnostics[f"grad_cos_{task_a}_{task_b}"] = _gradient_cosine(grads_by_task[task_a], grads_by_task[task_b])
    return diagnostics


def compute_upstream_gradient_diagnostics(
    *,
    model: BaseArchitecture,
    losses: dict[str, torch.Tensor],
    active_tasks: tuple[str, ...],
) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    early_named = model.named_early_encoder_parameters()
    late_named = model.named_late_encoder_parameters()
    early_params = [parameter for _, parameter in early_named]
    late_params = [parameter for _, parameter in late_named]
    early_grads_by_task: dict[str, list[torch.Tensor | None]] = {}
    late_grads_by_task: dict[str, list[torch.Tensor | None]] = {}
    for task_name in active_tasks:
        early_grads = torch.autograd.grad(
            losses[_task_loss_key(task_name)],
            early_params,
            retain_graph=True,
            allow_unused=True,
        )
        early_grads_by_task[task_name] = [gradient.detach() if gradient is not None else None for gradient in early_grads]
        diagnostics[f"grad_norm_{task_name}_early"] = _gradient_norm(early_grads_by_task[task_name])
        if late_params:
            late_grads = torch.autograd.grad(
                losses[_task_loss_key(task_name)],
                late_params,
                retain_graph=True,
                allow_unused=True,
            )
            late_grads_by_task[task_name] = [gradient.detach() if gradient is not None else None for gradient in late_grads]
            diagnostics[f"grad_norm_{task_name}_late"] = _gradient_norm(late_grads_by_task[task_name])
    for task_a, task_b in (("swe", "cpm"), ("swe", "aqm")):
        if task_a in early_grads_by_task and task_b in early_grads_by_task:
            diagnostics[f"grad_cos_{task_a}_{task_b}_early"] = _gradient_cosine(
                early_grads_by_task[task_a],
                early_grads_by_task[task_b],
            )
    return diagnostics


def run_epoch(
    *,
    architecture: str,
    model: BaseArchitecture,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
    epoch_index: int,
    stage_name: str,
    swe_only_loss: bool,
    active_tasks: tuple[str, ...],
    record_gradient_diagnostics: bool,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float]]:
    is_training = optimizer is not None
    use_kendall_weighting = isinstance(model, StaticCNNKendallAux)
    use_pcgrad = architecture_uses_pcgrad(architecture)
    use_gradsim = architecture_uses_gradsim(architecture)
    use_upstream_aux = architecture_uses_upstream_aux(architecture)
    use_custom_aux_gradient_handling = use_pcgrad or use_gradsim
    model.train(is_training)
    if is_training and use_custom_aux_gradient_handling and epoch_index == 1:
        _log_parameter_group_summary(model=model, architecture=architecture)
    totals = {"total_loss": 0.0, "kendall_total_loss": 0.0, "swe_loss": 0.0, "cpm_loss": 0.0, "aqm_loss": 0.0}
    count = 0
    predictions_out: list[dict[str, Any]] = []
    grad_diag_sums: dict[str, float] = {}
    grad_diag_counts: dict[str, int] = {}
    grad_diag_negative_counts: dict[str, int] = {}
    autocast_enabled = use_amp and device.type == "cuda"
    stage_wall_start = time.time()
    rng = np.random.default_rng(20260817 + epoch_index)
    with torch.set_grad_enabled(is_training):
        _log(f"START first batch fetch stage={stage_name} epoch={epoch_index}")
        for batch_inputs, batch_targets, batch_metadata in loader:
            batch_index = count + 1
            if batch_index == 1:
                _log(f"END first batch fetch stage={stage_name} epoch={epoch_index}")
                _log(
                    f"FIRST BATCH SHAPES stage={stage_name} epoch={epoch_index} "
                    f"inputs={tuple(batch_inputs.shape)} targets={tuple(batch_targets.shape)}"
                )
            if batch_index == 1:
                _log(f"START GPU transfer stage={stage_name} epoch={epoch_index}")
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            if batch_index == 1:
                _log(f"END GPU transfer stage={stage_name} epoch={epoch_index}")
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            if batch_index == 1:
                _log(f"START forward stage={stage_name} epoch={epoch_index}")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                outputs = model(batch_inputs)
                cpm_hat = outputs.get("cpm_hat", torch.full_like(outputs["swe_hat"], torch.nan))
                aqm_hat = outputs.get("aqm_hat", torch.full_like(outputs["swe_hat"], torch.nan))
                if batch_index == 1:
                    _log(f"END forward stage={stage_name} epoch={epoch_index}")
                    _log(f"START loss stage={stage_name} epoch={epoch_index}")
                losses = compute_losses(
                    outputs,
                    batch_targets,
                    swe_only=swe_only_loss,
                    kendall_model=model if use_kendall_weighting else None,
                )
                if batch_index == 1:
                    _log(f"END loss stage={stage_name} epoch={epoch_index}")
            batch_grad_diagnostics: dict[str, float] = {}
            task_weight_metrics = model.current_task_weights() if use_kendall_weighting else None
            projected_encoder_gradients: dict[str, list[torch.Tensor | None]] | None = None
            gradsim_encoder_combined: list[torch.Tensor | None] | None = None
            custom_method_diagnostics: dict[str, float] = {}
            if is_training and use_custom_aux_gradient_handling:
                encoder_named_params = model.named_encoder_parameters()
                swe_head_named_params = model.named_swe_head_parameters()
                cpm_head_named_params = model.named_cpm_head_parameters()
                aqm_head_named_params = model.named_aqm_head_parameters()

                assert len(encoder_named_params) > 0, "encoder_params is empty"
                assert len(swe_head_named_params) > 0, "swe_head_params is empty"
                assert len(cpm_head_named_params) > 0, "cpm_head_params is empty"
                assert len(aqm_head_named_params) > 0, "aqm_head_params is empty"

                g_swe_encoder, unused_swe_encoder = _gradients_for_named_parameters(
                    loss=losses["swe_loss"],
                    named_parameters=encoder_named_params,
                )
                g_cpm_encoder, unused_cpm_encoder = _gradients_for_named_parameters(
                    loss=losses["cpm_loss"],
                    named_parameters=encoder_named_params,
                )
                g_aqm_encoder, unused_aqm_encoder = _gradients_for_named_parameters(
                    loss=losses["aqm_loss"],
                    named_parameters=encoder_named_params,
                )
                encoder_gradients_by_task = {
                    "swe": g_swe_encoder,
                    "cpm": g_cpm_encoder,
                    "aqm": g_aqm_encoder,
                }
                _assert_gradient_list_invariants(
                    group_name="encoder_swe",
                    named_parameters=encoder_named_params,
                    gradients=g_swe_encoder,
                )
                _assert_gradient_list_invariants(
                    group_name="encoder_cpm",
                    named_parameters=encoder_named_params,
                    gradients=g_cpm_encoder,
                )
                _assert_gradient_list_invariants(
                    group_name="encoder_aqm",
                    named_parameters=encoder_named_params,
                    gradients=g_aqm_encoder,
                )

                g_swe_head, unused_swe_head = _gradients_for_named_parameters(
                    loss=losses["swe_loss"],
                    named_parameters=swe_head_named_params,
                )
                g_cpm_head, unused_cpm_head = _gradients_for_named_parameters(
                    loss=losses["cpm_loss"],
                    named_parameters=cpm_head_named_params,
                )
                g_aqm_head, unused_aqm_head = _gradients_for_named_parameters(
                    loss=losses["aqm_loss"],
                    named_parameters=aqm_head_named_params,
                )
                _assert_gradient_list_invariants(
                    group_name="swe_head",
                    named_parameters=swe_head_named_params,
                    gradients=g_swe_head,
                )
                _assert_gradient_list_invariants(
                    group_name="cpm_head",
                    named_parameters=cpm_head_named_params,
                    gradients=g_cpm_head,
                )
                _assert_gradient_list_invariants(
                    group_name="aqm_head",
                    named_parameters=aqm_head_named_params,
                    gradients=g_aqm_head,
                )
                custom_method_diagnostics.update(
                    {
                        "unused_encoder_swe": float(unused_swe_encoder),
                        "unused_encoder_cpm": float(unused_cpm_encoder),
                        "unused_encoder_aqm": float(unused_aqm_encoder),
                        "unused_swe_head": float(unused_swe_head),
                        "unused_cpm_head": float(unused_cpm_head),
                        "unused_aqm_head": float(unused_aqm_head),
                    }
                )
                if batch_index == 1 and epoch_index == 1:
                    _log_gradient_group_summary(
                        group_name="encoder_swe",
                        named_parameters=encoder_named_params,
                        gradients=g_swe_encoder,
                    )
                    _log_gradient_group_summary(
                        group_name="encoder_cpm",
                        named_parameters=encoder_named_params,
                        gradients=g_cpm_encoder,
                    )
                    _log_gradient_group_summary(
                        group_name="encoder_aqm",
                        named_parameters=encoder_named_params,
                        gradients=g_aqm_encoder,
                    )
                    _log_gradient_group_summary(
                        group_name="swe_head",
                        named_parameters=swe_head_named_params,
                        gradients=g_swe_head,
                    )
                    _log_gradient_group_summary(
                        group_name="cpm_head",
                        named_parameters=cpm_head_named_params,
                        gradients=g_cpm_head,
                    )
                    _log_gradient_group_summary(
                        group_name="aqm_head",
                        named_parameters=aqm_head_named_params,
                        gradients=g_aqm_head,
                    )
                if use_pcgrad:
                    projected_encoder_gradients, pcgrad_diagnostics = _pcgrad_project_encoder_gradients(
                        encoder_gradients_by_task=encoder_gradients_by_task,
                        rng=rng,
                    )
                    custom_method_diagnostics.update(pcgrad_diagnostics)
                elif use_gradsim:
                    gradsim_encoder_combined, gradsim_diagnostics = _gradsim_encoder_combination(
                        encoder_gradients_by_task=encoder_gradients_by_task,
                    )
                    custom_method_diagnostics.update(gradsim_diagnostics)
            if is_training and record_gradient_diagnostics:
                if use_custom_aux_gradient_handling:
                    batch_grad_diagnostics = {}
                    for task_name, gradients in encoder_gradients_by_task.items():
                        batch_grad_diagnostics[f"grad_norm_{task_name}"] = _gradient_norm(gradients)
                        if task_weight_metrics is not None and f"w_{task_name}" in task_weight_metrics:
                            batch_grad_diagnostics[f"weighted_grad_norm_{task_name}"] = (
                                float(task_weight_metrics[f"w_{task_name}"]) * batch_grad_diagnostics[f"grad_norm_{task_name}"]
                            )
                    for task_a, task_b in (("swe", "cpm"), ("swe", "aqm"), ("cpm", "aqm")):
                        if task_a in encoder_gradients_by_task and task_b in encoder_gradients_by_task:
                            batch_grad_diagnostics[f"grad_cos_{task_a}_{task_b}"] = _gradient_cosine(
                                encoder_gradients_by_task[task_a],
                                encoder_gradients_by_task[task_b],
                            )
                else:
                    if use_upstream_aux:
                        batch_grad_diagnostics = compute_upstream_gradient_diagnostics(
                            model=model,
                            losses=losses,
                            active_tasks=active_tasks,
                        )
                    else:
                        batch_grad_diagnostics = compute_gradient_diagnostics(
                            model=model,
                            losses=losses,
                            active_tasks=active_tasks,
                            task_weight_metrics=task_weight_metrics,
                        )
                batch_grad_diagnostics.update(custom_method_diagnostics)
                for key, value in batch_grad_diagnostics.items():
                    if math.isnan(value):
                        continue
                    grad_diag_sums[key] = grad_diag_sums.get(key, 0.0) + float(value)
                    grad_diag_counts[key] = grad_diag_counts.get(key, 0) + 1
                    if key.startswith("grad_cos_"):
                        negative_key = key.replace("grad_cos_", "grad_frac_negative_")
                        grad_diag_negative_counts[negative_key] = grad_diag_negative_counts.get(negative_key, 0) + int(value < 0.0)
            total_loss_finite = bool(torch.isfinite(losses["total_loss"]).all().item())
            if not total_loss_finite:
                _log(
                    f"NONFINITE LOSS DETECTED stage={stage_name} epoch={epoch_index} batch={batch_index} "
                    f"total_finite={total_loss_finite}"
                )
                _log(
                    f"LOSS COMPONENTS stage={stage_name} epoch={epoch_index} batch={batch_index} "
                    f"swe_loss={float(losses['swe_loss'].detach().cpu().item())} "
                    f"cpm_loss={float(losses['cpm_loss'].detach().cpu().item())} "
                    f"aqm_loss={float(losses['aqm_loss'].detach().cpu().item())}"
                )
                _log(
                    f"OUTPUT FINITE COUNTS stage={stage_name} epoch={epoch_index} batch={batch_index} "
                    f"swe_hat={int(torch.isfinite(outputs['swe_hat']).sum().item())}/{outputs['swe_hat'].numel()} "
                    f"cpm_hat={int(torch.isfinite(cpm_hat).sum().item())}/{cpm_hat.numel()} "
                    f"aqm_hat={int(torch.isfinite(aqm_hat).sum().item())}/{aqm_hat.numel()}"
                )
                _log(
                    f"TARGET FINITE COUNTS stage={stage_name} epoch={epoch_index} batch={batch_index} "
                    f"targets={int(torch.isfinite(batch_targets).sum().item())}/{batch_targets.numel()}"
                )
                _log(
                    f"INPUT FINITE COUNTS stage={stage_name} epoch={epoch_index} batch={batch_index} "
                    f"inputs={int(torch.isfinite(batch_inputs).sum().item())}/{batch_inputs.numel()} "
                    f"max_abs_input={float(torch.nan_to_num(batch_inputs).abs().max().detach().cpu().item()):.6f}"
                )
                for row_index, meta in enumerate(batch_metadata):
                    _log(
                        f"BAD BATCH SAMPLE stage={stage_name} epoch={epoch_index} batch={batch_index} "
                        f"row={row_index} metadata={json.dumps(meta, sort_keys=True)} "
                        f"swe_hat={float(outputs['swe_hat'][row_index].detach().cpu().item())} "
                        f"cpm_hat={float(cpm_hat[row_index].detach().cpu().item())} "
                        f"aqm_hat={float(aqm_hat[row_index].detach().cpu().item())} "
                        f"target={batch_targets[row_index].detach().cpu().tolist()}"
                    )
            if is_training:
                if batch_index == 1:
                    _log(f"START backward stage={stage_name} epoch={epoch_index}")
                if use_custom_aux_gradient_handling:
                    if use_pcgrad:
                        assert projected_encoder_gradients is not None
                        combined_encoder_grads = []
                        for encoder_index, reference in enumerate(projected_encoder_gradients["swe"]):
                            grad_sum = torch.zeros_like(reference)
                            for task_name in active_tasks:
                                grad_sum = grad_sum + projected_encoder_gradients[task_name][encoder_index]
                            combined_encoder_grads.append(grad_sum)
                    else:
                        assert gradsim_encoder_combined is not None
                        combined_encoder_grads = gradsim_encoder_combined
                    _assert_gradient_list_invariants(
                        group_name="merged_encoder",
                        named_parameters=encoder_named_params,
                        gradients=combined_encoder_grads,
                    )
                    for (_, parameter), gradient in zip(encoder_named_params, combined_encoder_grads, strict=True):
                        parameter.grad = gradient.detach().clone().to(parameter.dtype)
                    for (_, parameter), gradient in zip(swe_head_named_params, g_swe_head, strict=True):
                        parameter.grad = gradient.detach().clone().to(parameter.dtype)
                    for (_, parameter), gradient in zip(cpm_head_named_params, g_cpm_head, strict=True):
                        parameter.grad = gradient.detach().clone().to(parameter.dtype)
                    for (_, parameter), gradient in zip(aqm_head_named_params, g_aqm_head, strict=True):
                        parameter.grad = gradient.detach().clone().to(parameter.dtype)
                    if batch_index == 1:
                        _log(f"END backward stage={stage_name} epoch={epoch_index}")
                        _log(f"START optimizer step stage={stage_name} epoch={epoch_index}")
                    optimizer.step()
                elif scaler is not None and autocast_enabled:
                    scaler.scale(losses["total_loss"]).backward()
                    if batch_index == 1:
                        _log(f"END backward stage={stage_name} epoch={epoch_index}")
                        _log(f"START optimizer step stage={stage_name} epoch={epoch_index}")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total_loss"].backward()
                    if batch_index == 1:
                        _log(f"END backward stage={stage_name} epoch={epoch_index}")
                        _log(f"START optimizer step stage={stage_name} epoch={epoch_index}")
                    optimizer.step()
                if batch_index == 1:
                    _log(f"END optimizer step stage={stage_name} epoch={epoch_index}")
            for key in totals:
                totals[key] += float(losses[key].detach().item())
            count += 1
            predictions_out.extend(
                {
                    "metadata": meta,
                    "swe_hat_z": float(outputs["swe_hat"][row].detach().cpu().item()),
                    "cpm_hat_z": float(cpm_hat[row].detach().cpu().item()),
                    "aqm_hat_z": float(aqm_hat[row].detach().cpu().item()),
                    "swe_true_z": float(batch_targets[row, 0].detach().cpu().item()),
                    "cpm_true_z": float(batch_targets[row, 1].detach().cpu().item()),
                    "aqm_true_z": float(batch_targets[row, 2].detach().cpu().item()),
                }
                for row, meta in enumerate(batch_metadata)
            )
            if count == 1:
                _log(
                    f"FIRST BATCH COMPLETE stage={stage_name} epoch={epoch_index} "
                    f"loss={float(losses['total_loss'].detach().item()):.6f}"
                )
            if count % 10 == 0:
                _log(
                    f"BATCH PROGRESS stage={stage_name} epoch={epoch_index} "
                    f"batch={count} elapsed_seconds={time.time() - stage_wall_start:.2f} "
                    f"loss={float(losses['total_loss'].detach().item()):.6f}"
                )
    averages = {key: value / max(count, 1) for key, value in totals.items()}
    gradient_summary: dict[str, float] = {}
    for key, total in grad_diag_sums.items():
        gradient_summary[key] = total / max(grad_diag_counts.get(key, 1), 1)
        if key.startswith("grad_cos_"):
            negative_key = key.replace("grad_cos_", "grad_frac_negative_")
            gradient_summary[negative_key] = grad_diag_negative_counts.get(negative_key, 0) / max(grad_diag_counts.get(key, 1), 1)
    _log(
        f"END EPOCH STAGE stage={stage_name} epoch={epoch_index} "
        f"batches={count} elapsed_seconds={time.time() - stage_wall_start:.2f}"
    )
    return averages, predictions_out, gradient_summary


def invert_targets(values_z: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return values_z * sigma + mu


def summarize_predictions(predictions: list[dict[str, Any]], target_mu: np.ndarray, target_sigma: np.ndarray) -> dict[str, float]:
    swe_pred_z = np.asarray([row["swe_hat_z"] for row in predictions], dtype=np.float32)
    swe_true_z = np.asarray([row["swe_true_z"] for row in predictions], dtype=np.float32)
    cpm_pred_z = np.asarray([row["cpm_hat_z"] for row in predictions], dtype=np.float32)
    cpm_true_z = np.asarray([row["cpm_true_z"] for row in predictions], dtype=np.float32)
    aqm_pred_z = np.asarray([row["aqm_hat_z"] for row in predictions], dtype=np.float32)
    aqm_true_z = np.asarray([row["aqm_true_z"] for row in predictions], dtype=np.float32)

    swe_pred = invert_targets(swe_pred_z, target_mu[0], target_sigma[0])
    swe_true = invert_targets(swe_true_z, target_mu[0], target_sigma[0])
    cpm_pred = invert_targets(cpm_pred_z, target_mu[1], target_sigma[1])
    cpm_true = invert_targets(cpm_true_z, target_mu[1], target_sigma[1])
    aqm_pred = invert_targets(aqm_pred_z, target_mu[2], target_sigma[2])
    aqm_true = invert_targets(aqm_true_z, target_mu[2], target_sigma[2])

    swe_mse = np.mean((swe_pred - swe_true) ** 2)
    cpm_finite = np.isfinite(cpm_pred) & np.isfinite(cpm_true)
    aqm_finite = np.isfinite(aqm_pred) & np.isfinite(aqm_true)
    cpm_mse = np.mean((cpm_pred[cpm_finite] - cpm_true[cpm_finite]) ** 2) if np.any(cpm_finite) else float("nan")
    aqm_mse = np.mean((aqm_pred[aqm_finite] - aqm_true[aqm_finite]) ** 2) if np.any(aqm_finite) else float("nan")

    return {
        "swe_rmse": float(np.sqrt(swe_mse)),
        "swe_mae": float(np.mean(np.abs(swe_pred - swe_true))),
        "swe_r2": r2_score(swe_true, swe_pred),
        "swe_pearson_r": pearson_r(swe_true, swe_pred),
        "swe_spearman_rho": spearman_rho(swe_true, swe_pred),
        "swe_wet_dry_accuracy": wet_dry_accuracy(swe_true, swe_pred),
        "cpm_mse": float(cpm_mse),
        "cpm_pearson_r": pearson_r(cpm_true[cpm_finite], cpm_pred[cpm_finite]) if np.any(cpm_finite) else float("nan"),
        "aqm_mse": float(aqm_mse),
        "aqm_pearson_r": pearson_r(aqm_true[aqm_finite], aqm_pred[aqm_finite]) if np.any(aqm_finite) else float("nan"),
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def train_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    cache_dir = Path(config.predictor_cache_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"START train_experiment architecture={config.architecture} output_dir={output_dir}")

    manifest_rows = read_manifest(cache_dir / "manifest.csv")
    split, split_verification = load_or_build_split(
        manifest_rows,
        seed=config.seed,
        split_json_path=config.split_json_path,
        split_manifest_path=config.split_manifest_path,
    )
    write_json(output_dir / "split.json", split)
    write_json(output_dir / "split_verification.json", split_verification)

    physical_data = np.load(cache_dir / "inputs_physical.npy", mmap_mode="r")
    valid_mask = np.load(cache_dir / "inputs_valid_mask.npy", mmap_mode="r")
    targets = np.load(cache_dir / "targets.npy")

    normalization_start = time.time()
    _log("START train/validation normalization")
    feature_mu, feature_sigma, valid_count = compute_feature_stats(physical_data, split["train"])
    target_mu, target_sigma = compute_target_stats(targets, split["train"])
    _log(f"END train/validation normalization elapsed_seconds={time.time() - normalization_start:.2f}")
    np.savez_compressed(
        output_dir / "normalization_stats.npz",
        feature_mu=feature_mu,
        feature_sigma=feature_sigma,
        feature_valid_count=valid_count,
        target_mu=target_mu,
        target_sigma=target_sigma,
    )
    if (cache_dir / "predictor_inventory.json").exists():
        predictor_inventory = load_json(cache_dir / "predictor_inventory.json")
        write_json(output_dir / "predictor_inventory.json", predictor_inventory)
    else:
        predictor_inventory = {
            "predictor_fields": [],
            "predictor_count": int(physical_data.shape[2]),
            "mask_count": int(physical_data.shape[2]),
            "stacked_channel_count": int(physical_data.shape[2] * 2),
        }
        write_json(output_dir / "predictor_inventory.json", predictor_inventory)

    loader_start = time.time()
    _log("START DataLoader construction")
    train_dataset = CMIP6TensorDataset(
        physical_data=physical_data,
        valid_mask=valid_mask,
        targets=targets,
        metadata=manifest_rows,
        indices=split["train"],
        feature_mu=feature_mu,
        feature_sigma=feature_sigma,
        target_mu=target_mu,
        target_sigma=target_sigma,
    )
    val_dataset = CMIP6TensorDataset(
        physical_data=physical_data,
        valid_mask=valid_mask,
        targets=targets,
        metadata=manifest_rows,
        indices=split["val"],
        feature_mu=feature_mu,
        feature_sigma=feature_sigma,
        target_mu=target_mu,
        target_sigma=target_sigma,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_with_metadata,
    )
    train_eval_loader = (
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_with_metadata,
        )
        if architecture_uses_eval_mode_train_metrics(config.architecture)
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_with_metadata,
    )
    _log(f"END DataLoader construction elapsed_seconds={time.time() - loader_start:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.architecture, in_channels_per_month=physical_data.shape[2] * 2)
    model.to(device)
    historical_batch_sampler: HistoricalBatchOrderSampler | None = None
    historical_batch_verification: dict[str, Any] | None = None
    historical_train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]] | None = None
    if config.historical_batch_order_audit_path is not None:
        if config.architecture not in {
            "S0_static_cnn_swe_only",
            "S1_static_latent_self_attention",
            "S2_static_swe_token",
            "S3_static_residual_gated_attention",
        }:
            raise ValueError("Historical batch-order replay is supported only for S0-S3")
        position_batches_by_epoch, historical_batch_verification = build_historical_batch_orders(
            architecture=config.architecture,
            audit_path=Path(config.historical_batch_order_audit_path),
            train_source_indices=split["train"],
            val_source_indices=split["val"],
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            rng_state_after_model_init=torch.get_rng_state(),
        )
        historical_batch_sampler = HistoricalBatchOrderSampler(position_batches_by_epoch)
        historical_train_loader = DataLoader(
            train_dataset,
            batch_sampler=historical_batch_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_with_metadata,
        )
        write_json(output_dir / "historical_batch_order_verification.json", historical_batch_verification)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    parameter_count = count_parameters(model)
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    history_rows: list[dict[str, Any]] = []
    patience_counter = 0
    start_time = time.time()
    swe_only_loss = architecture_uses_swe_only_loss(config.architecture)
    use_eval_mode_train_metrics = architecture_uses_eval_mode_train_metrics(config.architecture)
    active_tasks = architecture_active_tasks(config.architecture)
    record_gradient_diagnostics = architecture_records_gradient_diagnostics(config.architecture)

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if historical_batch_sampler is not None:
            historical_batch_sampler.set_epoch(epoch)
        optimization_loader = historical_train_loader if historical_train_loader is not None else train_loader
        _log(f"START epoch={epoch} stage=train")
        train_losses, _train_step_predictions, train_gradient_summary = run_epoch(
            architecture=config.architecture,
            model=model,
            loader=optimization_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=config.amp,
            epoch_index=epoch,
            stage_name="train",
            swe_only_loss=swe_only_loss,
            active_tasks=active_tasks,
            record_gradient_diagnostics=record_gradient_diagnostics,
        )
        train_eval_loss_fields: dict[str, float] = {}
        if use_eval_mode_train_metrics:
            assert train_eval_loader is not None
            _log(f"START epoch={epoch} stage=train_eval")
            train_eval_losses, train_eval_predictions, _ = run_epoch(
                architecture=config.architecture,
                model=model,
                loader=train_eval_loader,
                device=device,
                optimizer=None,
                scaler=None,
                use_amp=config.amp,
                epoch_index=epoch,
                stage_name="train_eval",
                swe_only_loss=swe_only_loss,
                active_tasks=active_tasks,
                record_gradient_diagnostics=False,
            )
            train_metrics = summarize_predictions(train_eval_predictions, target_mu=target_mu, target_sigma=target_sigma)
            train_eval_loss_fields = {f"train_eval_{key}": value for key, value in train_eval_losses.items()}
        else:
            train_metrics = summarize_predictions(_train_step_predictions, target_mu=target_mu, target_sigma=target_sigma)
        _log(f"START epoch={epoch} stage=val")
        val_losses, val_predictions, _ = run_epoch(
            architecture=config.architecture,
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=config.amp,
            epoch_index=epoch,
            stage_name="val",
            swe_only_loss=swe_only_loss,
            active_tasks=active_tasks,
            record_gradient_diagnostics=False,
        )
        val_metrics = summarize_predictions(val_predictions, target_mu=target_mu, target_sigma=target_sigma)
        epoch_time_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_seconds": epoch_time_seconds,
            "walltime_seconds": time.time() - start_time,
            **train_losses,
            **train_eval_loss_fields,
            "train_swe_rmse": train_metrics["swe_rmse"],
            "train_swe_mae": train_metrics["swe_mae"],
            "train_swe_r2": train_metrics["swe_r2"],
            "train_swe_pearson_r": train_metrics["swe_pearson_r"],
            "train_swe_spearman_rho": train_metrics["swe_spearman_rho"],
            "train_swe_wet_dry_accuracy": train_metrics["swe_wet_dry_accuracy"],
            "train_cpm_mse": train_metrics["cpm_mse"],
            "train_cpm_pearson_r": train_metrics["cpm_pearson_r"],
            "train_aqm_mse": train_metrics["aqm_mse"],
            "train_aqm_pearson_r": train_metrics["aqm_pearson_r"],
            "val_total_loss": val_losses["total_loss"],
            "val_swe_loss": val_losses["swe_loss"],
            "val_cpm_loss": val_losses["cpm_loss"],
            "val_aqm_loss": val_losses["aqm_loss"],
            **val_metrics,
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
                if device.type == "cuda"
                else 0.0
            ),
        }
        row.update(train_gradient_summary)
        row.update(model.extra_history_metrics())
        history_rows.append(row)
        _log(
            f"EPOCH COMPLETE epoch={epoch} train_total_loss={train_losses['total_loss']:.6f} "
            f"val_total_loss={val_losses['total_loss']:.6f} elapsed_seconds={epoch_time_seconds:.2f}"
        )

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_total_loss": best_val_loss,
                },
                output_dir / "best_checkpoint.pt",
            )
            with (output_dir / "best_validation_predictions.json").open("w") as handle:
                json.dump(val_predictions, handle, indent=2)
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "config": asdict(config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_total_loss": val_losses["total_loss"],
            },
            output_dir / "last_checkpoint.pt",
        )

        if patience_counter >= config.early_stopping_patience:
            break

    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        if device.type == "cuda"
        else 0.0
    )
    best_row = next(row for row in history_rows if int(row["epoch"]) == int(best_epoch))
    summary = {
        "architecture": config.architecture,
        "uses_auxiliary_supervision": architecture_uses_auxiliary(config.architecture),
        "uses_kendall_weighting": architecture_uses_kendall_weighting(config.architecture),
        "uses_pcgrad": architecture_uses_pcgrad(config.architecture),
        "uses_gradsim": architecture_uses_gradsim(config.architecture),
        "uses_swe_only_loss": swe_only_loss,
        "uses_historical_batch_replay": historical_batch_verification is not None,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val_loss,
        "best_metrics": best_metrics,
        "best_train_metrics": {
            "swe_rmse": best_row["train_swe_rmse"],
            "swe_mae": best_row["train_swe_mae"],
            "swe_r2": best_row["train_swe_r2"],
            "swe_pearson_r": best_row["train_swe_pearson_r"],
            "cpm_mse": best_row["train_cpm_mse"],
            "cpm_pearson_r": best_row["train_cpm_pearson_r"],
            "aqm_mse": best_row["train_aqm_mse"],
            "aqm_pearson_r": best_row["train_aqm_pearson_r"],
        },
        "device": str(device),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "train_rows": len(split["train"]),
        "val_rows": len(split["val"]),
        "walltime_seconds": time.time() - start_time,
        "predictor_count": int(physical_data.shape[2]),
        "mask_count": int(physical_data.shape[2]),
        "cnn_input_channels": int(physical_data.shape[1] * physical_data.shape[2] * 2),
        "split_verification_path": str(output_dir / "split_verification.json"),
    }
    if record_gradient_diagnostics:
        summary["best_gradient_diagnostics"] = {
            key: best_row[key]
            for key in best_row.keys()
            if str(key).startswith("grad_") or str(key).startswith("weighted_grad_")
        }
    if architecture_uses_kendall_weighting(config.architecture):
        summary["best_kendall_weights"] = {
            key: best_row[key]
            for key in best_row.keys()
            if str(key).startswith("s_") or str(key).startswith("w_")
        }
    if historical_batch_verification is not None:
        summary["historical_batch_order_verification"] = historical_batch_verification
    summary.update(model.extra_history_metrics())
    write_json(output_dir / "metrics_summary.json", summary)
    write_json(output_dir / "config.json", asdict(config))
    return summary
