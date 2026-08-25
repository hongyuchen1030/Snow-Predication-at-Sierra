from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from snow_ml.cmip6_cnn_experiment import build_model, set_global_seed


EXPECTED_ORIGINAL_DIGESTS = (
    "5a0b619d2713d206d94cb638fad05586727688143cb3237f4ede45112dd46c60",
    "fe05b836475bebd1007fe60df5fd57089ac695a4ed60b2c59d33f074c12c3d91",
    "df16edef9b7e41120d18a83140e425ab6dd3bdf6d93f9daa5e2a5f69c5aa1474",
)


def digest(batches: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(batches, separators=(",", ":")).encode()).hexdigest()


def collect_orders(*, split: dict[str, list[int]], corrected: bool) -> list[list[list[int]]]:
    set_global_seed(20260813)
    model = build_model("S0_static_cnn_swe_only", in_channels_per_month=38)
    del model
    train_loader = DataLoader(split["train"], batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    train_eval_loader = (
        DataLoader(split["train"], batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
        if corrected
        else None
    )
    val_loader = DataLoader(split["val"], batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    orders: list[list[list[int]]] = []
    for _ in range(3):
        orders.append([list(map(int, batch.tolist())) for batch in train_loader])
        if train_eval_loader is not None:
            for _ in train_eval_loader:
                pass
        for _ in val_loader:
            pass
    return orders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split = json.loads(Path(args.split_json).read_text())
    original = collect_orders(split=split, corrected=False)
    original_digests = tuple(digest(epoch) for epoch in original)
    if original_digests != EXPECTED_ORIGINAL_DIGESTS:
        raise RuntimeError(f"Reconstructed S0 audit differs from the verified audit: {original_digests}")
    corrected = collect_orders(split=split, corrected=True)
    output = {
        "seed": 20260813,
        "batch_size": 2,
        "train_row_count": len(split["train"]),
        "val_row_count": len(split["val"]),
        "architectures": {
            "S0_static_cnn_swe_only": {
                "original": original,
                "corrected": corrected,
                "original_sha256": list(original_digests),
                "corrected_sha256": [digest(epoch) for epoch in corrected],
                "epoch_orders_identical": [old == new for old, new in zip(original, corrected, strict=True)],
            }
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
