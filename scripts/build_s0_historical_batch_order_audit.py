from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from snow_ml.cmip6_cnn_experiment import build_model, set_global_seed


EXPECTED_ORIGINAL_DIGESTS = {
    "S0_static_cnn_swe_only": (
        "5a0b619d2713d206d94cb638fad05586727688143cb3237f4ede45112dd46c60",
        "fe05b836475bebd1007fe60df5fd57089ac695a4ed60b2c59d33f074c12c3d91",
        "df16edef9b7e41120d18a83140e425ab6dd3bdf6d93f9daa5e2a5f69c5aa1474",
    ),
    "S1_static_latent_self_attention": (
        "a1153bb5426ab43fd36874b3d52759e3ebcbe76d40a58938f4ac943c1252ccde",
        "95d9bd5e04d0588c98991b4092c95127565a7524112a2cb2cacabecc99522c46",
        "431058811aab84acfb3e3152d6cdf801a772ae72871fcbe6f588e5730708860d",
    ),
    "S2_static_swe_token": (
        "d009ee46814a31a3b3a6caee6087486dd31351f9c05661dbe047d238cae5e7aa",
        "e4c457bb731a083a4d8438ed29556a9c34e2b307e96507382efa6182619d85bf",
        "ba6868a8555fc0d070c35c5aef80a273c9b5099adc2949759554320a4b88ea07",
    ),
    "S3_static_residual_gated_attention": (
        "57e9db514268e195e536b50ddc0bc6055edfd9481ceb44aef727de1dfca7e63e",
        "d8b1ea0d4941e32aa81efca8fa89aa08019abbc08c1a91ed5566302d04849a46",
        "1b645b8f22cb5a835f6157df75fbf38ce820f401347dea2fff25b28673eb5621",
    ),
}


def digest(batches: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(batches, separators=(",", ":")).encode()).hexdigest()


def collect_orders(*, architecture: str, split: dict[str, list[int]], corrected: bool) -> list[list[list[int]]]:
    set_global_seed(20260813)
    model = build_model(architecture, in_channels_per_month=38)
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
    parser.add_argument("--split-root", required=True, help="Historical experiment root containing S0-S3 split.json files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split_root = Path(args.split_root)
    architectures: dict[str, dict[str, object]] = {}
    for architecture, expected_digests in EXPECTED_ORIGINAL_DIGESTS.items():
        split = json.loads((split_root / architecture / "split.json").read_text())
        original = collect_orders(architecture=architecture, split=split, corrected=False)
        original_digests = tuple(digest(epoch) for epoch in original)
        if original_digests != expected_digests:
            raise RuntimeError(
                f"Reconstructed {architecture} audit differs from the verified audit: {original_digests}"
            )
        corrected = collect_orders(architecture=architecture, split=split, corrected=True)
        architectures[architecture] = {
            "original": original,
            "corrected": corrected,
            "original_sha256": list(original_digests),
            "corrected_sha256": [digest(epoch) for epoch in corrected],
            "epoch_orders_identical": [old == new for old, new in zip(original, corrected, strict=True)],
        }
    output = {
        "seed": 20260813,
        "batch_size": 2,
        "train_row_count": len(json.loads((split_root / "S0_static_cnn_swe_only" / "split.json").read_text())["train"]),
        "val_row_count": len(json.loads((split_root / "S0_static_cnn_swe_only" / "split.json").read_text())["val"]),
        "architectures": architectures,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
