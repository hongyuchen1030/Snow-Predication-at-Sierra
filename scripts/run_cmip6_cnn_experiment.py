from __future__ import annotations

import argparse
from pathlib import Path

from snow_ml.cmip6_cnn_experiment import ExperimentConfig, train_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        required=True,
        choices=[
            "M1_static_cnn",
            "D1_static_cnn_cpm_aqm_diag",
            "D2_static_cnn_cpm_only",
            "K1_static_cnn_kendall_aux",
            "P1_static_cnn_pcgrad",
            "P2_static_cnn_gradsim",
            "U1_stage2_cpm_aqm",
            "M7_static_cnn_swe_only",
            "S0_static_cnn_swe_only",
            "S1_static_latent_self_attention",
            "S2_static_swe_token",
            "S3_static_residual_gated_attention",
            "M2_temporal_conv",
            "M3_convgru",
            "M4_attention_swe",
            "M5_static_attention_swe_only",
            "M6_static_attention_with_aux",
        ],
    )
    parser.add_argument(
        "--predictor-cache-dir",
        default="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/data_cache",
    )
    parser.add_argument(
        "--output-root",
        default="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_cnn_architecture_screen_v1/experiments",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--historical-batch-order-audit", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / args.architecture
    config = ExperimentConfig(
        architecture=args.architecture,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=not args.disable_amp,
        predictor_cache_dir=args.predictor_cache_dir,
        output_dir=str(output_dir),
        split_json_path=args.split_json,
        split_manifest_path=args.split_manifest,
        historical_batch_order_audit_path=args.historical_batch_order_audit,
    )
    summary = train_experiment(config)
    print(summary)


if __name__ == "__main__":
    main()
