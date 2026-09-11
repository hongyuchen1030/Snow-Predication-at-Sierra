#!/usr/bin/env python3
"""Create non-executable manifests for the controlled ACE5 Snow-17 A/B test.

These manifests intentionally contain no Snow-17 parameters or synthetic
temperature disaggregation.  They make the scientific gate machine-readable.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra_ace2/lag_ensemble_wy2016")
OUT = Path("artifacts/ace5_snow17_temperature_ab_preparation")
MEMBERS = ["20151101", "20151102", "20151103", "20151104", "20151105"]
WUS_DAILY_T2 = [
    "/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily/ec-earth3_r1i1p1f1_2_ssp370_bc/postprocess/d02/t2.daily.ec-earth3.r1i1p1f1_2.ssp370.bias-correct.d02.2015.nc",
    "/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily/miroc6_r1i1p1f1_ssp370_bc/postprocess/d02/t2.daily.miroc6.r1i1p1f1.ssp370.bias-correct.d02.2015.nc",
    "/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily/mpi-esm1-2-hr_r3i1p1f1_ssp370_bc/postprocess/d02/t2.daily.mpi-esm1-2-hr.r3i1p1f1.ssp370.bias-correct.d02.2015.nc",
    "/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily/taiesm1_r1i1p1f1_ssp370_bc/postprocess/d02/t2.daily.taiesm1.r1i1p1f1.ssp370.bias-correct.d02.2015.nc",
]


def member_entry(member: str) -> dict[str, str]:
    start = f"{member[:4]}-{member[4:6]}-{member[6:]}T00:00:00"
    return {
        "member": member,
        "initialization": start,
        "ace_output": str(ROOT / f"member_{member}" / "autoregressive_predictions.nc"),
        "precipitation": "PRATEsfc; six-hour total = PRATEsfc * 21600; kg m-2 = mm",
        "target_end": "2016-04-01T00:00:00",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common = {
        "experiment_type": "controlled temperature-only Snow-17 sensitivity preparation",
        "member_order": MEMBERS,
        "ace_members": [member_entry(member) for member in MEMBERS],
        "cnn_swe_target": "native WUS-D3 d02 valid-cell physical-area weighted April 1 SWE; geographic criterion 35-42N, 122.5-118W",
        "required_snow17_gate": [
            "Sierra-applicable calibrated Snow-17 parameter set including ADC",
            "defensible carryover/restart state aligned independently for each Nov 1-5 initialization",
            "frozen native WUS-D3-to-ACE/Snow-17 spatial intersection and physical area weights",
        ],
        "prohibited": ["NOAA tutorial/example parameters", "zero cold start", "synthetic six-hour WUS-D3 temperature cycle", "UCLA-mask target"],
        "status": "BLOCKED_BEFORE_SNOW17_EXECUTION",
    }
    branch_a = dict(common)
    branch_a.update({
        "branch": "A: ACE PRATEsfc plus same-member ACE TMP2m",
        "temperature": "TMP2m converted K to degC; no lapse/bias correction requested",
        "temperature_is_member_specific": True,
        "blockers": common["required_snow17_gate"],
    })
    branch_b = dict(common)
    branch_b.update({
        "branch": "B: identical ACE PRATEsfc plus WUS-D3 t2",
        "temperature_candidates": WUS_DAILY_T2,
        "temperature_metadata": "t2 is 2-meter temperature in K, d02 native 340x270 grid, daily cadence, WY2016 coverage Sep 1 2015 through Aug 31 2016",
        "additional_blockers": [
            "No unique WUS-D3 model/member was selected for a synthetic ACE member.",
            "Only daily t2 is available; no documented six-hourly WUS-D3 temperature product or approved temporal disaggregation exists.",
            "A native WUS-D3-to-ACE/Snow-17 spatial mapping remains to be frozen.",
        ],
        "blockers": common["required_snow17_gate"],
    })
    for name, manifest in [("snow17_experiment_A_aceT2m", branch_a), ("snow17_experiment_B_wusd3T2m", branch_b)]:
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        (directory / "SCIENTIFIC_GATE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# ACE5 Snow-17 A/B Preparation\n\n"
        "Both branches are prepared as non-executable manifests only. No Snow-17 forcing or SWE labels were generated, "
        "because the required Sierra parameter/state package is unresolved and the WUS-D3 B temperature is daily and model-ambiguous.\n"
    )


if __name__ == "__main__":
    main()
