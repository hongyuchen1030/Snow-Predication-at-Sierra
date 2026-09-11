"""Shared Dataset for the 7-day -> 1-day Sierra SWE-change task.

Lazy/mmap-based: no 7-day window tensors are materialized on disk. Each
sample is assembled at __getitem__ time from mmap'd yearly predictor arrays,
with a small per-worker cache of open mmaps to avoid repeatedly reopening the
same yearly files (consecutive target days share 6 of their 7 history days).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PREDICTOR_ROOT = Path("/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/taiesm1_daily_12var_1day_v1")
N_HISTORY_DAYS = 7
N_CHANNELS = 12
GRID_H = 120
GRID_W = 240


def in_season_mask(month: np.ndarray, day: np.ndarray, start_month: int) -> np.ndarray:
    fall = (month >= start_month) & (month <= 12)
    winter = (month >= 1) & (month <= 3)
    apr1 = (month == 4) & (day == 1)
    return fall | winter | apr1


class DeltaSWEDataset(Dataset):
    def __init__(
        self,
        *,
        index_path: Path,
        water_years: set[int],
        season_start_month: int,
        sample_mode: str,  # "all" or "snow_active"
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mu: float,
        target_sigma: float,
        predictor_root: Path = PREDICTOR_ROOT,
    ) -> None:
        assert sample_mode in ("all", "snow_active")
        idx = np.load(index_path, allow_pickle=True)
        water_year = idx["water_year"]
        cal_month = idx["cal_month"]
        cal_day = idx["cal_day"]
        active = idx["active"]

        mask = np.isin(water_year, list(water_years)) & in_season_mask(cal_month, cal_day, season_start_month)
        if sample_mode == "snow_active":
            mask &= active
        self.rows = np.where(mask)[0]

        self.water_year = water_year[self.rows]
        self.cal_year = idx["cal_year"][self.rows]
        self.cal_month = cal_month[self.rows]
        self.cal_day = cal_day[self.rows]
        self.experiment_t = idx["experiment_t"][self.rows]
        self.swe_t = idx["swe_t"][self.rows]
        self.swe_t1 = idx["swe_t1"][self.rows]
        self.delta_swe = idx["delta_swe"][self.rows]
        self.active = active[self.rows]
        self.history_year = idx["history_year"][self.rows]
        self.history_doy = idx["history_doy"][self.rows]
        self.history_exp = idx["history_exp"][self.rows]
        self.exp_code_map = json.loads(str(idx["exp_code_map"]))
        self.code_to_exp = {v: k for k, v in self.exp_code_map.items()}

        self.feature_mean = feature_mean.astype(np.float32)  # [12,120,240]
        self.feature_std = feature_std.astype(np.float32)
        self.target_mu = float(target_mu)
        self.target_sigma = float(target_sigma)
        self.predictor_root = predictor_root
        self._mmap_cache: dict[tuple[str, int], np.memmap] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _get_year_array(self, exp_name: str, year: int) -> np.memmap:
        key = (exp_name, year)
        arr = self._mmap_cache.get(key)
        if arr is None:
            path = self.predictor_root / "years" / exp_name / f"predictors_physical_{year}.npy"
            arr = np.load(path, mmap_mode="r")
            if len(self._mmap_cache) > 8:  # small LRU-ish cap per worker
                self._mmap_cache.pop(next(iter(self._mmap_cache)))
            self._mmap_cache[key] = arr
        return arr

    def __getitem__(self, i: int) -> dict:
        history = np.empty((N_HISTORY_DAYS, N_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        for k in range(N_HISTORY_DAYS):
            exp_name = self.code_to_exp[int(self.history_exp[i, k])]
            year = int(self.history_year[i, k])
            doy = int(self.history_doy[i, k])
            arr = self._get_year_array(exp_name, year)
            history[k] = np.asarray(arr[doy - 1], dtype=np.float32)

        history = (history - self.feature_mean[None]) / self.feature_std[None]
        target_norm = (self.delta_swe[i] - self.target_mu) / self.target_sigma

        return {
            "x": torch.from_numpy(history),  # [7,12,120,240]
            "y": torch.tensor(target_norm, dtype=torch.float32),
            "delta_swe_mm": torch.tensor(self.delta_swe[i], dtype=torch.float32),
            "swe_t": torch.tensor(self.swe_t[i], dtype=torch.float32),
            "swe_t1": torch.tensor(self.swe_t1[i], dtype=torch.float32),
            "active": torch.tensor(self.active[i], dtype=torch.bool),
            "water_year": int(self.water_year[i]),
            "cal_year": int(self.cal_year[i]),
            "cal_month": int(self.cal_month[i]),
            "cal_day": int(self.cal_day[i]),
        }


def collate_delta_swe(batch: list[dict]) -> dict:
    out = {}
    for key in ("x", "y", "delta_swe_mm", "swe_t", "swe_t1", "active"):
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    for key in ("water_year", "cal_year", "cal_month", "cal_day"):
        out[key] = torch.tensor([b[key] for b in batch], dtype=torch.int32)
    return out
