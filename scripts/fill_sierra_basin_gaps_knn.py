"""
Fill Category-B (Sierra-interior, HUC8-keyword-miss) cells in the seed
North/Central/South basin assignment using spatial K-nearest-neighbor
majority voting over already-assigned seed cells.

Category-A cells (outside the intended Sierra geographic region) are left
untouched (NaN). The 5 HUC8 polygon names below are used ONLY to reproduce
the already-diagnosed and user-confirmed Category-B eligibility set (4,199
cells) -- they are NOT used to decide the North/Central/South label. The
label is decided purely by KNN majority vote among already-assigned
neighbors, with zero reference to HUC8 identity.
"""
import sys
import os
import json
from pathlib import Path
from collections import Counter

REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / ".python_vendor"))
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.geometry import Point

import plot_sierra_swe_huc8_basin_assignment_test as seedmod

SEED_NPZ = (
    "/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/artifacts/home_migrated_20260828/"
    "swe_target_spatial_diagnostic/basin_assignment_grid_wy2021.npz"
)
OUT_DIR = REPO_ROOT / "artifacts" / "sierra_basin_assignment_knn_filled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_NEIGHBORS = 16

# Used only to reproduce the already-diagnosed, user-confirmed eligibility
# set of 4,199 Category-B cells. Not used to decide the assigned label.
CATEGORY_B_HUC8_NAMES = {
    "Crowley Lake",
    "Fresno River",
    "Upper Poso",
    "Upper Deer-Upper White",
    "Upper Bear",
}

GROUP_LABEL = {1: "North", 2: "Central", 3: "South"}


def log(msg):
    print(msg, flush=True)


def main():
    seed = np.load(SEED_NPZ, allow_pickle=True)
    lat = seed["lat"]
    lon = seed["lon"]
    assignment_seed = seed["assignment"].astype(np.float32).copy()
    valid_sierra = seed["valid_sierra_grid"]
    valid_footprint = seed["valid_swe_footprint"]

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    seed_assigned_mask = np.isfinite(assignment_seed)
    log(f"Seed assigned cells: {int(seed_assigned_mask.sum())} "
        f"(North={int((assignment_seed==1).sum())}, "
        f"Central={int((assignment_seed==2).sum())}, "
        f"South={int((assignment_seed==3).sum())})")

    unassigned_mask = (~seed_assigned_mask) & (valid_sierra == 1) & (valid_footprint == 1)
    n_unassigned_total = int(unassigned_mask.sum())
    log(f"Total unassigned active-SWE-footprint cells (seed): {n_unassigned_total}")
    assert n_unassigned_total == 16918, f"unexpected total: {n_unassigned_total}"

    # --- re-derive the 5 target HUC8 polygons (same query/classify code as seed) ---
    wbd_cache = OUT_DIR / "wbd_huc8_query_cache.json"
    if wbd_cache.exists():
        log(f"Loading cached WBD HUC8 query result: {wbd_cache}")
        wbd = json.loads(wbd_cache.read_text())
    else:
        log("Re-querying USGS WBD HUC8 polygons (same source as the seed mask)...")
        wbd = seedmod.query_wbd_huc8_json()
        wbd_cache.write_text(json.dumps(wbd))
        log(f"Cached WBD HUC8 query result: {wbd_cache}")
    grouped = seedmod.build_grouped_huc8_features(wbd)

    target_polys = [
        f for f in grouped["feature_geoms"]
        if f["assigned_group"] == "unassigned" and f["name"] in CATEGORY_B_HUC8_NAMES
    ]
    found_names = {f["name"] for f in target_polys}
    missing = CATEGORY_B_HUC8_NAMES - found_names
    if missing:
        raise RuntimeError(f"Could not re-locate expected Category-B HUC8 polygons: {missing}")
    log(f"Re-located {len(target_polys)} Category-B HUC8 polygons: {sorted(found_names)}")

    # --- point-in-polygon test, restricted to unassigned cells only ---
    ii, jj = np.where(unassigned_mask)
    cell_huc8_name = np.full(len(ii), "", dtype=object)

    from shapely.prepared import prep
    prepared_polys = [(f["name"], prep(f["geometry"])) for f in target_polys]

    for k in range(len(ii)):
        i, j = ii[k], jj[k]
        pt = Point(float(lon[j]), float(lat[i]))
        for name, prepared in prepared_polys:
            if prepared.intersects(pt):
                cell_huc8_name[k] = name
                break

    eligible = cell_huc8_name != ""
    n_eligible = int(eligible.sum())
    log(f"Category-B eligible cells (re-derived by exact polygon test): {n_eligible}")
    assert n_eligible == 4199, f"expected 4199 Category-B cells, got {n_eligible}"

    n_category_a = n_unassigned_total - n_eligible
    log(f"Category-A cells remaining untouched: {n_category_a}")
    assert n_category_a == 12719, f"expected 12719 Category-A cells, got {n_category_a}"

    per_name_counts = Counter(cell_huc8_name[eligible])
    log("Category-B cell counts by HUC8 (re-derived, exact polygon test):")
    for name, count in sorted(per_name_counts.items(), key=lambda kv: -kv[1]):
        log(f"  {name}: {count}")

    # --- build KD-tree of ONLY already-assigned seed cells (no Category-A ever included) ---
    mean_lat_rad = np.deg2rad(np.mean(lat))
    lon_scale = np.cos(mean_lat_rad)  # approximate equirectangular correction

    assigned_lat = lat2d[seed_assigned_mask]
    assigned_lon = lon2d[seed_assigned_mask]
    assigned_labels = assignment_seed[seed_assigned_mask]

    assigned_coords = np.column_stack([assigned_lat, assigned_lon * lon_scale])
    tree = cKDTree(assigned_coords)

    eligible_ii = ii[eligible]
    eligible_jj = jj[eligible]
    eligible_full = np.zeros(assignment_seed.shape, dtype=bool)
    eligible_full[eligible_ii, eligible_jj] = True
    eligible_lat = lat[eligible_ii]
    eligible_lon = lon[eligible_jj]
    query_coords = np.column_stack([eligible_lat, eligible_lon * lon_scale])

    dist, idx = tree.query(query_coords, k=K_NEIGHBORS)

    new_assignment = assignment_seed.copy()
    n_ties = 0
    fill_labels = np.zeros(len(eligible_ii), dtype=np.float32)
    for k in range(len(eligible_ii)):
        neighbor_labels = assigned_labels[idx[k]]
        counts = Counter(neighbor_labels.tolist())
        top_count = max(counts.values())
        winners = [lbl for lbl, c in counts.items() if c == top_count]
        if len(winners) > 1:
            n_ties += 1
            # tie-break: label of the single nearest neighbor
            winner = assigned_labels[idx[k][0]]
        else:
            winner = winners[0]
        fill_labels[k] = winner
        new_assignment[eligible_ii[k], eligible_jj[k]] = winner

    log(f"KNN fill complete. Ties requiring nearest-neighbor tiebreak: {n_ties} / {n_eligible}")

    fill_counts = Counter(fill_labels.tolist())
    log("Category-B fill results by region:")
    for code in (1.0, 2.0, 3.0):
        log(f"  {GROUP_LABEL[int(code)]}: {int(fill_counts.get(code, 0))}")

    # --- per-named-cluster verification (verification only, not used to decide labels) ---
    log("\nPer-HUC8-name verification of filled labels:")
    for name in sorted(CATEGORY_B_HUC8_NAMES):
        sel = cell_huc8_name[eligible] == name
        labels_here = fill_labels[sel]
        counts_here = Counter(labels_here.tolist())
        breakdown = ", ".join(
            f"{GROUP_LABEL[int(c)]}={int(n)}" for c, n in sorted(counts_here.items())
        )
        log(f"  {name} (n={int(sel.sum())}): {breakdown}")

    # --- final checks ---
    still_nan_sierra = np.isnan(new_assignment) & (valid_sierra == 1) & (valid_footprint == 1)
    n_still_nan = int(still_nan_sierra.sum())
    log(f"\nRemaining NaN Sierra-interior-footprint cells after fill: {n_still_nan} "
        f"(should equal Category-A count 12719)")
    assert n_still_nan == 12719

    category_a_unchanged = np.all(
        np.isnan(new_assignment[unassigned_mask & (~eligible_full)])
    )
    log(f"Category-A cells confirmed still NaN: {category_a_unchanged}")

    seed_unchanged = np.array_equal(
        np.nan_to_num(assignment_seed[seed_assigned_mask], nan=-1.0),
        np.nan_to_num(new_assignment[seed_assigned_mask], nan=-1.0),
    )
    log(f"Original seed-assigned cells unchanged: {seed_unchanged}")

    # --- save new artifact (do not overwrite the seed file) ---
    out_npz = OUT_DIR / "basin_assignment_grid_wy2021_knn_filled.npz"
    np.savez(
        out_npz,
        lat=lat,
        lon=lon,
        assignment=new_assignment,
        valid_sierra_grid=valid_sierra,
        valid_swe_footprint=valid_footprint,
        seed_assignment=assignment_seed,
        category_b_filled_mask=eligible_full.astype(np.int8),
    )
    log(f"\nSaved filled mask: {out_npz}")

    import netCDF4

    out_nc = OUT_DIR / "basin_assignment_grid_wy2021_knn_filled.nc"
    with netCDF4.Dataset(out_nc, "w") as ds:
        ds.createDimension("lat", lat.shape[0])
        ds.createDimension("lon", lon.shape[0])
        lat_var = ds.createVariable("lat", "f8", ("lat",))
        lon_var = ds.createVariable("lon", "f8", ("lon",))
        lat_var[:] = lat
        lon_var[:] = lon
        assign_var = ds.createVariable("sierra_swe_region_label", "f4", ("lat", "lon"), fill_value=np.nan)
        assign_var[:, :] = new_assignment
        assign_var.description = (
            "North/Central/South Sierra SWE region label. Seed = USGS WBD HUC8 "
            "keyword-based classification (unchanged). Category-B Sierra-interior "
            "keyword-miss cells filled by K=16 nearest-neighbor majority vote over "
            "seed-assigned cells only. Category-A (outside Sierra) cells remain NaN."
        )
        assign_var.label_1 = "North"
        assign_var.label_2 = "Central"
        assign_var.label_3 = "South"
        ds.k_neighbors = K_NEIGHBORS
        ds.category_b_cells_filled = n_eligible
        ds.category_a_cells_unchanged = n_category_a
    log(f"Saved filled mask (netCDF): {out_nc}")

    summary = {
        "seed_source": SEED_NPZ,
        "method": (
            "Seed = original USGS WBD HUC8 keyword-based North/Central/South classification "
            "(unchanged). Category-B cells (Sierra-interior, HUC8-keyword-miss) filled by "
            "K-nearest-neighbor majority vote over already-assigned seed cells only. "
            "Category-A cells (outside intended Sierra region) intentionally left NaN."
        ),
        "k_neighbors": K_NEIGHBORS,
        "category_b_cells_filled": n_eligible,
        "category_b_fill_by_region": {GROUP_LABEL[int(c)]: int(n) for c, n in fill_counts.items()},
        "category_a_cells_unchanged_nan": n_category_a,
        "ties_requiring_1nn_tiebreak": n_ties,
        "category_b_by_huc8_name": {k: int(v) for k, v in per_name_counts.items()},
        "verification_by_huc8_name": {
            name: {
                GROUP_LABEL[int(c)]: int(n)
                for c, n in Counter(
                    fill_labels[cell_huc8_name[eligible] == name].tolist()
                ).items()
            }
            for name in sorted(CATEGORY_B_HUC8_NAMES)
        },
        "checks": {
            "remaining_nan_sierra_interior_cells": n_still_nan,
            "category_a_confirmed_unchanged": bool(category_a_unchanged),
            "seed_assigned_cells_confirmed_unchanged": bool(seed_unchanged),
        },
        "output_npz": str(out_npz),
        "output_nc": str(out_nc),
    }
    (OUT_DIR / "fill_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"\nWrote summary: {OUT_DIR / 'fill_summary.json'}")

    # --- before/after plot ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    colors = {1: "#1f78b4", 2: "#33a02c", 3: "#e31a1c"}
    labels = {1: "North", 2: "Central", 3: "South"}

    ax = axes[0]
    for code, color in colors.items():
        m = assignment_seed == code
        ax.scatter(lon2d[m], lat2d[m], s=0.5, color=color, label=labels[code], marker="s")
    ax.scatter(lon2d[unassigned_mask], lat2d[unassigned_mask], s=0.5, color="black",
               label=f"Unassigned (n={n_unassigned_total})", marker="s")
    ax.set_xlim(-123.0, -117.5)
    ax.set_ylim(34.5, 42.2)
    ax.set_title("BEFORE: seed HUC8-keyword assignment")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(markerscale=15, loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)

    ax2 = axes[1]
    for code, color in colors.items():
        m = new_assignment == code
        ax2.scatter(lon2d[m], lat2d[m], s=0.5, color=color, label=labels[code], marker="s")
    still_nan_mask = np.isnan(new_assignment) & (valid_sierra == 1) & (valid_footprint == 1)
    ax2.scatter(lon2d[still_nan_mask], lat2d[still_nan_mask], s=0.5, color="black",
                label=f"Still unassigned / Category-A (n={int(still_nan_mask.sum())})", marker="s")
    ax2.set_xlim(-123.0, -117.5)
    ax2.set_ylim(34.5, 42.2)
    ax2.set_title(f"AFTER: {n_eligible} Category-B cells filled by K={K_NEIGHBORS} NN majority vote")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.legend(markerscale=15, loc="upper right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Sierra North/Central/South mask: before vs after Category-B KNN fill", fontsize=13)
    fig.savefig(OUT_DIR / "before_after_knn_fill.png", dpi=200)
    log(f"Saved before/after plot: {OUT_DIR / 'before_after_knn_fill.png'}")

    # --- Crowley Lake close-up verification plot ---
    fig2, ax3 = plt.subplots(figsize=(7, 7), constrained_layout=True)
    crowley_sel = cell_huc8_name == "Crowley Lake"
    crowley_ii = ii[crowley_sel]
    crowley_jj = jj[crowley_sel]
    zoom_lat_min, zoom_lat_max = 36.6, 38.4
    zoom_lon_min, zoom_lon_max = -119.6, -117.9
    zoom = (
        (lat2d >= zoom_lat_min) & (lat2d <= zoom_lat_max) &
        (lon2d >= zoom_lon_min) & (lon2d <= zoom_lon_max)
    )
    for code, color in colors.items():
        m = (new_assignment == code) & zoom
        ax3.scatter(lon2d[m], lat2d[m], s=2.0, color=color, label=labels[code], marker="s")
    ax3.scatter(lon[crowley_jj], lat[crowley_ii], s=6.0, color="gold", edgecolor="black",
                linewidth=0.2, label=f"Crowley Lake cluster (n={int(crowley_sel.sum())})", marker="o")
    ax3.set_xlim(zoom_lon_min, zoom_lon_max)
    ax3.set_ylim(zoom_lat_min, zoom_lat_max)
    ax3.set_title("Crowley Lake cluster vs. filled surrounding region")
    ax3.legend(markerscale=6, loc="upper right", fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.4)
    fig2.savefig(OUT_DIR / "crowley_lake_verification.png", dpi=200)
    log(f"Saved Crowley Lake verification plot: {OUT_DIR / 'crowley_lake_verification.png'}")

    log("\nDONE.")


if __name__ == "__main__":
    main()
