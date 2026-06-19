"""
b_scoring.py — scoring & characterisation for the parking-intel project.

Reads violations_clean.parquet and zones_base.parquet from data/built/ and
writes:
  - zones_scored.parquet   : zones with ZRI and persistence labels
  - repeat_vehicles.parquet: vehicles with >= REPEAT_MIN tickets
"""

import pathlib

import h3
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
W_DENSITY, W_HEAVY, W_REPEAT, W_SPILL, W_JUNCTION = 0.35, 0.20, 0.15, 0.15, 0.15
REPEAT_MIN = 3       # tickets before a vehicle is a "repeat offender"
CHRONIC_MIN_DAYS = 30  # distinct active days for a zone to be "chronic"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILT_DIR = ROOT / "data" / "built"
IN_VIOLATIONS = BUILT_DIR / "violations_clean.parquet"
IN_ZONES = BUILT_DIR / "zones_base.parquet"
OUT_ZONES_SCORED = BUILT_DIR / "zones_scored.parquet"
OUT_REPEAT = BUILT_DIR / "repeat_vehicles.parquet"


# ---------------------------------------------------------------------------
# Step 1 — repeat-offender vehicles
# ---------------------------------------------------------------------------
def get_repeat_offenders(df: pd.DataFrame) -> set:
    """Return set of vehicle_number values with >= REPEAT_MIN tickets."""
    clean = df[df["vehicle_number"].notna() & (df["vehicle_number"].str.strip() != "")]
    counts = clean["vehicle_number"].value_counts()
    return set(counts[counts >= REPEAT_MIN].index)


# ---------------------------------------------------------------------------
# Step 2 — per-zone feature components
# ---------------------------------------------------------------------------
def compute_zone_components(df: pd.DataFrame, repeat_offenders: set) -> pd.DataFrame:
    """Return a DataFrame indexed by h3_index with all five raw components."""

    # --- heavy_mix ----------------------------------------------------------
    heavy_mix = df.groupby("h3_index")["is_heavy"].mean().rename("heavy_mix")

    # --- repeat_rate --------------------------------------------------------
    df["_is_repeat"] = df["vehicle_number"].isin(repeat_offenders)
    repeat_rate = df.groupby("h3_index")["_is_repeat"].mean().rename("repeat_rate")

    # --- junction_prox ------------------------------------------------------
    df["_has_junction"] = (
        df["junction_name"].notna()
        & (df["junction_name"].str.strip() != "")
        & (df["junction_name"].str.strip() != "No Junction")
    )
    junction_prox = df.groupby("h3_index")["_has_junction"].mean().rename("junction_prox")

    # --- spillover ----------------------------------------------------------
    count_map: dict = df.groupby("h3_index").size().to_dict()

    def _spillover(cell: str) -> float:
        neighbours = set(h3.grid_disk(cell, 1)) - {cell}
        neighbour_sum = sum(count_map.get(n, 0) for n in neighbours)
        centre = count_map.get(cell, 0)
        return neighbour_sum / (centre + neighbour_sum + 1)

    spillover_series = pd.Series(
        {cell: _spillover(cell) for cell in count_map},
        name="spillover",
    )
    spillover_series.index.name = "h3_index"

    # --- active_days (for persistence label) --------------------------------
    df["_date"] = df["ts_ist"].dt.date
    active_days = (
        df.groupby("h3_index")["_date"]
        .nunique()
        .rename("active_days")
    )

    components = pd.concat(
        [heavy_mix, repeat_rate, junction_prox, spillover_series, active_days],
        axis=1,
    )
    return components.reset_index()  # promotes h3_index from index to column


# ---------------------------------------------------------------------------
# Step 3 — density_norm
# ---------------------------------------------------------------------------
def add_density_norm(zones: pd.DataFrame) -> pd.DataFrame:
    log_count = np.log1p(zones["violation_count"])
    lo, hi = log_count.min(), log_count.max()
    zones["density_norm"] = (log_count - lo) / (hi - lo) if hi > lo else 0.0
    return zones


# ---------------------------------------------------------------------------
# Step 4 — ZRI
# ---------------------------------------------------------------------------
def compute_zri(zones: pd.DataFrame) -> pd.Series:
    return 100 * (
        W_DENSITY  * zones["density_norm"]
        + W_HEAVY  * zones["heavy_mix"]
        + W_REPEAT * zones["repeat_rate"]
        + W_SPILL  * zones["spillover"]
        + W_JUNCTION * zones["junction_prox"]
    )


# ---------------------------------------------------------------------------
# Step 6 — repeat_vehicles table
# ---------------------------------------------------------------------------
def build_repeat_vehicles(df: pd.DataFrame, repeat_offenders: set) -> pd.DataFrame:
    sub = df[df["vehicle_number"].isin(repeat_offenders)].copy()

    agg = sub.groupby("vehicle_number").agg(
        total_count=("vehicle_number", "count"),
        zones_visited=("h3_index", "nunique"),
        first_seen=("ts_ist", "min"),
        last_seen=("ts_ist", "max"),
    ).reset_index()

    agg["recidivist_score"] = agg["total_count"]

    def _fine_tier(n: int) -> int:
        if n <= 5:
            return 1
        if n <= 10:
            return 2
        return 3

    agg["fine_tier"] = agg["total_count"].apply(_fine_tier)

    return agg.sort_values("total_count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Reading {IN_VIOLATIONS} ...")
    df = pd.read_parquet(IN_VIOLATIONS)

    print(f"Reading {IN_ZONES} ...")
    zones = pd.read_parquet(IN_ZONES).copy()

    # Step 1
    repeat_offenders = get_repeat_offenders(df)
    print(f"  Repeat offenders (>= {REPEAT_MIN} tickets): {len(repeat_offenders):,}")

    # Step 2
    print("Computing per-zone components ...")
    components = compute_zone_components(df, repeat_offenders)

    # Merge components onto zones (left join so every zone is kept)
    zones = zones.merge(components, on="h3_index", how="left")
    zones[["heavy_mix", "repeat_rate", "junction_prox", "spillover"]] = (
        zones[["heavy_mix", "repeat_rate", "junction_prox", "spillover"]].fillna(0.0)
    )
    zones["active_days"] = zones["active_days"].fillna(0).astype(int)

    # Step 3 — density_norm
    zones = add_density_norm(zones)

    # Step 3 — ZRI
    zones["zri"] = compute_zri(zones).round(4)

    # Step 4 — persistence_label
    zones["persistence_label"] = zones["active_days"].apply(
        lambda d: "chronic" if d >= CHRONIC_MIN_DAYS else "sporadic"
    )

    # Step 5 — zones_scored: select and order final columns
    zones_scored = zones[[
        "h3_index", "centroid_lat", "centroid_lon",
        "violation_count", "example_address",
        "density_norm", "heavy_mix", "repeat_rate",
        "junction_prox", "spillover",
        "zri", "persistence_label",
    ]].sort_values("zri", ascending=False).reset_index(drop=True)

    # Step 6 — repeat vehicles
    repeat_veh = build_repeat_vehicles(df, repeat_offenders)

    # Write outputs
    zones_scored.to_parquet(OUT_ZONES_SCORED, index=False)
    repeat_veh.to_parquet(OUT_REPEAT, index=False)
    print(f"  Written -> {OUT_ZONES_SCORED}")
    print(f"  Written -> {OUT_REPEAT}")

    # Summary
    n_chronic = (zones_scored["persistence_label"] == "chronic").sum()
    n_sporadic = (zones_scored["persistence_label"] == "sporadic").sum()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Zones scored          : {len(zones_scored):,}")
    print(f"  Chronic zones         : {n_chronic:,}  (>= {CHRONIC_MIN_DAYS} active days)")
    print(f"  Sporadic zones        : {n_sporadic:,}")
    print(f"  Repeat-offender vehicles : {len(repeat_veh):,}")

    print("\n  Top 10 zones by ZRI:")
    print(f"  {'h3_index':<20} {'zri':>6}  {'violations':>10}  {'label'}")
    print("  " + "-" * 54)
    for _, row in zones_scored.head(10).iterrows():
        print(
            f"  {row['h3_index']:<20} {row['zri']:>6.2f}  "
            f"{row['violation_count']:>10,}  {row['persistence_label']}"
        )

    print("\n  Top 5 vehicles by ticket count:")
    print(f"  {'vehicle_number':<20} {'count':>6}  {'zones':>6}  {'tier':>5}")
    print("  " + "-" * 44)
    for _, row in repeat_veh.head(5).iterrows():
        print(
            f"  {row['vehicle_number']:<20} {row['total_count']:>6,}  "
            f"{row['zones_visited']:>6}  {row['fine_tier']:>5}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
