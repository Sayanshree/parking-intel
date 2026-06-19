"""
c_time.py — time analysis & burst detection for the parking-intel project.

Reads violations_clean.parquet and zones_scored.parquet from data/built/ and
writes:
  - zone_time.parquet  : 24x7 ticket counts per zone
  - zone_peaks.parquet : top-3 peak (dow, hour) windows per zone
  - bursts.parquet     : burst events (>=BURST_MIN tickets in BURST_BIN window)
"""

import pathlib

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BURST_BIN = "30min"
BURST_MIN = 5
TOP_PEAKS = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILT_DIR = ROOT / "data" / "built"
IN_VIOLATIONS = BUILT_DIR / "violations_clean.parquet"
IN_ZONES_SCORED = BUILT_DIR / "zones_scored.parquet"
OUT_ZONE_TIME = BUILT_DIR / "zone_time.parquet"
OUT_ZONE_PEAKS = BUILT_DIR / "zone_peaks.parquet"
OUT_BURSTS = BUILT_DIR / "bursts.parquet"

# Canonical weekday order for sorting
_DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_DOW_RANK = {d: i for i, d in enumerate(_DOW_ORDER)}


# ---------------------------------------------------------------------------
# Step 1 — 24x7 grid
# ---------------------------------------------------------------------------
def build_zone_time(df: pd.DataFrame) -> pd.DataFrame:
    grid = (
        df.groupby(["h3_index", "dow", "hour"])
        .size()
        .reset_index(name="count")
    )
    # keep only nonzero (groupby already does this, but explicit is safer)
    grid = grid[grid["count"] > 0].copy()
    return grid


# ---------------------------------------------------------------------------
# Step 2 — top-N peak windows per zone
# ---------------------------------------------------------------------------
def build_zone_peaks(zone_time: pd.DataFrame) -> pd.DataFrame:
    # Sort descending within each zone, then take top TOP_PEAKS rows
    zone_time = zone_time.copy()
    zone_time["_dow_rank"] = zone_time["dow"].map(_DOW_RANK)
    zone_time = zone_time.sort_values(
        ["h3_index", "count", "_dow_rank", "hour"],
        ascending=[True, False, True, True],
    )

    peaks = (
        zone_time.groupby("h3_index", sort=False)
        .head(TOP_PEAKS)
        .copy()
    )
    peaks["rank"] = peaks.groupby("h3_index").cumcount() + 1
    peaks = peaks[["h3_index", "rank", "dow", "hour", "count"]].reset_index(drop=True)
    return peaks


# ---------------------------------------------------------------------------
# Step 3 — burst detection
# ---------------------------------------------------------------------------
def build_bursts(df: pd.DataFrame) -> pd.DataFrame:
    # Mark rows whose junction_name counts as "near junction"
    df = df.copy()
    df["_near_jct"] = (
        df["junction_name"].notna()
        & (df["junction_name"].str.strip() != "")
        & (df["junction_name"].str.strip() != "No Junction")
    )

    burst_records = []

    for zone, grp in df.groupby("h3_index"):
        # Set tz-aware datetime index; resample needs a DatetimeIndex
        grp = grp.set_index("ts_ist").sort_index()

        # Resample into bins: count tickets and OR the near-junction flag
        counts = grp["id"].resample(BURST_BIN).count()
        # .max() on a bool column is equivalent to .any() per bin
        near_jct = grp["_near_jct"].resample(BURST_BIN).max()

        burst_bins = counts[counts >= BURST_MIN]
        for window_start, cnt in burst_bins.items():
            burst_records.append({
                "h3_index": zone,
                "window_start": window_start,
                "count": int(cnt),
                "near_junction": bool(near_jct.loc[window_start]),
            })

    bursts = pd.DataFrame(
        burst_records,
        columns=["h3_index", "window_start", "count", "near_junction"],
    )
    return bursts.sort_values(["h3_index", "window_start"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4 — city-wide hourly insight (print only)
# ---------------------------------------------------------------------------
def print_city_hourly(df: pd.DataFrame) -> None:
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)

    print("\n  City-wide violations by hour (IST):")
    print(f"  {'Hour':>5}  {'Count':>8}")
    print("  " + "-" * 16)
    for h, cnt in hourly.items():
        bar = "#" * min(40, cnt // 200)
        print(f"  {h:>5}  {cnt:>8,}  {bar}")

    busiest_hour = int(hourly.idxmax())
    evening_total = int(hourly.loc[18:21].sum())
    print(f"\n  Busiest hour    : {busiest_hour:02d}:00  ({hourly[busiest_hour]:,} tickets)")
    print(f"  Evening 18-21h  : {evening_total:,} tickets")
    print(
        "\n  Note: these counts reflect when officers issue tickets, "
        "not necessarily when illegal parking occurs."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Reading {IN_VIOLATIONS} ...")
    df = pd.read_parquet(IN_VIOLATIONS)

    print(f"Reading {IN_ZONES_SCORED} ...")
    zones_scored = pd.read_parquet(IN_ZONES_SCORED)

    # Step 1
    print("Building 24x7 grid ...")
    zone_time = build_zone_time(df)
    zone_time.to_parquet(OUT_ZONE_TIME, index=False)
    print(f"  Written -> {OUT_ZONE_TIME}")

    # Step 2
    print("Computing peak windows ...")
    zone_peaks = build_zone_peaks(zone_time)
    zone_peaks.to_parquet(OUT_ZONE_PEAKS, index=False)
    print(f"  Written -> {OUT_ZONE_PEAKS}")

    # Step 3
    print("Detecting bursts ...")
    bursts = build_bursts(df)
    bursts.to_parquet(OUT_BURSTS, index=False)
    print(f"  Written -> {OUT_BURSTS}")

    # Step 4
    print_city_hourly(df)

    # Summary
    top_burst_zones = (
        bursts.groupby("h3_index")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    top_zri_zone = zones_scored.sort_values("zri", ascending=False).iloc[0]["h3_index"]
    top_zone_peaks = (
        zone_peaks[zone_peaks["h3_index"] == top_zri_zone]
        .sort_values("rank")
    )

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Zones with a time grid : {zone_time['h3_index'].nunique():,}")
    print(f"  Total burst events     : {len(bursts):,}")

    print("\n  Top 5 zones by burst count:")
    print(f"  {'h3_index':<22} {'burst tickets':>14}")
    print("  " + "-" * 37)
    for zone, total in top_burst_zones.items():
        print(f"  {zone:<22} {total:>14,}")

    print(f"\n  Top-ZRI zone: {top_zri_zone}")
    print(f"  Peak windows (Mon..Sun order):")
    print(f"  {'rank':>4}  {'dow':>4}  {'hour':>5}  {'count':>8}")
    print("  " + "-" * 28)
    for _, row in top_zone_peaks.iterrows():
        print(f"  {int(row['rank']):>4}  {row['dow']:>4}  {int(row['hour']):>5}  {int(row['count']):>8,}")
    print("=" * 64)


if __name__ == "__main__":
    main()
