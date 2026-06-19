"""
a_foundation.py — data foundation for the parking-intel project.

Reads data/raw/violations.csv and writes two parquet files to data/built/:
  - violations_clean.parquet : one row per cleaned ticket
  - zones_base.parquet       : one row per H3 hex zone with aggregated stats
"""

import ast
import os
import pathlib

import h3
import pandas as pd

# ---------------------------------------------------------------------------
# Config — easy to change without touching the logic
# ---------------------------------------------------------------------------
RESOLUTION = 9       # H3 hex resolution (~170 m cells)
ONLY_APPROVED = False  # if True, keep only validation_status == 'approved'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "violations.csv"
BUILT_DIR = ROOT / "data" / "built"

# ---------------------------------------------------------------------------
# Vehicle-type → is_heavy mapping
# ---------------------------------------------------------------------------
# Keys are lowercase; anything not listed defaults to True (treat as heavy)
_LIGHT = {
    "two wheeler", "2 wheeler", "twowheeler",
    "scooter", "motorcycle", "moped", "bike", "bicycle",
    "auto", "auto rickshaw", "autorickshaw", "e-rickshaw", "e rickshaw",
    "e-scooter", "e scooter", "e-bike", "e bike",
}

_HEAVY = {
    "car", "van", "truck", "lorry", "tanker", "bus",
    "maxi-cab", "maxi cab", "maxicab", "cab",
    "tempo", "mini truck", "mini-truck", "minitruck",
    "tractor", "jeep", "suv", "pickup",
    "ambulance", "fire truck", "police vehicle",
}

def _is_heavy(vtype: str) -> bool:
    """Return True for heavy vehicles; False for two-wheelers / autos."""
    key = str(vtype).lower().strip()
    if key in _LIGHT:
        return False
    if key in _HEAVY:
        return True
    # Heuristic fallback: names containing these substrings are light
    for substr in ("wheeler", "scooter", "moped", "bike", "cycle", "auto"):
        if substr in key:
            return False
    return True  # unknown → default to heavy


def _parse_violation_list(raw: str) -> list:
    try:
        result = ast.literal_eval(str(raw))
        if isinstance(result, list):
            return result
        return []
    except Exception:
        return []


def _resolve_vehicle_type(row) -> str:
    """Prefer updated_vehicle_type when present and not the string 'NULL'."""
    updated = row["updated_vehicle_type"]
    if pd.notna(updated) and str(updated).strip().upper() != "NULL" and str(updated).strip() != "":
        return str(updated).strip()
    return str(row["vehicle_type"]).strip() if pd.notna(row["vehicle_type"]) else ""


def build_violations_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # --- lat / lon: drop rows where either is missing or non-numeric -------
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    # --- timestamp: parse as UTC, convert to IST ----------------------------
    df["ts_utc"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"])
    df["ts_ist"] = df["ts_utc"].dt.tz_convert("Asia/Kolkata")
    df["hour"] = df["ts_ist"].dt.hour
    df["dow"] = df["ts_ist"].dt.day_name().str[:3]  # Mon, Tue, …

    # --- vehicle type -------------------------------------------------------
    df["vehicle_type_resolved"] = df.apply(_resolve_vehicle_type, axis=1)

    # Print unique values so the caller can verify the mapping
    print("\nUnique vehicle_type values (resolved):")
    unique_types = sorted(df["vehicle_type_resolved"].str.upper().unique())
    for vt in unique_types:
        print(f"  {vt}")
    print()

    df["is_heavy"] = df["vehicle_type_resolved"].apply(_is_heavy)

    # --- violation list -----------------------------------------------------
    df["violation_list"] = df["violation_type"].apply(_parse_violation_list)

    # --- h3 index -----------------------------------------------------------
    df["h3_index"] = df.apply(
        lambda r: h3.latlng_to_cell(r["lat"], r["lon"], RESOLUTION), axis=1
    )

    # --- optional filter ----------------------------------------------------
    if ONLY_APPROVED:
        df = df[df["validation_status"] == "approved"]

    # --- select & rename output columns ------------------------------------
    out = df[[
        "id",
        "lat",
        "lon",
        "ts_ist",
        "hour",
        "dow",
        "vehicle_number",
        "vehicle_type_resolved",
        "violation_list",
        "is_heavy",
        "junction_name",
        "police_station",
        "validation_status",
        "h3_index",
        "location",          # carried through for zones_base
    ]].rename(columns={"vehicle_type_resolved": "vehicle_type"})

    return out


def build_zones_base(df_clean: pd.DataFrame) -> pd.DataFrame:
    grp = df_clean.groupby("h3_index")

    zones = pd.DataFrame({
        "h3_index": grp.size().index,
        "violation_count": grp.size().values,
        "example_address": grp["location"].first().values,
    })

    # Derive centroids from H3
    def _centroid(cell):
        lat, lon = h3.cell_to_latlng(cell)
        return pd.Series({"centroid_lat": lat, "centroid_lon": lon})

    centroids = zones["h3_index"].apply(_centroid)
    zones = pd.concat([zones, centroids], axis=1)

    zones = zones[["h3_index", "centroid_lat", "centroid_lon",
                   "violation_count", "example_address"]]
    return zones.sort_values("violation_count", ascending=False).reset_index(drop=True)


def main():
    BUILT_DIR.mkdir(parents=True, exist_ok=True)

    # --- load ---------------------------------------------------------------
    print(f"Loading {RAW_CSV} …")
    df_raw = pd.read_csv(RAW_CSV, low_memory=False)
    rows_in = len(df_raw)
    print(f"  Rows loaded : {rows_in:,}")

    # --- clean --------------------------------------------------------------
    print("Cleaning …")
    df_clean = build_violations_clean(df_raw)
    rows_kept = len(df_clean)

    # --- zones --------------------------------------------------------------
    print("Aggregating zones …")
    df_zones = build_zones_base(df_clean)

    # --- write parquet (drop 'location' col from final violations file) ----
    violations_out = df_clean.drop(columns=["location"])
    out_violations = BUILT_DIR / "violations_clean.parquet"
    out_zones = BUILT_DIR / "zones_base.parquet"

    violations_out.to_parquet(out_violations, index=False)
    df_zones.to_parquet(out_zones, index=False)
    print(f"  Written -> {out_violations}")
    print(f"  Written -> {out_zones}")

    # --- summary ------------------------------------------------------------
    date_min = df_clean["ts_ist"].min()
    date_max = df_clean["ts_ist"].max()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Rows in          : {rows_in:,}")
    print(f"  Rows kept        : {rows_kept:,}  ({rows_kept/rows_in*100:.1f}%)")
    print(f"  Date range (IST) : {date_min:%Y-%m-%d %H:%M} to {date_max:%Y-%m-%d %H:%M}")
    print(f"  Unique H3 zones  : {len(df_zones):,}")
    print("\n  Top 5 zones by violation count:")
    for _, row in df_zones.head(5).iterrows():
        print(f"    {row['h3_index']}  count={row['violation_count']:,}  "
              f"({row['centroid_lat']:.5f}, {row['centroid_lon']:.5f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
