"""
d_prescribe.py — patrol planning, officer simulation, and challan generation.

Reads zones_scored, zone_peaks, and repeat_vehicles from data/built/ and
writes:
  - patrol_plan.parquet
  - challans.parquet

The simulate() and generate_challan() functions are also importable by the
dashboard without running the script.
"""

import pathlib

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_N_PATROL = 20
ZONES_PER_OFFICER = 3
FINE_BY_TIER = {1: 500, 2: 1000, 3: 2000}   # illustrative INR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILT_DIR = ROOT / "data" / "built"
IN_ZONES_SCORED = BUILT_DIR / "zones_scored.parquet"
IN_ZONE_PEAKS = BUILT_DIR / "zone_peaks.parquet"
IN_REPEAT_VEH = BUILT_DIR / "repeat_vehicles.parquet"
OUT_PATROL = BUILT_DIR / "patrol_plan.parquet"
OUT_CHALLANS = BUILT_DIR / "challans.parquet"

# Module-level cache so dashboard imports don't re-read files on every call
_zones_scored: pd.DataFrame | None = None
_challans: pd.DataFrame | None = None


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read all three inputs; cache zones_scored for simulate()."""
    global _zones_scored, _challans
    zs = pd.read_parquet(IN_ZONES_SCORED)
    zp = pd.read_parquet(IN_ZONE_PEAKS)
    rv = pd.read_parquet(IN_REPEAT_VEH)
    _zones_scored = zs
    return zs, zp, rv


# ---------------------------------------------------------------------------
# Part 1 — patrol_plan
# ---------------------------------------------------------------------------
def build_patrol_plan(
    zones_scored: pd.DataFrame,
    zone_peaks: pd.DataFrame,
) -> pd.DataFrame:
    top = (
        zones_scored
        .sort_values("zri", ascending=False)
        .head(TOP_N_PATROL)
        .reset_index(drop=True)
    )
    top["rank"] = top.index + 1

    # Attach rank-1 peak window (left join in case a zone has no peaks entry)
    rank1 = zone_peaks[zone_peaks["rank"] == 1][["h3_index", "dow", "hour"]].rename(
        columns={"dow": "peak_dow", "hour": "peak_hour"}
    )
    top = top.merge(rank1, on="h3_index", how="left")

    top["recommended_action"] = top["persistence_label"].map({
        "chronic": "Permanent fix (signage/bollards) + patrol",
        "sporadic": "Patrol at peak window",
    })

    return top[[
        "rank", "h3_index", "centroid_lat", "centroid_lon",
        "zri", "violation_count", "persistence_label",
        "peak_dow", "peak_hour", "recommended_action",
    ]]


# ---------------------------------------------------------------------------
# Part 2 — simulate (importable by dashboard)
# ---------------------------------------------------------------------------
def simulate(
    num_officers: int,
    zones_per_officer: int = ZONES_PER_OFFICER,
) -> dict:
    """
    Allocate num_officers officers each covering zones_per_officer zones.

    Returns a dict with:
      officers, zones_covered, coverage_pct, zri_coverage_pct,
      violations_covered, h3_list
    """
    global _zones_scored
    if _zones_scored is None:
        _zones_scored = pd.read_parquet(IN_ZONES_SCORED)

    zs = _zones_scored.sort_values("zri", ascending=False).reset_index(drop=True)
    n_cover = num_officers * zones_per_officer
    covered = zs.head(n_cover)

    total_violations = int(zs["violation_count"].sum())
    violations_covered = int(covered["violation_count"].sum())
    coverage_pct = (violations_covered / total_violations * 100) if total_violations > 0 else 0.0

    total_zri = zs["zri"].sum()
    covered_zri = covered["zri"].sum()
    zri_coverage_pct = (covered_zri / total_zri * 100) if total_zri > 0 else 0.0

    return {
        "officers": num_officers,
        "zones_covered": len(covered),
        "coverage_pct": round(coverage_pct, 2),
        "zri_coverage_pct": round(zri_coverage_pct, 2),
        "violations_covered": violations_covered,
        "h3_list": covered["h3_index"].tolist(),
    }


# ---------------------------------------------------------------------------
# Part 3 — challans
# ---------------------------------------------------------------------------
def build_challans(repeat_vehicles: pd.DataFrame) -> pd.DataFrame:
    ch = repeat_vehicles.copy()
    ch["fine_per_violation"] = ch["fine_tier"].map(FINE_BY_TIER)
    ch["total_due"] = ch["total_count"] * ch["fine_per_violation"]

    return (
        ch[[
            "vehicle_number", "total_count", "zones_visited",
            "first_seen", "last_seen",
            "fine_tier", "fine_per_violation", "total_due",
        ]]
        .sort_values("total_due", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Part 4 — generate_challan (importable by dashboard)
# ---------------------------------------------------------------------------
def generate_challan(vehicle_number: str) -> dict:
    """
    Return a formatted notice dict for one vehicle_number.
    Reads challans.parquet if the module-level cache is empty.
    """
    global _challans
    if _challans is None:
        _challans = pd.read_parquet(OUT_CHALLANS)

    row = _challans[_challans["vehicle_number"] == vehicle_number]
    if row.empty:
        return {"error": f"Vehicle {vehicle_number!r} not found in repeat-offender list."}

    r = row.iloc[0]
    return {
        "plate": r["vehicle_number"],
        "total_violations": int(r["total_count"]),
        "zones_visited": int(r["zones_visited"]),
        "first_seen": str(r["first_seen"]),
        "last_seen": str(r["last_seen"]),
        "tier": int(r["fine_tier"]),
        "fine_per_violation_inr": int(r["fine_per_violation"]),
        "total_due_inr": int(r["total_due"]),
        "note": (
            "Fine amounts are illustrative only and do not constitute "
            "legal or official penalty slabs."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading inputs ...")
    zones_scored, zone_peaks, repeat_vehicles = _load()

    # Part 1
    print("Building patrol plan ...")
    patrol_plan = build_patrol_plan(zones_scored, zone_peaks)
    patrol_plan.to_parquet(OUT_PATROL, index=False)
    print(f"  Written -> {OUT_PATROL}")

    # Part 3
    print("Building challans ...")
    challans = build_challans(repeat_vehicles)
    _challans = challans  # populate cache
    challans.to_parquet(OUT_CHALLANS, index=False)
    print(f"  Written -> {OUT_CHALLANS}")

    # --- print: top 10 patrol plan -----------------------------------------
    print("\n" + "=" * 80)
    print("TOP 10 PATROL PLAN")
    print("=" * 80)
    fmt = "{:>4}  {:<22} {:>6}  {:>10}  {:<10}  {:>4}  {:>4}  {}"
    print(fmt.format(
        "rank", "h3_index", "zri", "violations", "label", "dow", "hr", "action"
    ))
    print("-" * 80)
    for _, row in patrol_plan.head(10).iterrows():
        print(fmt.format(
            int(row["rank"]),
            row["h3_index"],
            f"{row['zri']:.2f}",
            f"{row['violation_count']:,}",
            row["persistence_label"],
            str(row["peak_dow"]) if pd.notna(row["peak_dow"]) else "-",
            str(int(row["peak_hour"])) if pd.notna(row["peak_hour"]) else "-",
            row["recommended_action"],
        ))

    # --- print: simulate(8) ------------------------------------------------
    print("\n" + "=" * 80)
    print("OFFICER SIMULATION — 8 officers")
    print("=" * 80)
    sim = simulate(8)
    for k, v in sim.items():
        if k != "h3_list":
            print(f"  {k:<24}: {v}")
    print(f"  {'h3_list (first 5)':<22}: {sim['h3_list'][:5]} ...")

    # --- print: top 10 challans --------------------------------------------
    print("\n" + "=" * 80)
    print("TOP 10 CHALLANS BY TOTAL DUE")
    print("=" * 80)
    cfmt = "{:<20}  {:>6}  {:>6}  {:>5}  {:>12}"
    print(cfmt.format("vehicle_number", "count", "zones", "tier", "total_due(INR)"))
    print("-" * 57)
    for _, row in challans.head(10).iterrows():
        print(cfmt.format(
            row["vehicle_number"],
            int(row["total_count"]),
            int(row["zones_visited"]),
            int(row["fine_tier"]),
            f"{int(row['total_due']):,}",
        ))

    # --- print: example challan notice -------------------------------------
    top_vehicle = challans.iloc[0]["vehicle_number"]
    print("\n" + "=" * 80)
    print(f"EXAMPLE CHALLAN NOTICE — {top_vehicle}")
    print("=" * 80)
    notice = generate_challan(top_vehicle)
    for k, v in notice.items():
        print(f"  {k:<28}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
