# Parking Intelligence — Bengaluru

AI-driven parking intelligence that detects illegal-parking hotspots, scores their
impact on traffic, and turns a raw log of ~298,000 parking tickets into a prioritized,
predictive enforcement plan — built entirely from data the city already collects, with
no new sensors or cameras.

> **The headline:** just **24 zones — under 1% of the city's 2,534 zones — contain ~17% of all recorded violations.** Targeted enforcement beats random patrolling.

---

## The problem

On-street illegal and spillover parking near markets, metro stations, and commercial areas
chokes carriageways and intersections. Today enforcement is reactive and patrol-based: officers
go where they happen to be, with no map of where parking does the most damage and no way to
prioritize. This project answers: **how can we detect illegal-parking hotspots, quantify their
traffic impact, and tell officers exactly where and when to enforce?**

## What it does

A single dashboard with five layers:

1. **Detect** — clusters 298k violations into ~170 m H3 hex zones and maps the hotspots.
2. **Quantify** — scores each zone with an explainable **Zone Risk Index (ZRI)** blending
   violation density, heavy-vehicle mix, repeat-offender rate, spillover, and junction proximity.
3. **Characterise** — labels zones **chronic** (need a permanent fix: signage/bollards) vs
   **sporadic** (need patrols), and flags violation **bursts**.
4. **Predict / Live** — builds a 24×7 time grid per zone to find peak windows, and replays the
   data as a simulated live feed with burst alerts.
5. **Prescribe** — a ranked patrol plan (where + when + action), a deployment **simulator**
   ("N officers → % of violations covered"), and an escalating-fine **challan engine** for
   habitual offenders.

## Key insights surfaced

- **Volume ≠ impact:** the single biggest zone by raw count isn't even in the ZRI top 10, while
  some low-volume zones rank high because of heavy vehicles, repeat offenders, and junction risk.
- **Evening enforcement blind spot:** violations are logged overwhelmingly pre-dawn to midday
  (peak 10–11 AM) and are near-zero in the evening (6–9 PM ≈ 367 tickets total).
- **Habitual offenders are real:** 11,852 vehicles caught 3+ times; the worst was ticketed
  55 times across just 3 zones.
- **Note:** this is *enforcement* data (where officers ticket), not ground-truth occurrence —
  the blind spots reflect patrol behaviour as much as parking behaviour.

## Tech stack

Python · pandas · H3 (hex grid) · scikit-learn · pyarrow (Parquet) · pydeck · Streamlit

## Project structure

```
parking-intel/
├── src/
│   ├── a_foundation.py   # load + clean + H3 zoning  -> violations_clean, zones_base
│   ├── b_scoring.py      # ZRI + persistence + repeat offenders -> zones_scored, repeat_vehicles
│   ├── c_time.py         # 24x7 grid + peak windows + bursts -> zone_time, zone_peaks, bursts
│   ├── d_prescribe.py    # patrol plan + simulator + challan engine -> patrol_plan, challans
│   └── e_dashboard.py    # Streamlit app tying it all together
├── data/
│   ├── raw/              # source CSVs (NOT in repo — too large; see below)
│   └── built/            # generated Parquet artifacts (in repo)
├── notebooks/            # data exploration
├── requirements.txt
└── README.md
```

## Setup & run (local)

```bash
git clone <your-repo-url>
cd parking-intel
pip install -r requirements.txt
streamlit run src/e_dashboard.py
```

The dashboard reads the prebuilt Parquet files in `data/built/`, so this is all you need to run it.

## Rebuilding from raw data (optional)

Only needed if you want to regenerate the artifacts from scratch:

1. Get the source CSVs (`violations.csv`, `events.csv`) from the shared Google Drive folder
   and place them in `data/raw/`. *(They are not committed — the violations file exceeds
   GitHub's 100 MB limit.)*
2. Run the pipeline in order:
   ```bash
   python src/a_foundation.py
   python src/b_scoring.py
   python src/c_time.py
   python src/d_prescribe.py
   ```
3. Then launch the dashboard.

## The pipeline (what each module produces)

| Module | Reads | Writes |
|--------|-------|--------|
| `a_foundation.py` | `data/raw/violations.csv` | `violations_clean.parquet`, `zones_base.parquet` |
| `b_scoring.py` | clean + base | `zones_scored.parquet`, `repeat_vehicles.parquet` |
| `c_time.py` | clean + scored | `zone_time.parquet`, `zone_peaks.parquet`, `bursts.parquet` |
| `d_prescribe.py` | scored + peaks + repeats | `patrol_plan.parquet`, `challans.parquet` + `simulate()` / `generate_challan()` |
| `e_dashboard.py` | all built artifacts | the Streamlit app |

## Live demo

🔗 https://parking-intel-wbrpxjsna4ur5wk5xumggx.streamlit.app/

## Data note

The dataset is anonymized Bengaluru traffic-police parking violations (Nov 2023 – Apr 2024).
It contains no traffic-speed measurement, so "traffic impact" is an explainable composite
estimate (ZRI), not a direct measurement. `closed_datetime` is empty in the source, so no
resolution-time features are used. Challan fine amounts are illustrative only and are not
official penalty slabs.

## Limitations & future work

- Data reflects enforcement activity, not every real violation.
- Impact is estimated, not measured (no flow/speed data).
- Five months of history → forecasts capture broad patterns, not rare one-offs.
- **Future:** live officer-app feed, CCTV/ANPR detection, real traffic-flow fusion, and
  integration with the official e-challan system.


