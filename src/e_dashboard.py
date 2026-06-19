"""
e_dashboard.py — Streamlit dashboard for the parking-intel project.

Run with:  streamlit run src/e_dashboard.py
"""

import datetime
import pathlib
import sys

import pandas as pd
import pydeck as pdk
import streamlit as st

# Add src/ to path so sibling imports resolve correctly when launched with
# `streamlit run src/e_dashboard.py` from the project root.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from d_prescribe import generate_challan, simulate

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILT_DIR = ROOT / "data" / "built"

# Free tile style that needs no Mapbox API key
_DARK_TILES = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Parking Intelligence — Bengaluru",
    layout="wide",
    page_icon=":vertical_traffic_light:",
)
st.title("Parking Intelligence — Bengaluru")

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_zones_scored() -> pd.DataFrame:
    return pd.read_parquet(BUILT_DIR / "zones_scored.parquet")


@st.cache_data
def load_patrol_plan() -> pd.DataFrame:
    return pd.read_parquet(BUILT_DIR / "patrol_plan.parquet")


@st.cache_data
def load_bursts() -> pd.DataFrame:
    return pd.read_parquet(BUILT_DIR / "bursts.parquet")


@st.cache_data
def load_challans() -> pd.DataFrame:
    return pd.read_parquet(BUILT_DIR / "challans.parquet")


@st.cache_data
def load_violations_slim() -> pd.DataFrame:
    """Load only the columns needed for Tab 4 to keep memory low."""
    return pd.read_parquet(
        BUILT_DIR / "violations_clean.parquet",
        columns=["id", "lat", "lon", "ts_ist", "h3_index", "junction_name"],
    )


def _try(loader):
    try:
        return loader(), None
    except Exception as exc:
        return None, str(exc)


# Load everything upfront so errors surface before any tab renders
zones_scored, _e = _try(load_zones_scored)
if _e:
    st.error(f"Cannot load zones_scored.parquet — run a_foundation.py and b_scoring.py first.\n{_e}")
    st.stop()

patrol_plan, _pp_err = _try(load_patrol_plan)
bursts, _b_err = _try(load_bursts)
challans, _ch_err = _try(load_challans)
violations, _v_err = _try(load_violations_slim)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Hotspot Map",
    "Priority List",
    "Patrol Simulator",
    "Live Ops (replay)",
    "Repeat Offenders & Challans",
])

# ============================================================
# TAB 1 — Hotspot Map
# ============================================================
with tab1:
    total_viol = int(zones_scored["violation_count"].sum())
    n_zones = len(zones_scored)
    n_chronic = int((zones_scored["persistence_label"] == "chronic").sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Violations", f"{total_viol:,}")
    m2.metric("H3 Zones", f"{n_zones:,}")
    m3.metric("Chronic Zones", f"{n_chronic:,}")

    # Filter to zones with meaningful activity; sparse zones clutter the view
    df_map = zones_scored[
        (zones_scored["violation_count"] >= 15)
    ][["h3_index", "zri", "violation_count", "persistence_label"]].copy()
    df_map["zri_label"] = df_map["zri"].round(1)

    # Colour ramp on a fixed 0-100 ZRI scale: green -> yellow -> orange -> red
    # Breakpoints: 0=green, 33=yellow, 66=orange, 100=red
    def _zri_color(zri: float) -> list[int]:
        t = max(0.0, min(1.0, zri / 100.0))
        if t < 0.33:
            s = t / 0.33
            r, g, b = int(0 + 255 * s), 200, 0          # green -> yellow
        elif t < 0.66:
            s = (t - 0.33) / 0.33
            r, g, b = 255, int(200 - 140 * s), 0        # yellow -> orange
        else:
            s = (t - 0.66) / 0.34
            r, g, b = 255, int(60 - 60 * s), 0          # orange -> red
        return [r, g, b, int(0.55 * 255)]

    df_map["fill_color"] = df_map["zri"].apply(_zri_color)

    hex_layer = pdk.Layer(
        "H3HexagonLayer",
        data=df_map,
        get_hexagon="h3_index",
        get_fill_color="fill_color",
        extruded=False,
        filled=True,
        pickable=True,
        auto_highlight=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[hex_layer],
            initial_view_state=pdk.ViewState(
                latitude=12.97, longitude=77.59, zoom=12, pitch=0, bearing=0
            ),
            tooltip={
                "html": (
                    "<b>Zone:</b> {h3_index}<br>"
                    "<b>ZRI:</b> {zri_label}<br>"
                    "<b>Violations:</b> {violation_count}<br>"
                    "<b>Type:</b> {persistence_label}"
                ),
                "style": {
                    "backgroundColor": "#1a1a2e",
                    "color": "white",
                    "fontSize": "13px",
                    "padding": "6px",
                },
            },
            map_style=_DARK_TILES,
        )
    )

# ============================================================
# TAB 2 — Priority List
# ============================================================
with tab2:
    if patrol_plan is None:
        st.error(f"patrol_plan.parquet missing — run d_prescribe.py first.\n{_pp_err}")
    else:
        st.markdown(
            "**Chronic** zones were active on 30+ distinct calendar days and "
            "need permanent infrastructure fixes (signage, bollards). "
            "**Sporadic** zones flare at predictable windows and respond well "
            "to targeted patrol deployment."
        )

        display_pp = patrol_plan[[
            "rank", "h3_index", "zri", "violation_count",
            "persistence_label", "peak_dow", "peak_hour", "recommended_action",
        ]].rename(columns={
            "rank": "Rank",
            "h3_index": "Zone",
            "zri": "ZRI",
            "violation_count": "Violations",
            "persistence_label": "Label",
            "peak_dow": "Peak Day",
            "peak_hour": "Peak Hour",
            "recommended_action": "Recommended Action",
        })

        st.dataframe(
            display_pp,
            width="stretch",
            hide_index=True,
            column_config={
                "ZRI": st.column_config.NumberColumn(format="%.2f"),
                "Violations": st.column_config.NumberColumn(format="%d"),
                "Peak Hour": st.column_config.NumberColumn(format="%d:00"),
            },
        )

# ============================================================
# TAB 3 — Patrol Simulator
# ============================================================
with tab3:
    num_officers = st.slider("Number of officers", min_value=1, max_value=30, value=8)
    sim = simulate(num_officers)

    total_zones = len(zones_scored)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Officers", sim["officers"])
    c2.metric("Zones Covered", sim["zones_covered"])
    c3.metric("City Violations Covered", f"{sim['coverage_pct']:.1f}%")
    c4.metric("Violations Covered", f"{sim['violations_covered']:,}")

    st.caption(
        f"{sim['zones_covered']} zones (out of {total_zones:,}) "
        f"cover {sim['coverage_pct']:.1f}% of all violations "
        f"({sim['zri_coverage_pct']:.1f}% of total ZRI weight)."
    )

    covered_df = (
        zones_scored[zones_scored["h3_index"].isin(sim["h3_list"])][[
            "h3_index", "zri", "violation_count", "persistence_label",
        ]]
        .sort_values("zri", ascending=False)
        .reset_index(drop=True)
    )
    covered_df.index += 1

    st.dataframe(
        covered_df.rename(columns={
            "h3_index": "Zone",
            "zri": "ZRI",
            "violation_count": "Violations",
            "persistence_label": "Label",
        }),
        width="stretch",
        column_config={
            "ZRI": st.column_config.NumberColumn(format="%.2f"),
        },
    )

# ============================================================
# TAB 4 — Live Ops (replay)
# ============================================================
with tab4:
    if violations is None or bursts is None:
        missing = []
        if violations is None:
            missing.append(f"violations_clean.parquet ({_v_err})")
        if bursts is None:
            missing.append(f"bursts.parquet ({_b_err})")
        st.error("Missing data files:\n" + "\n".join(missing))
    else:
        ts_min = violations["ts_ist"].min()
        ts_max = violations["ts_ist"].max()

        # Streamlit slider needs tz-naive datetimes; we strip and re-attach IST
        dt_min = ts_min.to_pydatetime().replace(tzinfo=None)
        dt_max = ts_max.to_pydatetime().replace(tzinfo=None)

        now_naive = st.slider(
            "Replay time (IST)",
            min_value=dt_min,
            max_value=dt_max,
            value=dt_max,
            step=datetime.timedelta(hours=6),
            format="YYYY-MM-DD HH:mm",
        )

        now_aware = pd.Timestamp(now_naive, tz="Asia/Kolkata")

        # Violations up to "now"
        vc_now = violations[violations["ts_ist"] <= now_aware]
        st.metric("Violations so far", f"{len(vc_now):,}")

        # Recent burst events
        st.subheader("Recent Burst Events")
        recent_bursts = (
            bursts[bursts["window_start"] <= now_aware]
            .sort_values("window_start", ascending=False)
            .head(20)
        )

        if recent_bursts.empty:
            st.info("No burst events recorded up to this point.")
        else:
            for _, row in recent_bursts.iterrows():
                jct_tag = "  **[NEAR JUNCTION]**" if row["near_junction"] else ""
                st.warning(
                    f"Zone **{row['h3_index']}** | "
                    f"{row['window_start'].strftime('%Y-%m-%d %H:%M')} IST | "
                    f"{row['count']} tickets in 30 min{jct_tag}"
                )

        # Scatter map: violations in last 2 hours before "now"
        two_hrs_ago = now_aware - pd.Timedelta(hours=2)
        recent_vc = vc_now[vc_now["ts_ist"] >= two_hrs_ago]

        st.subheader(f"Violations in last 2 hours — {len(recent_vc):,} tickets")
        if recent_vc.empty:
            st.info("No violations in the 2 hours before the selected time.")
        else:
            scatter = pdk.Layer(
                "ScatterplotLayer",
                data=recent_vc[["lat", "lon"]],
                get_position=["lon", "lat"],
                get_color=[255, 80, 0, 200],
                get_radius=80,
                pickable=False,
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=[scatter],
                    initial_view_state=pdk.ViewState(
                        latitude=12.97, longitude=77.59, zoom=11
                    ),
                    map_style=_DARK_TILES,
                )
            )

# ============================================================
# TAB 5 — Repeat Offenders & Challans
# ============================================================
with tab5:
    if challans is None:
        st.error(f"challans.parquet missing — run d_prescribe.py first.\n{_ch_err}")
    else:
        st.subheader("Top Offenders by Total Fine Due")

        display_ch = challans[[
            "vehicle_number", "total_count", "zones_visited",
            "fine_tier", "fine_per_violation", "total_due",
        ]].rename(columns={
            "vehicle_number": "Vehicle",
            "total_count": "Tickets",
            "zones_visited": "Zones",
            "fine_tier": "Tier",
            "fine_per_violation": "Fine / Violation (INR)",
            "total_due": "Total Due (INR)",
        })

        st.dataframe(
            display_ch,
            width="stretch",
            hide_index=True,
            column_config={
                "Total Due (INR)": st.column_config.NumberColumn(format="₹%d"),
                "Fine / Violation (INR)": st.column_config.NumberColumn(format="₹%d"),
            },
        )

        st.divider()
        st.subheader("Generate Challan Notice")

        vehicles = challans["vehicle_number"].tolist()
        selected = st.selectbox("Select vehicle number", vehicles)

        if selected:
            notice = generate_challan(selected)

            if "error" in notice:
                st.error(notice["error"])
            else:
                n1, n2, n3, n4 = st.columns(4)
                n1.metric("Total Violations", notice["total_violations"])
                n2.metric("Zones Visited", notice["zones_visited"])
                n3.metric("Fine Tier", f"Tier {notice['tier']}")
                n4.metric("Total Due", f"₹{notice['total_due_inr']:,}")

                st.info(
                    f"**Vehicle plate:** {notice['plate']}  \n"
                    f"**First seen:** {notice['first_seen']}  \n"
                    f"**Last seen:** {notice['last_seen']}  \n"
                    f"**Fine per violation (Tier {notice['tier']}):** "
                    f"₹{notice['fine_per_violation_inr']:,}"
                )

                st.caption(f"⚠️ {notice['note']}")
