import json
import ast
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px

from temp_3 import build_block_profile_with_charging 

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
OUT_ROOT = Path("../data/processed")
BLOCK_SUMMARY_PATH = OUT_ROOT / "block_success_summary_depot_only.parquet"
CANDIDATE_STOP_MAP_PATH = OUT_ROOT / "candidate_stop_map.parquet"
SOC_THRESHOLD_PERCENT = 20.0


# --------------------------------------------------------------------
# Helpers: parsing & data loading
# --------------------------------------------------------------------
def parse_combined_sequence_json(raw: Any) -> List[Dict[str, Any]]:
    """Parse combined_sequence_json safely into list[dict]."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, list):
        return raw

    s = str(raw).strip()
    # Try JSON first
    try:
        return json.loads(s)
    except Exception:
        # Fallback to Python literal
        return ast.literal_eval(s)


@st.cache_data(show_spinner=False)
def load_block_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Restrict to 40-ft if desired
    df = df[df["asset_class"] == "40-ft"].copy()
    # Parse trip sequences once
    df["block_trips"] = df["combined_sequence_json"].apply(parse_combined_sequence_json)
    return df


@st.cache_data(show_spinner=False)
def load_candidate_stop_map(path: str | Path) -> pd.DataFrame:
    if not Path(path).exists():
        st.error(f"Candidate stop map not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def build_candidate_dict(candidate_df: pd.DataFrame) -> Dict[str, Set[int]]:
    """
    Build mapping: candidate_name -> set(stop_codes)
    from candidate_stop_map.
    """
    candidate_dict: Dict[str, Set[int]] = {}
    if candidate_df.empty:
        return candidate_dict

    for name, group in candidate_df.groupby("candidate_name"):
        codes: Set[int] = set()
        for v in group["stop_code"]:
            if pd.isna(v):
                continue
            try:
                codes.add(int(str(v).strip()))
            except ValueError:
                continue
        candidate_dict[name] = codes
    return candidate_dict


# --------------------------------------------------------------------
# On-route simulation helpers (block-level)
# --------------------------------------------------------------------
def simulate_block_for_mode(
    block_trips: List[Dict[str, Any]],
    matched_codes: Set[int],
    mode: str,
    soc_threshold: float = SOC_THRESHOLD_PERCENT,
) -> Tuple[float, float, float, str]:
    """
    Simulate a single block in given mode ('medium' or 'heavy') using
    your build_block_profile_with_charging() function.

    Returns:
        soc_min, soc_end, total_charged_kwh, success_flag
        - soc_min: minimum SOC (%) across the whole block
        - soc_end: final SOC (%) at the end of block
        - total_charged_kwh: total energy received from on-route chargers (kWh)
        - success_flag: "SUCCESS" or "FAILURE"
    """
    if not block_trips:
        return math.nan, math.nan, math.nan, "FAILURE"

    profile = build_block_profile_with_charging(
        block_trips,
        matched_codes=matched_codes,
        mode=mode,
    )

    if profile is None or len(profile) == 0:
        return math.nan, math.nan, math.nan, "FAILURE"

    soc_series = profile["soc_pct"]
    soc_min = float(soc_series.min())
    soc_end = float(soc_series.iloc[-1])

    # Total energy received from chargers = final cumulative charged kWh
    total_charged_kwh = float(profile["cum_charged_kwh"].iloc[-1])

    success_flag = "SUCCESS" if soc_min >= soc_threshold else "FAILURE"
    return soc_min, soc_end, total_charged_kwh, success_flag


def simulate_all_blocks_for_scenario(
    blocks_df: pd.DataFrame,
    matched_codes: Set[int],
) -> pd.DataFrame:
    """
    Run on-route simulation for ALL blocks (medium + heavy) under a given set of
    charger stop codes (matched_codes).

    Returns a DataFrame that is basically the original blocks_df plus extra
    on-route columns.
    """
    results = []
    for _, row in blocks_df.iterrows():
        block_id = row["block_id"]
        trips = row["block_trips"]

        # Medium
        soc_min_med, soc_end_med, total_recv_med, succ_med = simulate_block_for_mode(
            trips, matched_codes, mode="medium"
        )

        # Heavy
        soc_min_hev, soc_end_hev, total_recv_hev, succ_hev = simulate_block_for_mode(
            trips, matched_codes, mode="heavy"
        )

        results.append(
            {
                "block_id": block_id,
                "soc_min_medium_on_route": soc_min_med,
                "soc_left_medium_percent_on_route_charge": soc_end_med,
                "medium_success_on_route_charge": succ_med,
                "total_energy_received_medium": total_recv_med,
                "soc_min_heavy_on_route": soc_min_hev,
                "soc_left_heavy_percent_on_route_charge": soc_end_hev,
                "heavy_success_on_route_charge": succ_hev,
                "total_energy_received_heavy": total_recv_hev,
            }
        )

    results_df = pd.DataFrame(results)
    # Merge back onto original block summary; keep all original columns
    df_report = blocks_df.merge(results_df, on="block_id", how="left")
    return df_report


# --------------------------------------------------------------------
# Plot helper: SOC vs Distance (Depot-only vs On-route)
# --------------------------------------------------------------------
def make_block_soc_plot(
    block_trips: List[Dict[str, Any]],
    matched_codes_onroute: Set[int],
    mode: str = "heavy",
) -> px.line:
    """
    Build a SOC vs distance plot for a single block, comparing:
    - depot-only (no chargers along route)
    - on-route charging scenario
    """
    # Depot-only approximation: no on-route chargers
    profile_depot = build_block_profile_with_charging(
        block_trips,
        matched_codes=set(),
        mode=mode,
    )
    profile_depot = profile_depot.copy()
    profile_depot["scenario"] = "Depot-only"

    # On-route scenario
    profile_on = build_block_profile_with_charging(
        block_trips,
        matched_codes=matched_codes_onroute,
        mode=mode,
    )
    profile_on = profile_on.copy()
    profile_on["scenario"] = "On-route charging"

    combo = pd.concat([profile_depot, profile_on], ignore_index=True)

    fig = px.line(
        combo,
        x="dist_km",
        y="soc_pct",
        color="scenario",
        title=f"SOC vs Distance ({mode.capitalize()} duty)",
        labels={"dist_km": "Distance (km)", "soc_pct": "SOC (%)"},
    )
    fig.update_layout(legend_title_text="Scenario")
    return fig


# --------------------------------------------------------------------
# Main UI renderer (for multi-panel app)
# --------------------------------------------------------------------
def render_onroute_panel():
    st.markdown("## Energy Model--On-Route Charge Summary Panel")

    # ---- Load base block summary & candidate stops ----
    blocks_base = load_block_summary(BLOCK_SUMMARY_PATH)
    blocks_base = blocks_base.rename(
        columns={
            "medium_success": "medium_success_depot_only",
            "heavy_success": "heavy_success_depot_only",
        }
    )
    if blocks_base.empty:
        st.stop()

    candidate_df = load_candidate_stop_map(CANDIDATE_STOP_MAP_PATH)
    candidate_dict = build_candidate_dict(candidate_df)
    candidate_names = sorted(candidate_dict.keys())

    # ---- Scenario cache init ----
    if "onroute_scenario_cache" not in st.session_state:
        st.session_state["onroute_scenario_cache"] = {}

    # -------------------------------
    # 1. Sidebar: On-route charger locations (no Run button)
    # -------------------------------
    with st.sidebar:
        st.header("On-route charger locations")

        with st.expander("Show all charger locations", expanded=False):
            st.markdown(
                "All locations are selected by default. "
                "Uncheck the ones you want to remove."
            )

            selected_candidates: List[str] = []
            for name in candidate_names:
                checked = st.checkbox(
                    label=name,
                    value=True,
                    key=f"cand_{name}",
                )
                if checked:
                    selected_candidates.append(name)

        # Determine matched_codes for current scenario
        if not selected_candidates:
            matched_codes: Set[int] = set()
            scenario_name = "NONE"
        else:
            matched_codes = set()
            for name in selected_candidates:
                matched_codes |= candidate_dict.get(name, set())
            scenario_name = (
                "ALL"
                if len(selected_candidates) == len(candidate_names)
                else "|".join(sorted(selected_candidates))
            )

        st.markdown(
            f"**Active scenario:** `{scenario_name}` "
            f"({len(matched_codes)} charger stop codes)"
        )

    # -------------------------------
    # 2. Run or reuse scenario simulation (auto on location change)
    # -------------------------------
    cache = st.session_state["onroute_scenario_cache"]
    if scenario_name not in cache:
        with st.spinner("Running on-route simulation for all blocks..."):
            report_df = simulate_all_blocks_for_scenario(blocks_base, matched_codes)
        cache[scenario_name] = report_df
    else:
        report_df = cache[scenario_name]

    block_inv = report_df  # alias to mirror beb_dashboard pattern

    # -------------------------------
    # 3. Sidebar: Filters (ON-ROUTE success)
    # -------------------------------
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Filters")

        # 3.1 Duty / energy mode
        energy_mode = st.radio(
            "Duty / Energy mode",
            ["Heavy-duty", "Medium-duty"],
            index=0,
        )
        energy_mode_key = "heavy" if energy_mode.startswith("Heavy") else "medium"

        # IMPORTANT: success now uses ON-ROUTE success columns
        success_col = (
            "heavy_success_on_route_charge"
            if energy_mode_key == "heavy"
            else "medium_success_on_route_charge"
        )

        # Guard: empty inventory
        if block_inv.empty:
            st.warning("No blocks available in this scenario.")
            st.stop()

        # 3.2 Depot selection
        depot_options = sorted(block_inv["depot_code"].dropna().unique())
        depot = st.selectbox("Depot", depot_options)

        df_depot = block_inv[block_inv["depot_code"] == depot].copy()

        # 3.3 Success / failure filter for the chosen mode (ON-ROUTE)
        status_label = st.radio(
            "On-route simulation result",
            ["Success", "Failure"],
            index=0,
            horizontal=True,
        )
        status_value = "SUCCESS" if status_label == "Success" else "FAILURE"

        df_mode = df_depot[df_depot[success_col] == status_value].copy()

        if df_mode.empty:
            st.warning("No blocks match this depot + on-route result combination.")
            st.stop()

        # 3.4 Select service_day FIRST
        service_day_options = sorted(df_mode["service_day"].dropna().unique())
        service_day = st.selectbox("Service day", service_day_options)

        df_day = df_mode[df_mode["service_day"] == service_day].copy()

        if df_day.empty:
            st.warning("No records for this service_day after filters.")
            st.stop()

        # 3.5 Select line_group within that service_day
        line_group_options = sorted(df_day["line_group"].dropna().unique())
        line_group = st.selectbox("Line group", line_group_options)

        df_lg = df_day[df_day["line_group"] == line_group].copy()

        if df_lg.empty:
            st.warning("No records for this service_day + line_group after filters.")
            st.stop()

        # 3.6 Select block_number (within service_day & line_group)
        block_number_options = sorted(df_lg["block_number"].dropna().unique())
        block_number = st.selectbox("Block number", block_number_options)

        df_final = df_lg[df_lg["block_number"] == block_number].copy()

        if df_final.empty:
            st.warning(
                "No records for this combination of depot + result + "
                "service_day + line_group + block_number."
            )
            st.stop()

        # Expect exactly one row per (service_day, line_group, block_number)
        row_block = df_final.iloc[0]

        # Parse combined_sequence_json for this block (for SOC profile)
        try:
            sequence = parse_combined_sequence_json(row_block["combined_sequence_json"])
            if not isinstance(sequence, list):
                raise ValueError("combined_sequence_json is not a list")
        except Exception as e:
            st.error(
                f"Unable to parse combined_sequence_json for "
                f"DAY:{service_day}  LG:{line_group}  BK:{block_number} — {e}"
            )
            st.stop()

        # Show original JSON, nicely formatted, without modifying structure
        with st.expander("Show original combined_sequence_json", expanded=False):
            raw_json_str = row_block["combined_sequence_json"]
            try:
                parsed = parse_combined_sequence_json(raw_json_str)
                pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                st.code(pretty, language="json")
            except Exception:
                st.code(str(raw_json_str), language="json")

    # -------------------------------
    # 4. Main area: KPIs + SOC plot + block-level results
    # -------------------------------

    st.subheader("KPI: depot-only vs on-route (all blocks)")

    # Masks
    med_depot_mask = block_inv["medium_success_depot_only"] == "SUCCESS"
    med_on_mask    = block_inv["medium_success_on_route_charge"] == "SUCCESS"
    hev_depot_mask = block_inv["heavy_success_depot_only"] == "SUCCESS"
    hev_on_mask    = block_inv["heavy_success_on_route_charge"] == "SUCCESS"

    # Rates (%)
    med_depot_rate = med_depot_mask.mean() * 100.0
    med_on_rate    = med_on_mask.mean() * 100.0
    hev_depot_rate = hev_depot_mask.mean() * 100.0
    hev_on_rate    = hev_on_mask.mean() * 100.0

    # Counts (# of successful blocks)
    med_depot_n = int(med_depot_mask.sum())
    med_on_n    = int(med_on_mask.sum())
    hev_depot_n = int(hev_depot_mask.sum())
    hev_on_n    = int(hev_on_mask.sum())

    # --- Row 1: success RATES ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Medium depot-only success (%)", f"{med_depot_rate:.2f}")
    c2.metric(
        "Medium on-route success (%)",
        f"{med_on_rate:.2f}",
        delta=f"{(med_on_rate - med_depot_rate):+.2f} pts",
    )
    c3.metric("Heavy depot-only success (%)", f"{hev_depot_rate:.2f}")
    c4.metric(
        "Heavy on-route success (%)",
        f"{hev_on_rate:.2f}",
        delta=f"{(hev_on_rate - hev_depot_rate):+.2f} pts",
    )

    # --- Row 2: success COUNTS ---
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Medium successful blocks (depot-only)",
        f"{med_depot_n}",
    )
    c6.metric(
        "Medium successful blocks (on-route)",
        f"{med_on_n}",
        delta=f"{med_on_n - med_depot_n:+d}",
    )
    c7.metric(
        "Heavy successful blocks (depot-only)",
        f"{hev_depot_n}",
    )
    c8.metric(
        "Heavy successful blocks (on-route)",
        f"{hev_on_n}",
        delta=f"{hev_on_n - hev_depot_n:+d}",
    )

    # -------------------------------
    # Depot-level summary (all blocks in this scenario, by depot)
    # -------------------------------
    st.markdown("### Depot-level summary (depot-only vs on-route)")

    depot_rows = []
    for depot_code, g in block_inv.groupby("depot_code"):
        # Medium masks
        med_dep_mask = g["medium_success_depot_only"] == "SUCCESS"
        med_on_mask  = g["medium_success_on_route_charge"] == "SUCCESS"

        # Heavy masks
        hev_dep_mask = g["heavy_success_depot_only"] == "SUCCESS"
        hev_on_mask  = g["heavy_success_on_route_charge"] == "SUCCESS"

        total_blocks = len(g)

        med_dep_n = int(med_dep_mask.sum())
        med_on_n  = int(med_on_mask.sum())
        hev_dep_n = int(hev_dep_mask.sum())
        hev_on_n  = int(hev_on_mask.sum())

        med_dep_rate = med_dep_n / total_blocks * 100.0 if total_blocks > 0 else float("nan")
        med_on_rate  = med_on_n  / total_blocks * 100.0 if total_blocks > 0 else float("nan")
        hev_dep_rate = hev_dep_n / total_blocks * 100.0 if total_blocks > 0 else float("nan")
        hev_on_rate  = hev_on_n  / total_blocks * 100.0 if total_blocks > 0 else float("nan")

        depot_rows.append(
            {
                "depot_code": depot_code,
                "total_blocks": total_blocks,

                # Medium
                "med_success_blocks_depot_only": med_dep_n,
                "med_success_rate_depot_only_%": med_dep_rate,
                "med_success_blocks_on_route": med_on_n,
                "med_success_rate_on_route_%": med_on_rate,

                # Heavy
                "heavy_success_blocks_depot_only": hev_dep_n,
                "heavy_success_rate_depot_only_%": hev_dep_rate,
                "heavy_success_blocks_on_route": hev_on_n,
                "heavy_success_rate_on_route_%": hev_on_rate,
            }
        )

    depot_summary_df = pd.DataFrame(depot_rows).sort_values("depot_code")

    st.dataframe(depot_summary_df, width="stretch")

    st.markdown("---")

    # 4.1 SOC vs distance plot for selected block
    st.subheader(
        f"SOC profile for Depot-only vs On-route charging "
        f"(Depot {row_block['depot_code']}, Day {service_day}, LG {line_group}, Block {block_number})"
    )

    block_trips = row_block["block_trips"]
    plot_mode = energy_mode_key 

    fig = make_block_soc_plot(
        block_trips=block_trips,
        matched_codes_onroute=matched_codes,
        mode=plot_mode,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4.2 Block-level results table
    st.subheader("Block-level results (depot-only vs on-route, all blocks in this scenario)")

    df_table = block_inv.copy()

    cols_for_block = [
        "block_id",
        "depot_code",
        "service_day",
        "line_group",
        "block_number",
        # depot-only baseline
        "total_energy_medium_kwh",
        "total_energy_heavy_kwh",
        "soc_left_medium_percent",
        "soc_left_heavy_percent",
        "medium_success_depot_only",
        "heavy_success_depot_only",
        # on-route metrics
        "soc_left_medium_percent_on_route_charge",
        "soc_left_heavy_percent_on_route_charge",
        "total_energy_received_medium",
        "total_energy_received_heavy",
        "medium_success_on_route_charge",
        "heavy_success_on_route_charge",
    ]
    cols_for_block = [c for c in cols_for_block if c in df_table.columns]

    df_table = df_table[cols_for_block].sort_values(
        ["service_day", "line_group", "block_number"]
    )

    st.dataframe(df_table, width="stretch")


# --------------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Energy Model--On-Route Charge Summary Panel",
        layout="wide",
    )
    render_onroute_panel()


if __name__ == "__main__":
    main()
