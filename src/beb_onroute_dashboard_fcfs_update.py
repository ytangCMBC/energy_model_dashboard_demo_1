from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from onroute_fcfs_helpers_update import (
    build_block_profile_with_charging,
    build_block_profile_with_assigned_events,
    build_candidate_dict,
    build_events_for_blocks,
    compute_depot_summary,
    compute_service_day_summary,
    load_candidate_stop_map,
    parse_combined_sequence_json,
    allocate_sessions_fcfs,
    simulate_all_blocks_with_allocation,
    summarize_assignment,
    dispensers_needed_by_candidate,
    scenario_output_dir,
    persist_scenario_artifacts,
    prune_scenario_artifacts,
    time_to_sec,
    _infer_trip_distance_km,
    _normalize_stop_code,
    INTERLINE_TYPES,
    build_final_proposed_dispensers,
    simulate_all_blocks_with_allocation_by_service_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "processed"
GTFS_DIR = OUT_ROOT / "gtfs_bus_only"
BLOCK_SUMMARY_PATH = OUT_ROOT / "block_success_summary_depot_only.parquet"
CANDIDATE_STOP_MAP_PATH = OUT_ROOT / "candidate_stop_map.parquet"
FCFS_CACHE_DIR = OUT_ROOT / "onroute_fcfs_cache"
BUS_STOPS_EXPORT_PATH = OUT_ROOT / "bus_stops_df_export.xlsx"
RUNTIME_SCENARIO_COLLECTION = "onroute_fcfs_runtime_scenarios"
MAX_SESSION_SCENARIOS = 3
MAX_SAVED_SCENARIOS = 10

def inject_scrollable_dropdown_css():
    st.markdown(
        """
        <style>
        div[data-baseweb="popover"] {
            max-height: 52vh !important;
            overflow-y: auto !important;
        }
        div[data-baseweb="popover"] [role="listbox"],
        div[data-baseweb="popover"] [data-baseweb="menu"] {
            max-height: 48vh !important;
            overflow-y: auto !important;
            overscroll-behavior: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False, max_entries=1)
def load_block_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["asset_class_new"] == "40-ft"].copy()
    df["block_trips"] = df["combined_sequence_json"].apply(parse_combined_sequence_json)
    df = df.rename(columns={
        "medium_success": "medium_success_depot_only",
        "heavy_success": "heavy_success_depot_only",
    })
    return df

@st.cache_data(show_spinner=False, max_entries=1)
def load_trip_to_route_short_name(gtfs_dir: str | Path) -> Dict[str, str]:
    gtfs_dir = Path(gtfs_dir)
    trips = pd.read_csv(gtfs_dir / "trips.txt", dtype=str, low_memory=False)[["trip_id", "route_id"]].dropna()
    routes = pd.read_csv(gtfs_dir / "routes.txt", dtype=str, low_memory=False)[["route_id", "route_short_name"]].dropna()
    merged = trips.merge(routes, on="route_id", how="left")
    return dict(zip(merged["trip_id"], merged["route_short_name"]))

@st.cache_data(show_spinner=False, max_entries=1)
def load_bus_stops_export(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path).copy()

    keep_cols = [c for c in ["stop_code", "stop_name_simple"] if c in df.columns]
    df = df[keep_cols].copy()

    def _parse_stop_code_cell(val):
        if pd.isna(val):
            return []

        # already numeric
        if isinstance(val, (int, float)):
            try:
                return [int(float(val))]
            except Exception:
                return []

        s = str(val).strip()
        if not s:
            return []

        # handle strings like "[52183.0, 52230.0]" or "[60995.0]"
        if s.startswith("[") and s.endswith("]"):
            s_inner = s[1:-1].strip()
            if not s_inner:
                return []
            out = []
            for part in s_inner.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    out.append(int(float(part)))
                except Exception:
                    continue
            return out

        # handle plain scalar string
        try:
            return [int(float(s))]
        except Exception:
            return []

    df["stop_code_list"] = df["stop_code"].apply(_parse_stop_code_cell)
    df = df.explode("stop_code_list").copy()
    df = df.rename(columns={"stop_code_list": "stop_code_parsed"})

    df = df.dropna(subset=["stop_code_parsed"]).copy()
    df["stop_code"] = df["stop_code_parsed"].astype(int)

    if "stop_name_simple" in df.columns:
        df["stop_name_simple"] = df["stop_name_simple"].astype(str).str.strip()
    else:
        df["stop_name_simple"] = ""

    df = df[["stop_code", "stop_name_simple"]].drop_duplicates(subset=["stop_code"], keep="first").copy()
    return df

@st.cache_data(show_spinner=False, max_entries=1)
def load_gtfs_stops(gtfs_dir: str | Path) -> pd.DataFrame:
    gtfs_dir = Path(gtfs_dir)

    candidates = [
        gtfs_dir / "stops.txt",
        gtfs_dir / "stops.csv",
        gtfs_dir / "stops.parquet",
    ]

    stop_path = None
    for p in candidates:
        if p.exists():
            stop_path = p
            break

    if stop_path is None:
        return pd.DataFrame(columns=["stop_code", "stop_name"])

    if stop_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(stop_path).copy()
    else:
        df = pd.read_csv(stop_path).copy()

    keep_cols = [c for c in ["stop_code", "stop_name", "stop_desc"] if c in df.columns]
    df = df[keep_cols].copy()

    if "stop_code" not in df.columns:
        return pd.DataFrame(columns=["stop_code", "stop_name"])

    df["stop_code"] = pd.to_numeric(df["stop_code"], errors="coerce")
    df = df.dropna(subset=["stop_code"]).copy()
    df["stop_code"] = df["stop_code"].astype(int)

    if "stop_name" not in df.columns:
        df["stop_name"] = None

    if "stop_desc" in df.columns:
        df["stop_name"] = df["stop_name"].fillna(df["stop_desc"])

    df["stop_name"] = df["stop_name"].astype(str).str.strip()
    df = df[["stop_code", "stop_name"]].drop_duplicates(subset=["stop_code"], keep="first").copy()
    return df

@st.cache_data(show_spinner=False, max_entries=1)
def build_combined_stop_name_map(
    bus_stop_export_path: str | Path,
    gtfs_dir: str | Path,
) -> Dict[int, str]:
    gtfs_df = load_gtfs_stops(gtfs_dir).copy()
    gtfs_map = dict(zip(gtfs_df["stop_code"], gtfs_df["stop_name"]))
    return gtfs_map

# @st.cache_data(show_spinner=False)
# def build_combined_stop_name_map(
#     bus_stop_export_path: str | Path,
#     gtfs_dir: str | Path,
# ) -> Dict[int, str]:
#     excel_df = load_bus_stops_export(bus_stop_export_path).copy()
#     gtfs_df = load_gtfs_stops(gtfs_dir).copy()

#     excel_map = dict(zip(excel_df["stop_code"], excel_df["stop_name_simple"]))
#     gtfs_map = dict(zip(gtfs_df["stop_code"], gtfs_df["stop_name"]))

#     combined = dict(gtfs_map)
#     combined.update(excel_map)  # Excel wins if both exist
#     return combined



@st.cache_data(show_spinner=False, max_entries=4)
def prepare_base_events(
    block_summary_path: str | Path,
    candidate_stop_map_path: str | Path,
    gtfs_dir: str | Path,
    sim_mode: str,
    proposal_mode: str = "heavy",
    exclude_p1: bool = False,
):
    blocks = load_block_summary(block_summary_path)
    candidate_df = load_candidate_stop_map(candidate_stop_map_path)

    sim_blocks = _filter_p1_blocks(blocks, sim_mode) if exclude_p1 else blocks
    proposal_blocks = _filter_p1_blocks(blocks, proposal_mode) if exclude_p1 else blocks

    # 1) events used for current selected duty simulation
    events_df, _ = build_final_proposed_dispensers(
        blocks_df=sim_blocks,
        candidate_df=candidate_df,
        gtfs_dir=gtfs_dir,
        mode=sim_mode,
    )

    # 2) proposed dispensers always based on heavy duty
    _, disp_final = build_final_proposed_dispensers(
        blocks_df=proposal_blocks,
        candidate_df=candidate_df,
        gtfs_dir=gtfs_dir,
        mode=proposal_mode,
    )

    return blocks, candidate_df, events_df, disp_final


def _filter_p1_blocks(blocks: pd.DataFrame, duty_key: str) -> pd.DataFrame:
    success_col = f"{duty_key}_success_depot_only"
    if success_col not in blocks.columns:
        return blocks.copy()

    success = (
        blocks[success_col].eq(True)
        | blocks[success_col].astype(str).str.upper().eq("SUCCESS")
    )
    return blocks.loc[~success].copy()


def _scenario_key(installed_disp: Dict[str, int], duty_key: str, exclude_p1: bool) -> str:
    p1_scope = "exclude_p1" if exclude_p1 else "include_p1"
    text = duty_key + "|" + p1_scope + "|" + "|".join(f"{k}:{installed_disp[k]}" for k in sorted(installed_disp))
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def _disp_key(duty_key: str, p1_scope_key: str, candidate_name: str) -> str:
    return f"disp_{duty_key}_{p1_scope_key}_{candidate_name}"


def _set_all_dispensers(
    candidate_names: List[str],
    proposed_map: Dict[str, int],
    duty_key: str,
    p1_scope_key: str,
    value_mode: str,
) -> None:
    for name in candidate_names:
        key = _disp_key(duty_key, p1_scope_key, name)
        if value_mode == "clear":
            st.session_state[key] = 0
        elif value_mode == "reset":
            st.session_state[key] = int(proposed_map.get(name, 0))


def _set_one_dispenser(
    candidate_name: str,
    proposed_map: Dict[str, int],
    duty_key: str,
    p1_scope_key: str,
    value_mode: str,
) -> None:
    key = _disp_key(duty_key, p1_scope_key, candidate_name)
    if value_mode == "clear":
        st.session_state[key] = 0
    elif value_mode == "reset":
        st.session_state[key] = int(proposed_map.get(candidate_name, 0))


def _clear_scenario_memory() -> None:
    st.session_state["fcfs_scenario_cache"] = {}
    st.session_state["fcfs_scenario_order"] = []


def _touch_scenario_cache_key(scenario_key: str) -> None:
    order = st.session_state.setdefault("fcfs_scenario_order", [])
    if scenario_key in order:
        order.remove(scenario_key)
    order.append(scenario_key)


def _remember_scenario(scenario_key: str, data: dict) -> None:
    cache = st.session_state.setdefault("fcfs_scenario_cache", {})
    cache[scenario_key] = data
    _touch_scenario_cache_key(scenario_key)

    order = st.session_state.setdefault("fcfs_scenario_order", [])
    while len(order) > MAX_SESSION_SCENARIOS:
        old_key = order.pop(0)
        if old_key != scenario_key:
            cache.pop(old_key, None)


def _service_day_sort_key(service_day: str) -> tuple[int, str]:
    order = {"MF": 0, "WEEKDAY": 0, "SAT": 1, "SATURDAY": 1, "SUN": 2, "SUNDAY": 2}
    label = str(service_day).upper()
    return order.get(label, 99), str(service_day)


def make_depot_service_heatmaps(report_df: pd.DataFrame, duty: str):
    if report_df.empty:
        return go.Figure(), go.Figure(), go.Figure()

    success_depot_col = f"{duty}_success_depot_only"
    success_onroute_col = f"{duty}_success_on_route_charge"
    dist_col = "total_distance_km"

    df = report_df.copy()
    df["depot_code"] = df["depot_code"].astype(str)
    df["service_day"] = df["service_day"].astype(str)
    df[dist_col] = pd.to_numeric(df[dist_col], errors="coerce").fillna(0.0)

    grouped = (
        df.groupby(["depot_code", "service_day"], dropna=False)
        .agg(
            total_blocks=("block_id", "count"),
            success_blocks_depot=(success_depot_col, lambda s: int((s == "SUCCESS").sum())),
            success_blocks_onroute=(success_onroute_col, lambda s: int((s == "SUCCESS").sum())),
            beb_km_depot=(
                dist_col,
                lambda s: float(df.loc[s.index].loc[df.loc[s.index, success_depot_col] == "SUCCESS", dist_col].sum()),
            ),
            beb_km_onroute=(
                dist_col,
                lambda s: float(df.loc[s.index].loc[df.loc[s.index, success_onroute_col] == "SUCCESS", dist_col].sum()),
            ),
        )
        .reset_index()
    )
    grouped["success_rate_depot"] = np.where(
        grouped["total_blocks"] > 0,
        grouped["success_blocks_depot"] / grouped["total_blocks"] * 100.0,
        0.0,
    )
    grouped["success_rate_onroute"] = np.where(
        grouped["total_blocks"] > 0,
        grouped["success_blocks_onroute"] / grouped["total_blocks"] * 100.0,
        0.0,
    )
    grouped["unlocked_blocks"] = grouped["success_blocks_onroute"] - grouped["success_blocks_depot"]
    grouped["success_rate_delta"] = grouped["success_rate_onroute"] - grouped["success_rate_depot"]
    grouped["beb_km_delta"] = grouped["beb_km_onroute"] - grouped["beb_km_depot"]

    depots = sorted(grouped["depot_code"].dropna().unique().tolist())
    service_days = sorted(grouped["service_day"].dropna().unique().tolist(), key=_service_day_sort_key)

    metric_fields = [
        "total_blocks",
        "success_blocks_depot",
        "success_blocks_onroute",
        "unlocked_blocks",
        "success_rate_depot",
        "success_rate_onroute",
        "success_rate_delta",
        "beb_km_depot",
        "beb_km_onroute",
        "beb_km_delta",
    ]

    indexed = grouped.set_index(["depot_code", "service_day"])

    def matrix_for(field: str) -> list[list[float]]:
        rows = []
        for depot in depots:
            vals = []
            for service_day in service_days:
                if (depot, service_day) in indexed.index:
                    vals.append(float(indexed.loc[(depot, service_day), field]))
                else:
                    vals.append(0.0)
            rows.append(vals)
        return rows

    customdata = []
    for depot in depots:
        row = []
        for service_day in service_days:
            if (depot, service_day) in indexed.index:
                rec = indexed.loc[(depot, service_day)]
                row.append([rec[field] for field in metric_fields])
            else:
                row.append([0 for _ in metric_fields])
        customdata.append(row)

    hovertemplate = (
        "<b>%{y} - %{x}</b><br>"
        "Total blocks: %{customdata[0]:,.0f}<br>"
        "<br><b>Depot-only</b><br>"
        "Success blocks: %{customdata[1]:,.0f}<br>"
        "Success rate: %{customdata[4]:.1f}%<br>"
        "BEB distance: %{customdata[7]:,.1f} km<br>"
        "<br><b>On-route</b><br>"
        "Success blocks: %{customdata[2]:,.0f}<br>"
        "Success rate: %{customdata[5]:.1f}%<br>"
        "BEB distance: %{customdata[8]:,.1f} km<br>"
        "<br><b>Change</b><br>"
        "Unlocked blocks: %{customdata[3]:+,.0f}<br>"
        "Rate delta: %{customdata[6]:+.1f} pts<br>"
        "Distance delta: %{customdata[9]:+,.1f} km"
        "<extra></extra>"
    )

    business_green_scale = [
        [0.00, "#f7faf7"],
        [0.25, "#dcebdd"],
        [0.50, "#a9cfac"],
        [0.75, "#5f9f68"],
        [1.00, "#2f6f3e"],
    ]

    def build_heatmap(metric: str, colorbar_title: str, text_format: str, text_suffix: str = ""):
        z = matrix_for(metric)
        text = [[f"{format(v, text_format)}{text_suffix}" for v in row] for row in z]
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=service_days,
                y=depots,
                customdata=customdata,
                text=text,
                texttemplate="%{text}",
                hovertemplate=hovertemplate,
                colorscale=business_green_scale,
                colorbar=dict(title=colorbar_title, outlinewidth=0),
            )
        )
        fig.update_layout(
            xaxis_title="Service day",
            yaxis_title="Depot",
            height=max(360, 110 + 32 * len(depots)),
            margin=dict(l=10, r=10, t=16, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return fig

    fig_blocks = build_heatmap(
        "unlocked_blocks",
        "Blocks",
        "+.0f",
    )
    fig_rate = build_heatmap(
        "success_rate_delta",
        "%",
        "+.1f",
        "%",
    )
    fig_km = build_heatmap(
        "beb_km_delta",
        "KM",
        "+.0f",
        " KM",
    )

    return fig_blocks, fig_rate, fig_km


def make_block_soc_plot(
    block_trips,
    accepted_events_df: pd.DataFrame,
    stop_name_map: Dict[int, str],
    candidate_df: pd.DataFrame | None = None,
    profile_onroute_df: pd.DataFrame | None = None,
    mode: str = "heavy",
    trip_to_route_short: Dict[str, str] | None = None,
):
    """
    Plot depot-only baseline vs current on-route FCFS scenario for one block.

    Visual rules:
    - depot-only: line only
    - on-route: line only
    - assigned charging session: red dots with detailed hover
    - not-assigned potential opportunity: gray dots with concise hover
    """

    def _normalize_plot_stop_code(series: pd.Series) -> pd.Series:
        vals = pd.to_numeric(series, errors="coerce")
        return vals.astype("Int64")

    def _add_location_columns(df: pd.DataFrame, scenario_label: str) -> pd.DataFrame:
        out = df.copy()
        if "stop_code" not in out.columns:
            out["stop_code"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        else:
            out["stop_code"] = _normalize_plot_stop_code(out["stop_code"])

        out["location_name"] = out["stop_code"].map(
            lambda x: stop_name_map.get(int(x), None) if pd.notna(x) else None
        )
        out["location_name"] = out["location_name"].where(out["location_name"].notna(), "N/A")
        out["stop_code_display"] = out["stop_code"].astype(str).replace("<NA>", "N/A")

        if "phase" not in out.columns:
            out["phase"] = "point"
        out["phase"] = out["phase"].fillna("point").astype(str)

        if "charge_kwh" not in out.columns:
            out["charge_kwh"] = 0.0
        if "charge_duration_sec" not in out.columns:
            out["charge_duration_sec"] = 0.0
        if "candidate_name" not in out.columns:
            out["candidate_name"] = None
        if "event_type" not in out.columns:
            out["event_type"] = None

        out["duration_min"] = pd.to_numeric(out["charge_duration_sec"], errors="coerce").fillna(0.0) / 60.0
        out["energy_kwh"] = pd.to_numeric(out["charge_kwh"], errors="coerce").fillna(0.0)
        out["scenario"] = scenario_label
        return out

    profile_depot = build_block_profile_with_charging(
        block_trips,
        matched_codes=set(),
        mode=mode,
    ).copy()

    if profile_onroute_df is not None and not profile_onroute_df.empty:
        profile_fcfs = profile_onroute_df.copy()
    else:
        assigned_points = []
        if accepted_events_df is not None and not accepted_events_df.empty:
            accepted_sorted = accepted_events_df.sort_values(["start_sec", "end_sec"], kind="stable").copy()
            for _, r in accepted_sorted.iterrows():
                assigned_points.append({
                    "start_sec": int(r["start_sec"]),
                    "end_sec": int(r["end_sec"]),
                    "duration_sec": int(r.get("duration_sec", int(r["end_sec"]) - int(r["start_sec"]))),
                    "stop_code": r.get("stop_code"),
                    "candidate_name": r.get("candidate_name"),
                    "event_type": r.get("event_type"),
                    "charged_kwh_request": r.get("charged_kwh"),
                })

        profile_fcfs = build_block_profile_with_assigned_events(
            block_trips=block_trips,
            assigned_points=assigned_points,
            mode=mode,
        ).copy()

    profile_depot = _add_location_columns(profile_depot, "Depot-only")
    profile_fcfs = _add_location_columns(profile_fcfs, "On-route (FCFS assigned)")

    fig = go.Figure()

    # Depot-only line: no detailed hover
    fig.add_trace(
        go.Scatter(
            x=profile_depot["dist_km"],
            y=profile_depot["soc_pct"],
            mode="lines",
            name="Depot-only",
            line=dict(width=3),
            hoverinfo="skip",
        )
    )

    # On-route line: no point detail hover
    fig.add_trace(
        go.Scatter(
            x=profile_fcfs["dist_km"],
            y=profile_fcfs["soc_pct"],
            mode="lines",
            name="On-route (FCFS assigned)",
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate=(
                "Distance: %{x:.2f} km<br>"
                "SOC: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    # Assigned charge overlay (red dots)
    pts = pd.DataFrame()
    needed_cols = {"phase", "dist_km", "soc_pct", "charge_kwh", "charge_duration_sec", "candidate_name", "event_type"}
    if needed_cols.issubset(profile_fcfs.columns):
        pts = profile_fcfs[profile_fcfs["phase"] == "charge"].copy()

        if not pts.empty:
            pts = pts.sort_values(["dist_km"], kind="stable").reset_index(drop=True)
            pts["start_soc_pct"] = None

            for i in pts.index:
                src_idx = pts.index[i]
                original_idx = pts.iloc[i].name

            charge_rows = profile_fcfs[profile_fcfs["phase"] == "charge"].copy()
            charge_rows = charge_rows.sort_index()

            start_soc_list = []
            for idx in charge_rows.index:
                prev_rows = profile_fcfs.loc[:idx - 1] if idx > profile_fcfs.index.min() else pd.DataFrame()
                prev_same_dist = prev_rows[prev_rows["dist_km"] == profile_fcfs.loc[idx, "dist_km"]]
                if not prev_same_dist.empty:
                    start_soc = float(prev_same_dist.iloc[-1]["soc_pct"])
                elif not prev_rows.empty:
                    start_soc = float(prev_rows.iloc[-1]["soc_pct"])
                else:
                    start_soc = float(profile_fcfs.loc[idx, "soc_pct"])
                start_soc_list.append(start_soc)

            charge_rows = charge_rows.copy()
            charge_rows["start_soc_pct"] = start_soc_list
            charge_rows["end_soc_pct"] = charge_rows["soc_pct"].astype(float)
            charge_rows["duration_min"] = pd.to_numeric(charge_rows["charge_duration_sec"], errors="coerce").fillna(0.0) / 60.0
            charge_rows["energy_kwh"] = pd.to_numeric(charge_rows["charge_kwh"], errors="coerce").fillna(0.0)
            charge_rows["candidate_label"] = (
                charge_rows["candidate_name"]
                .fillna(charge_rows["location_name"])
                .fillna("Unknown")
            )

            # Merge extra context from accepted_events_df
            if accepted_events_df is not None and not accepted_events_df.empty:
                event_ctx = accepted_events_df.copy()

                for c in [
                    "event_type",
                    "stop_code",
                    "start_sec",
                    "end_sec",
                    "prev_route_short_name",
                    "next_route_short_name",
                    "prev_trip_end_stop_name",
                    "next_trip_start_stop_name",
                ]:
                    if c not in event_ctx.columns:
                        event_ctx[c] = None

                event_ctx["stop_code"] = pd.to_numeric(event_ctx["stop_code"], errors="coerce").astype("Int64")

                if "stop_code" not in charge_rows.columns:
                    charge_rows["stop_code"] = pd.Series([pd.NA] * len(charge_rows), dtype="Int64")
                else:
                    charge_rows["stop_code"] = pd.to_numeric(charge_rows["stop_code"], errors="coerce").astype("Int64")

                for c in ["start_sec", "end_sec"]:
                    if c not in charge_rows.columns:
                        charge_rows[c] = pd.NA

                charge_rows = charge_rows.merge(
                    event_ctx[
                        [
                            "event_type",
                            "stop_code",
                            "start_sec",
                            "end_sec",
                            "prev_route_short_name",
                            "next_route_short_name",
                            "prev_trip_end_stop_name",
                            "next_trip_start_stop_name",
                        ]
                    ].drop_duplicates(),
                    on=["event_type", "stop_code", "start_sec", "end_sec"],
                    how="left",
                )
            else:
                charge_rows["prev_route_short_name"] = None
                charge_rows["next_route_short_name"] = None
                charge_rows["prev_trip_end_stop_name"] = None
                charge_rows["next_trip_start_stop_name"] = None

            fig.add_trace(
                go.Scatter(
                    x=charge_rows["dist_km"],
                    y=charge_rows["end_soc_pct"],
                    mode="markers",
                    name="Assigned charge",
                    marker=dict(size=9, color="red"),
                    customdata=charge_rows[
                        [
                            "candidate_label",
                            "dist_km",
                            "start_soc_pct",
                            "end_soc_pct",
                            "duration_min",
                            "energy_kwh",
                            "event_type",
                            "prev_route_short_name",
                            "next_route_short_name",
                            "prev_trip_end_stop_name",
                            "next_trip_start_stop_name",
                        ]
                    ].values,
                    hovertemplate=(
                        "<b>Assigned charging session</b><br>"
                        "Candidate Name: %{customdata[0]}<br>"
                        "Distance: %{customdata[1]:.2f} km<br>"
                        "Start SOC: %{customdata[2]:.2f}%<br>"
                        "End SOC: %{customdata[3]:.2f}%<br>"
                        "Duration: %{customdata[4]:.1f} min<br>"
                        "Energy Received: %{customdata[5]:.2f} kWh<br>"
                        "Type: %{customdata[6]}<br>"
                        "Previous Trip Route: %{customdata[7]}<br>"
                        "Next Trip Route: %{customdata[8]}<br>"
                        "Previous Trip End Station: %{customdata[9]}<br>"
                        "Next Trip Start Station: %{customdata[10]}<extra></extra>"
                    ),
                )
            )
    # Potential opportunities from structure
    opp_df = extract_potential_opportunities_for_block(
        block_trips=block_trips,
        stop_name_map=stop_name_map,
        candidate_df=candidate_df,
        trip_to_route_short=trip_to_route_short,
    )

    # Map opportunities onto ON-ROUTE line, not depot-only line
    if not opp_df.empty:
        fcfs_anchor = profile_fcfs[["dist_km", "soc_pct"]].drop_duplicates().sort_values("dist_km", kind="stable")
        opp_df = opp_df.sort_values("dist_km", kind="stable").copy()

        opp_plot = pd.merge_asof(
            opp_df,
            fcfs_anchor,
            on="dist_km",
            direction="nearest",
        )

        # Remove assigned opportunities from the gray-dot layer
        assigned_keys = set()
        if accepted_events_df is not None and not accepted_events_df.empty:
            tmp = accepted_events_df.copy()
            for _, r in tmp.iterrows():
                key = (
                    r.get("event_type"),
                    _normalize_stop_code(r.get("stop_code")),
                    None if pd.isna(r.get("start_sec")) else int(r.get("start_sec")),
                    None if pd.isna(r.get("end_sec")) else int(r.get("end_sec")),
                )
                assigned_keys.add(key)

        opp_plot["match_key"] = opp_plot.apply(
            lambda r: (
                r.get("event_type"),
                _normalize_stop_code(r.get("stop_code")),
                None if pd.isna(r.get("start_sec")) else int(r.get("start_sec")),
                None if pd.isna(r.get("end_sec")) else int(r.get("end_sec")),
            ),
            axis=1,
        )

        opp_unassigned = opp_plot[~opp_plot["match_key"].isin(assigned_keys)].copy()

        if not opp_unassigned.empty:
            opp_unassigned["candidate_label"] = (
                opp_unassigned["candidate_name"]
                .fillna(opp_unassigned["location_name"])
                .fillna("Unknown")
            )
            opp_unassigned["layover_duration_min"] = pd.to_numeric(
                opp_unassigned["layover_duration_min"], errors="coerce"
            ).fillna(0.0)

            for c in [
                "prev_route_short_name",
                "next_route_short_name",
                "prev_trip_end_stop_name",
                "next_trip_start_stop_name",
            ]:
                if c not in opp_unassigned.columns:
                    opp_unassigned[c] = None

            fig.add_trace(
                go.Scatter(
                    x=opp_unassigned["dist_km"],
                    y=opp_unassigned["soc_pct"],
                    mode="markers",
                    name="Not assigned opportunity",
                    marker=dict(size=8, color="gray", symbol="circle-open"),
                    customdata=opp_unassigned[
                        [
                            "candidate_label",
                            "dist_km",
                            "soc_pct",
                            "layover_duration_min",
                            "event_type",
                            "prev_route_short_name",
                            "next_route_short_name",
                            "prev_trip_end_stop_name",
                            "next_trip_start_stop_name",
                        ]
                    ].values,
                    hovertemplate=(
                        "<b>Not-assigned charging session</b><br>"
                        "Candidate Name: %{customdata[0]}<br>"
                        "Distance: %{customdata[1]:.2f} km<br>"
                        "SOC: %{customdata[2]:.2f}%<br>"
                        "Layover Duration: %{customdata[3]:.1f} min<br>"
                        "Type: %{customdata[4]}<br>"
                        "Previous Trip Route: %{customdata[5]}<br>"
                        "Next Trip Route: %{customdata[6]}<br>"
                        "Previous Trip End Station: %{customdata[7]}<br>"
                        "Next Trip Start Station: %{customdata[8]}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f"SOC vs Distance ({mode.capitalize()} duty)",
        legend_title_text="Scenario",
    )

    y_all = pd.concat(
        [
            profile_depot["soc_pct"],
            profile_fcfs["soc_pct"],
        ],
        ignore_index=True,
    )
    ymin = float(y_all.min()) if not y_all.empty else -5.0
    ymax = float(y_all.max()) if not y_all.empty else 100.0
    fig.update_yaxes(range=[ymin - 5.0, ymax + 5.0], title="soc_pct")
    fig.update_xaxes(title="dist_km")

    return fig, pts, opp_df


def extract_potential_opportunities_for_block(
    block_trips,
    stop_name_map: Dict[int, str],
    candidate_df: pd.DataFrame | None = None,
    trip_to_route_short: Dict[str, str] | None = None,
    layover_assume_min: int = 8,
    prep_time_min: int = 3,
) -> pd.DataFrame:
    """
    Extract all potential charging opportunities for plotting, including
    opportunities that are NOT candidate locations.
    """
    if not block_trips:
        return pd.DataFrame(
            columns=[
                "dist_km",
                "event_type",
                "stop_code",
                "location_name",
                "candidate_name",
                "is_candidate",
                "start_sec",
                "end_sec",
                "layover_duration_min",
                "prev_route_short_name",
                "next_route_short_name",
                "prev_trip_end_stop_name",
                "next_trip_start_stop_name",
            ]
        )

    stop_to_candidate = {}
    if candidate_df is not None and not candidate_df.empty:
        tmp = candidate_df[["stop_code", "candidate_name"]].dropna().copy()
        tmp["stop_code"] = pd.to_numeric(tmp["stop_code"], errors="coerce")
        tmp = tmp.dropna(subset=["stop_code"]).copy()
        tmp["stop_code"] = tmp["stop_code"].astype(int)
        tmp["candidate_name"] = tmp["candidate_name"].astype(str).str.strip()
        tmp = tmp.drop_duplicates(subset=["stop_code"], keep="first")
        stop_to_candidate = dict(zip(tmp["stop_code"], tmp["candidate_name"]))

    buffer_total = int(prep_time_min * 60)
    buffer_half = buffer_total // 2

    trips = []
    for t in block_trips:
        t2 = dict(t)
        if t2.get("start_time") is not None and t2.get("end_time") is not None:
            t2["start_sec"] = time_to_sec(t2["start_time"])
            t2["end_sec"] = time_to_sec(t2["end_time"])
        else:
            t2["start_sec"] = None
            t2["end_sec"] = None
        trips.append(t2)

    in_idxs = [i for i, t in enumerate(trips) if t.get("type") == "in_service"]
    first_in_idx = in_idxs[0] if in_idxs else None

    rows = []
    dist_cum_km = 0.0

    def add_opportunity(
        dist_km,
        event_type,
        stop_code,
        s0,
        s1,
        prev_trip=None,
        next_trip=None,
        prev_end_stop_code=None,
        next_start_stop_code=None,
    ):
        code_int = _normalize_stop_code(stop_code)

        candidate_name = stop_to_candidate.get(code_int) if code_int is not None else None
        if candidate_name is None and code_int is not None:
            candidate_name = stop_name_map.get(code_int, None)

        layover_duration_min = None
        if s0 is not None and s1 is not None and s1 > s0:
            layover_duration_min = (int(s1) - int(s0)) / 60.0

        prev_route_short_name = None
        next_route_short_name = None
        if trip_to_route_short is not None:
            if prev_trip is not None and prev_trip.get("trip_id") is not None:
                prev_route_short_name = trip_to_route_short.get(str(prev_trip.get("trip_id")))
            if next_trip is not None and next_trip.get("trip_id") is not None:
                next_route_short_name = trip_to_route_short.get(str(next_trip.get("trip_id")))

        rows.append(
            {
                "dist_km": float(dist_km),
                "event_type": event_type,
                "stop_code": code_int,
                "location_name": stop_name_map.get(code_int, "N/A") if code_int is not None else "N/A",
                "candidate_name": candidate_name,
                "is_candidate": code_int in stop_to_candidate if code_int is not None else False,
                "start_sec": None if s0 is None else int(s0),
                "end_sec": None if s1 is None else int(s1),
                "layover_duration_min": layover_duration_min,
                "prev_route_short_name": prev_route_short_name,
                "next_route_short_name": next_route_short_name,
                "prev_trip_end_stop_name": stop_name_map.get(_normalize_stop_code(prev_end_stop_code), "N/A")
                    if _normalize_stop_code(prev_end_stop_code) is not None else "N/A",
                "next_trip_start_stop_name": stop_name_map.get(_normalize_stop_code(next_start_stop_code), "N/A")
                    if _normalize_stop_code(next_start_stop_code) is not None else "N/A",
            }
        )

    for i, trip in enumerate(trips):
        ttype = trip.get("type")
        dist_km = _infer_trip_distance_km(trip)
        dist_end = dist_cum_km + dist_km

        if ttype in INTERLINE_TYPES and 0 < i < len(trips) - 1:
            prev_trip = trips[i - 1]
            next_trip = trips[i + 1]
            if prev_trip.get("type") == "in_service" and next_trip.get("type") == "in_service":
                inter_start_code = trip.get("start_stop_code")
                inter_end_code = trip.get("end_stop_code")
                assumed_charge = max(0, layover_assume_min * 60 - buffer_total)

                if assumed_charge > 0 and trip.get("start_sec") is not None:
                    s0 = trip["start_sec"] + buffer_half
                    s1 = s0 + assumed_charge
                    add_opportunity(
                        dist_cum_km, "interline_start", inter_start_code, s0, s1,
                        prev_trip=prev_trip,
                        next_trip=next_trip,
                        prev_end_stop_code=prev_trip.get("end_stop_code"),
                        next_start_stop_code=next_trip.get("start_stop_code"),
                    )

                dist_cum_km = dist_end

                if assumed_charge > 0 and trip.get("end_sec") is not None:
                    s1 = trip["end_sec"] - buffer_half
                    s0 = s1 - assumed_charge
                    add_opportunity(
                        dist_cum_km, "interline_end", inter_end_code, s0, s1,
                        prev_trip=prev_trip,
                        next_trip=next_trip,
                        prev_end_stop_code=prev_trip.get("end_stop_code"),
                        next_start_stop_code=next_trip.get("start_stop_code"),
                    )
                continue

        dist_cum_km = dist_end

        if ttype == "pull_out" and first_in_idx is not None:
            next_trip = trips[first_in_idx]
            prev_end_code = trip.get("end_stop_code")
            next_start_code = next_trip.get("start_stop_code")
            assumed_charge = max(0, layover_assume_min * 60 - buffer_total)

            if assumed_charge > 0 and trip.get("end_sec") is not None:
                s0 = trip["end_sec"] + buffer_half
                s1 = s0 + assumed_charge
                loc = _normalize_stop_code(prev_end_code)
                if loc is None:
                    loc = _normalize_stop_code(next_start_code)
                add_opportunity(
                    dist_cum_km, "pull_out_assumed", loc, s0, s1,
                    prev_trip=trip,
                    next_trip=next_trip,
                    prev_end_stop_code=prev_end_code,
                    next_start_stop_code=next_start_code,
                )
            continue

        if ttype == "in_service" and i < len(trips) - 1:
            next_trip = trips[i + 1]
            if next_trip.get("type") == "pull_in":
                continue

            if next_trip.get("type") == "in_service":
                prev_end_code = trip.get("end_stop_code")
                next_start_code = next_trip.get("start_stop_code")

                if trip.get("end_sec") is not None and next_trip.get("start_sec") is not None:
                    s0 = trip["end_sec"] + buffer_half
                    s1 = next_trip["start_sec"] - buffer_half
                    loc = _normalize_stop_code(prev_end_code)
                    if loc is None:
                        loc = _normalize_stop_code(next_start_code)
                    add_opportunity(
                        dist_cum_km, "in_service_layover", loc, s0, s1,
                        prev_trip=trip,
                        next_trip=next_trip,
                        prev_end_stop_code=prev_end_code,
                        next_start_stop_code=next_start_code,
                    )
            continue

    return pd.DataFrame(rows)

def add_depot_beb_distance_columns(
    report_df: pd.DataFrame,
    depot_summary_df: pd.DataFrame,
    duty: str,
) -> pd.DataFrame:
    """
    Add depot-only and on-route BEB distance KM columns to depot summary.
    """
    if report_df.empty or depot_summary_df.empty:
        out = depot_summary_df.copy()
        if "beb_km_depot" not in out.columns:
            out["beb_km_depot"] = 0.0
        if "beb_km_onroute" not in out.columns:
            out["beb_km_onroute"] = 0.0
        return out

    success_depot_col = f"{duty}_success_depot_only"
    success_onroute_col = f"{duty}_success_on_route_charge"

    base = report_df.copy()
    base["depot_code"] = base["depot_code"].astype(str)

    base = base.copy()

    dist_col = "total_distance_km" 
    base["beb_km_depot_tmp"] = base[dist_col].where(
        base[success_depot_col] == "SUCCESS", 0.0
    )

    base["beb_km_onroute_tmp"] = base[dist_col].where(
        base[success_onroute_col] == "SUCCESS", 0.0
    )

    km_by_depot = (
        base.groupby("depot_code", dropna=False)[
            ["beb_km_depot_tmp", "beb_km_onroute_tmp"]
        ]
        .sum()
        .reset_index()
        .rename(columns={
            "beb_km_depot_tmp": "beb_km_depot",
            "beb_km_onroute_tmp": "beb_km_onroute",
        })
    )

    out = depot_summary_df.copy()
    out["depot_code"] = out["depot_code"].astype(str)
    out = out.merge(km_by_depot, on="depot_code", how="left")

    out["beb_km_depot"] = pd.to_numeric(out["beb_km_depot"], errors="coerce").fillna(0.0)
    out["beb_km_onroute"] = pd.to_numeric(out["beb_km_onroute"], errors="coerce").fillna(0.0)

    return out



def render_onroute_panel():
    st.markdown("## On-Route Charge Summary Panel")
    inject_scrollable_dropdown_css()

    if "fcfs_scenario_cache" not in st.session_state:
        _clear_scenario_memory()
    if "fcfs_saved_scenarios_pruned" not in st.session_state:
        prune_scenario_artifacts(
            OUT_ROOT,
            keep_latest=MAX_SAVED_SCENARIOS,
            collection_name=RUNTIME_SCENARIO_COLLECTION,
        )
        st.session_state["fcfs_saved_scenarios_pruned"] = True

    with st.sidebar:
        st.header("Scenario controls")
        duty = st.radio("Duty / energy mode", ["Heavy-duty", "Medium-duty"], index=0)
        duty_key = "heavy" if duty.startswith("Heavy") else "medium"
        p1_scope = st.radio(
            "P1 block handling",
            ["Exclude P1 blocks", "Include P1 blocks"],
            index=0,
            horizontal=True,
        )
        exclude_p1 = p1_scope.startswith("Exclude")
        p1_scope_key = "exclude_p1" if exclude_p1 else "include_p1"

        st.header("Memory controls")
        save_scenario_outputs = st.checkbox(
            "Save scenario outputs to disk",
            value=False,
            help="Off by default to avoid accumulating large scenario files.",
        )
        saved_scenarios_to_keep = st.number_input(
            "Saved scenario folders to keep",
            min_value=1,
            max_value=50,
            value=MAX_SAVED_SCENARIOS,
            step=1,
            disabled=not save_scenario_outputs,
        )
        if st.button("Clear scenario memory", width="stretch"):
            _clear_scenario_memory()
            st.rerun()
        st.caption(
            f"In memory: current run plus up to {MAX_SESSION_SCENARIOS - 1} recent runs. "
            "Restarting the dashboard starts with an empty scenario memory."
        )

    blocks_base, candidate_df, events_df, disp_final = prepare_base_events(
        BLOCK_SUMMARY_PATH,
        CANDIDATE_STOP_MAP_PATH,
        GTFS_DIR,
        duty_key,
        proposal_mode="heavy",
        exclude_p1=exclude_p1,
    )

    if blocks_base.empty:
        st.warning("Block summary is empty.")
        st.stop()

    candidate_dict = build_candidate_dict(candidate_df)
    candidate_names = sorted(candidate_dict)

    proposed_map = (
        dict(zip(disp_final["candidate_name"], disp_final["final_proposed_dispensers"]))
        if not disp_final.empty
        else {n: 0 for n in candidate_names}
    )

    with st.sidebar:
        st.header("Dispenser controls")

        top_c1, top_c2 = st.columns(2)
        with top_c1:
            if st.button("Clear all", key=f"clear_all_{duty_key}_{p1_scope_key}", width="stretch"):
                _set_all_dispensers(candidate_names, proposed_map, duty_key, p1_scope_key, "clear")
                st.rerun()
        with top_c2:
            if st.button("Reset all", key=f"reset_all_{duty_key}_{p1_scope_key}", width="stretch"):
                _set_all_dispensers(candidate_names, proposed_map, duty_key, p1_scope_key, "reset")
                st.rerun()

        st.caption("Expand the list below to adjust installed dispensers by location.")

        installed_disp: Dict[str, int] = {}

        with st.expander("Locations", expanded=False):
            for name in candidate_names:
                max_n = int(proposed_map.get(name, 0))
                key = _disp_key(duty_key, p1_scope_key, name)

                if key not in st.session_state:
                    st.session_state[key] = max_n

                with st.expander(f"{name}  (proposed: {max_n})", expanded=False):
                    row_c1, row_c2 = st.columns(2)
                    with row_c1:
                        if st.button("Clear", key=f"clear_{duty_key}_{p1_scope_key}_{name}", width="stretch"):
                            _set_one_dispenser(name, proposed_map, duty_key, p1_scope_key, "clear")
                            st.rerun()
                    with row_c2:
                        if st.button("Reset", key=f"reset_{duty_key}_{p1_scope_key}_{name}", width="stretch"):
                            _set_one_dispenser(name, proposed_map, duty_key, p1_scope_key, "reset")
                            st.rerun()

                    st.number_input(
                        "Installed dispensers",
                        min_value=0,
                        max_value=max_n,
                        step=1,
                        key=key,
                        width="stretch",
                    )

                installed_disp[name] = int(st.session_state[key])

        st.caption("Reset = proposed value. Clear = 0.")

    scenario_key = _scenario_key(installed_disp, duty_key, exclude_p1)
    cache = st.session_state["fcfs_scenario_cache"]

    if scenario_key not in cache:
        with st.spinner("Allocating charging sessions and recomputing KPIs."):
            assigned_events_df, report_df, profiles_df = simulate_all_blocks_with_allocation_by_service_id(
                blocks_df=blocks_base,
                events_df=events_df,
                installed_disp=installed_disp,
                mode=duty_key,
            )

            assignment_summary_df = summarize_assignment(assigned_events_df)
            depot_summary_df = compute_depot_summary(report_df, duty=duty_key)
            service_day_summary_df = compute_service_day_summary(report_df, duty=duty_key)

            out_dir = None
            if save_scenario_outputs:
                out_dir = scenario_output_dir(
                    OUT_ROOT,
                    installed_disp,
                    scenario_key=scenario_key,
                    collection_name=RUNTIME_SCENARIO_COLLECTION,
                )
                persist_scenario_artifacts(
                    out_dir,
                    assigned_events_df,
                    report_df,
                    profiles_df,
                    assignment_summary_df,
                    write_parquet=False,
                    write_profiles=False,
                )
                prune_scenario_artifacts(
                    OUT_ROOT,
                    keep_latest=int(saved_scenarios_to_keep),
                    collection_name=RUNTIME_SCENARIO_COLLECTION,
                )

            _remember_scenario(scenario_key, {
                "assigned_events_df": assigned_events_df,
                "report_df": report_df,
                "profiles_df": profiles_df,
                "assignment_summary_df": assignment_summary_df,
                "depot_summary_df": depot_summary_df,
                "service_day_summary_df": service_day_summary_df,
                "out_dir": str(out_dir) if out_dir is not None else "",
            })
    else:
        _touch_scenario_cache_key(scenario_key)

    data = cache[scenario_key]
    assigned_events_df = data["assigned_events_df"]
    report_df = data["report_df"]
    profiles_df = data["profiles_df"]
    assignment_summary_df = data["assignment_summary_df"]
    depot_summary_df = data["depot_summary_df"]
    if save_scenario_outputs and not data.get("out_dir"):
        out_dir = scenario_output_dir(
            OUT_ROOT,
            installed_disp,
            scenario_key=scenario_key,
            collection_name=RUNTIME_SCENARIO_COLLECTION,
        )
        persist_scenario_artifacts(
            out_dir,
            assigned_events_df,
            report_df,
            profiles_df,
            assignment_summary_df,
            write_parquet=False,
            write_profiles=False,
        )
        prune_scenario_artifacts(
            OUT_ROOT,
            keep_latest=int(saved_scenarios_to_keep),
            collection_name=RUNTIME_SCENARIO_COLLECTION,
        )
        data["out_dir"] = str(out_dir)

    depot_summary_df = add_depot_beb_distance_columns(report_df=report_df, depot_summary_df=depot_summary_df, duty=duty_key)
    service_day_summary_df = data["service_day_summary_df"]




    success_depot_col = f"{duty_key}_success_depot_only"
    success_onroute_col = f"{duty_key}_success_on_route_charge"

    dep_ok = (report_df[success_depot_col] == "SUCCESS")
    on_ok = (report_df[success_onroute_col] == "SUCCESS")

    total_blocks = len(report_df)

    success_blocks_depot = int(dep_ok.sum())
    success_blocks_onroute = int(on_ok.sum())

    success_rate_depot = (success_blocks_depot / total_blocks * 100.0) if total_blocks > 0 else 0.0
    success_rate_onroute = (success_blocks_onroute / total_blocks * 100.0) if total_blocks > 0 else 0.0

    dist_col = "total_distance_km"

    beb_km_depot = float(report_df.loc[dep_ok, dist_col].sum()) if dist_col in report_df.columns else 0.0
    beb_km_onroute = float(report_df.loc[on_ok, dist_col].sum()) if dist_col in report_df.columns else 0.0

    st.markdown("### KPI Summary")

    d1, d2, d3 = st.columns(3)
    d1.metric("Depot-only Success blocks", f"{success_blocks_depot}")
    d2.metric("Depot-only Success rate", f"{success_rate_depot:.2f}%")
    d3.metric("Depot-only BEB distance KM", f"{beb_km_depot:,.1f}")

    o1, o2, o3 = st.columns(3)
    o1.metric("On-route Success blocks", f"{success_blocks_onroute}", delta=f"{success_blocks_onroute - success_blocks_depot:+d}")
    o2.metric("On-route Success rate", f"{success_rate_onroute:.2f}%", delta=f"{success_rate_onroute - success_rate_depot:+.2f} pts")
    o3.metric("On-route BEB distance KM", f"{beb_km_onroute:,.1f}", delta=f"{beb_km_onroute - beb_km_depot:+,.1f}")

    st.markdown("### Depot x service day comparison")

    fig_blocks, fig_rate, fig_km = make_depot_service_heatmaps(report_df, duty_key)

    top_c1, top_c2 = st.columns(2)
    with top_c1:
        st.markdown(f"#### Unlocked blocks (Blocks, {duty_key.capitalize()} duty)")
        st.plotly_chart(fig_blocks, width="stretch")
    with top_c2:
        st.markdown(f"#### Success rate improvement (%, {duty_key.capitalize()} duty)")
        st.plotly_chart(fig_rate, width="stretch")

    bot_c1, bot_c2 = st.columns(2)
    with bot_c1:
        st.markdown(f"#### BEB distance improvement (KM, {duty_key.capitalize()} duty)")
        st.plotly_chart(fig_km, width="stretch")
    with bot_c2:
        st.empty()

    st.markdown("### Depot-level summary")
    depot_summary_show = depot_summary_df.copy()

    preferred_cols = [
        "depot_code",
        "total_blocks",
        "success_blocks_depot_only",
        "success_rate_depot_only_%",
        "beb_km_depot",
        "success_blocks_on_route",
        "success_rate_on_route_%",
        "beb_km_onroute",
    ]

    show_cols = [c for c in preferred_cols if c in depot_summary_show.columns]
    depot_summary_show = depot_summary_show[show_cols].copy()
    for c in ["beb_km_depot", "beb_km_onroute", "success_rate_depot_only_%", "success_rate_on_route_%"]:
        if c in depot_summary_show.columns:
            depot_summary_show[c] = depot_summary_show[c].round(1)
    rename_map = {
        "depot_code": "Depot",
        "total_blocks": "Total blocks",
        "success_blocks_depot_only": "Depot-only success blocks",
        "success_rate_depot_only_%": "Depot-only success rate %",
        "beb_km_depot": "Depot-only BEB distance KM",
        "success_blocks_on_route": "On-route success blocks",
        "success_rate_on_route_%": "On-route success rate %",
        "beb_km_onroute": "On-route BEB distance KM",
    }
    depot_summary_show = depot_summary_show.rename(columns=rename_map)
    st.dataframe(depot_summary_show, width="stretch", hide_index=True)

    st.markdown("### Blocks unlocked by current location inputs")

    show_unlocked_only = st.toggle(
        "Only show blocks unlocked by on-route charging",
        value=True,
        help="When turned on, only show blocks where Depot-only = FAILURE and On-route = SUCCESS.",
    )

    successful_blocks_df = report_df[
        report_df[f"{duty_key}_success_on_route_charge"] == "SUCCESS"
    ].copy()

    if show_unlocked_only:
        successful_blocks_df = successful_blocks_df[
            (successful_blocks_df[f"{duty_key}_success_depot_only"] == "FAILURE")
            & (successful_blocks_df[f"{duty_key}_success_on_route_charge"] == "SUCCESS")
        ].copy()

    preferred_block_cols = [
        "depot_code",
        "service_day",
        "line_group",
        "block_number",
        "block_id",
        "block_distance_km",
        f"{duty_key}_success_depot_only",
        f"{duty_key}_success_on_route_charge",
    ]

    show_block_cols = [c for c in preferred_block_cols if c in successful_blocks_df.columns]
    successful_blocks_df = successful_blocks_df[show_block_cols].copy()

    rename_block_cols = {
        "depot_code": "Depot",
        "service_day": "Service day",
        "line_group": "Line group",
        "block_number": "Block number",
        "block_id": "Block ID",
        "block_distance_km": "Block distance KM",
        f"{duty_key}_success_depot_only": "Depot-only result",
        f"{duty_key}_success_on_route_charge": "On-route result",
    }
    successful_blocks_df = successful_blocks_df.rename(columns=rename_block_cols)

    if "Block distance KM" in successful_blocks_df.columns:
        successful_blocks_df["Block distance KM"] = pd.to_numeric(
            successful_blocks_df["Block distance KM"], errors="coerce"
        ).round(1)

    if show_unlocked_only:
        st.caption(f"Total unlocked blocks in current scenario: {len(successful_blocks_df)}")
    else:
        st.caption(f"Total on-route successful blocks in current scenario: {len(successful_blocks_df)}")

    st.dataframe(successful_blocks_df, width="stretch", hide_index=True)


    stop_name_map = build_combined_stop_name_map(BUS_STOPS_EXPORT_PATH, GTFS_DIR)    
    st.markdown("### Block drill-down")
    with st.sidebar:
        st.markdown("---")
        st.subheader("Block drill-down")
        success_col = f"{duty_key}_success_on_route_charge"
        depot_options = sorted(report_df["depot_code"].dropna().astype(str).unique().tolist())
        depot = st.selectbox("Depot", depot_options, key="onroute_drill_depot") if depot_options else None
        df1 = report_df[report_df["depot_code"].astype(str) == str(depot)] if depot is not None else report_df.copy()
        status_label = st.radio(
            "On-route FCFS result",
            ["Success", "Failure"],
            index=0,
            horizontal=True,
            key="onroute_drill_status",
        )
        status_value = "SUCCESS" if status_label == "Success" else "FAILURE"
        df2 = df1[df1[success_col] == status_value].copy()
        service_day_options = sorted(df2["service_day"].dropna().unique().tolist())
        service_day = st.selectbox(
            "Service day",
            service_day_options,
            key="onroute_drill_service_day",
        ) if service_day_options else None
        df3 = df2[df2["service_day"] == service_day].copy() if service_day is not None else df2
        lg_options = sorted(pd.to_numeric(df3["line_group"], errors="coerce").dropna().astype(int).unique().tolist())
        line_group = st.selectbox(
            "Line group",
            lg_options,
            key="onroute_drill_line_group",
        ) if lg_options else None
        df4 = df3[pd.to_numeric(df3["line_group"], errors="coerce").astype("Int64") == int(line_group)] if line_group is not None else df3
        block_options = sorted(pd.to_numeric(df4["block_number"], errors="coerce").dropna().astype(int).unique().tolist())
        block_number = st.selectbox(
            "Block number",
            block_options,
            key="onroute_drill_block_number",
        ) if block_options else None

    if block_number is not None and not df4.empty:
        df5 = df4[pd.to_numeric(df4["block_number"], errors="coerce").astype("Int64") == int(block_number)].copy()
        if not df5.empty:
            row_block = df5.iloc[0]
            sequence = parse_combined_sequence_json(row_block["combined_sequence_json"])
            block_events = assigned_events_df[
                (assigned_events_df["block_id"].astype(str) == str(row_block["block_id"]))
                & (assigned_events_df["mode"] == duty_key)
                & (assigned_events_df["assigned"] == True)
            ].copy()
            block_profile_onroute = profiles_df[
                (profiles_df["block_id"].astype(str) == str(row_block["block_id"]))
                & (profiles_df["mode"] == duty_key)
            ].copy()

            if not block_profile_onroute.empty and "dist_km" in block_profile_onroute.columns:
                block_profile_onroute = block_profile_onroute.sort_values("dist_km", kind="stable")
            trip_to_route_short = load_trip_to_route_short_name(GTFS_DIR)
            fig, pts, opp_df = make_block_soc_plot(
                sequence,
                block_events,
                stop_name_map=stop_name_map,
                candidate_df=candidate_df,
                profile_onroute_df=block_profile_onroute,
                mode=duty_key,
                trip_to_route_short=trip_to_route_short,
            )
            fig.add_hline(
                y=20,
                line_dash="dash",
                annotation_text="20% SOC threshold",
                annotation_position="top left",
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown("#### Assigned charging events for selected block")
            if not block_events.empty:
                show_cols = [
                    c for c in [
                        "candidate_name",
                        "stop_code",
                        "start_dt",
                        "end_dt",
                        "duration_min",
                        "charged_kwh",
                        "dispenser_idx",
                        "event_type",
                        "prev_route_short_name",
                        "next_route_short_name",
                        "prev_trip_end_stop_name",
                        "next_trip_start_stop_name",
                    ]
                    if c in block_events.columns
                ]

                rename_cols = {
                    "candidate_name": "Candidate",
                    "stop_code": "Stop code",
                    "start_dt": "Start time",
                    "end_dt": "End time",
                    "duration_min": "Duration (min)",
                    "charged_kwh": "Energy received (kWh)",
                    "dispenser_idx": "Assigned dispenser",
                    "event_type": "Session type",
                    "prev_route_short_name": "Previous trip route",
                    "next_route_short_name": "Next trip route",
                    "prev_trip_end_stop_name": "Previous trip end station",
                    "next_trip_start_stop_name": "Next trip start station",
                }

                sort_cols = [c for c in ["start_sec", "end_sec"] if c in block_events.columns]
                block_events_view = block_events.sort_values(sort_cols, kind="stable") if sort_cols else block_events.copy()
                block_events_view = block_events_view[show_cols].rename(columns=rename_cols)

                st.dataframe(block_events_view, width="stretch", hide_index=True)
            else:
                st.info("No assigned charging events for this block in the current scenario.")
