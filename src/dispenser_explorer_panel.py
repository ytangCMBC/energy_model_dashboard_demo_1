import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from heapq import heappush, heappop
from pathlib import Path

from onroute_fcfs_helpers_update import (
    build_final_proposed_dispensers,
    load_candidate_stop_map,
    parse_combined_sequence_json,
    simulate_all_blocks_with_allocation_by_service_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "processed"
DATA_DIR = PROJECT_ROOT / "data" 
GTFS_DIR = OUT_ROOT / "gtfs_bus_only"

EVENTS_PATH = OUT_ROOT / "charging_opportunity_output.csv"
DISPENSERS_PATH = OUT_ROOT / "final_proposed_dispensers.csv"
SUMMARY_PATH = OUT_ROOT / "block_success_summary_on_route_charger_564kwh.csv"
EVENTS_EXCLUDE_P1_PATH = OUT_ROOT / "charging_opportunity_output_exclude_p1.csv"
DISPENSERS_EXCLUDE_P1_PATH = OUT_ROOT / "final_proposed_dispensers_exclude_p1.csv"
SUMMARY_EXCLUDE_P1_PATH = OUT_ROOT / "block_success_summary_on_route_charger_564kwh_exclude_p1.csv"
BLOCK_SUMMARY_PATH = OUT_ROOT / "block_success_summary_depot_only.parquet"
CANDIDATE_STOP_MAP_PATH = OUT_ROOT / "candidate_stop_map.parquet"

MIN_SESSION_MINUTES = 0
BASE_DAY = pd.Timestamp("2026-01-20")


# =========================
# Helpers
# =========================
def _require_cols(df: pd.DataFrame, cols: set, name: str):
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _filter_p1_blocks(blocks: pd.DataFrame, duty_key: str) -> pd.DataFrame:
    success_col = f"{duty_key}_success_depot_only"
    if success_col not in blocks.columns:
        return blocks.copy()

    success = (
        blocks[success_col].eq(True)
        | blocks[success_col].astype(str).str.upper().eq("SUCCESS")
    )
    return blocks.loc[~success].copy()


def _load_blocks_for_export() -> pd.DataFrame:
    blocks = pd.read_parquet(BLOCK_SUMMARY_PATH)
    blocks = blocks[blocks["asset_class_new"] == "40-ft"].copy()
    blocks["block_trips"] = blocks["combined_sequence_json"].apply(parse_combined_sequence_json)
    blocks = blocks.rename(columns={
        "medium_success": "medium_success_depot_only",
        "heavy_success": "heavy_success_depot_only",
    })
    return blocks


def _assigned_events_for_export(assigned_events: pd.DataFrame) -> pd.DataFrame:
    assigned = assigned_events.copy()
    if assigned.empty:
        return assigned

    if "dispenser_idx" in assigned.columns:
        assigned = assigned.rename(columns={"dispenser_idx": "assigned_dispenser"})

    sort_cols = [c for c in ["mode", "service_id", "candidate_name", "start_sec", "end_sec", "block_id"] if c in assigned.columns]
    assigned = assigned.sort_values(sort_cols, kind="stable").reset_index(drop=True) if sort_cols else assigned.reset_index(drop=True)
    assigned.insert(0, "opportunity_id", range(1, len(assigned) + 1))

    output_cols = [
        "opportunity_id",
        "block_id",
        "line_group",
        "block_number",
        "asset_class_new",
        "depot_code",
        "service_id",
        "service_day",
        "mode",
        "prev_trip_id",
        "next_trip_id",
        "prev_route_short_name",
        "next_route_short_name",
        "prev_trip_end_stop_code",
        "next_trip_start_stop_code",
        "prev_trip_end_stop_name",
        "next_trip_start_stop_name",
        "stop_code",
        "candidate_name",
        "start_sec",
        "end_sec",
        "event_type",
        "soc_start_pct",
        "duration_sec",
        "duration_min",
        "soc_end_pct",
        "charged_kwh",
        "start_dt",
        "end_dt",
        "assigned",
        "assigned_dispenser",
    ]
    return assigned[[c for c in output_cols if c in assigned.columns]].copy()


def _summary_for_export(blocks: pd.DataFrame, medium_report: pd.DataFrame, heavy_report: pd.DataFrame) -> pd.DataFrame:
    summary = blocks.copy()

    rename_base = {
        "total_energy_medium_kwh": "total_energy_require_medium_kwh",
        "total_energy_heavy_kwh": "total_energy_require_heavy_kwh",
        "soc_left_medium_percent": "soc_left_medium_percent_depot_only",
        "soc_left_heavy_percent": "soc_left_heavy_percent_depot_only",
    }
    summary = summary.rename(columns=rename_base)

    medium_cols = [
        "block_id",
        "soc_left_medium_percent_on_route_charge",
        "medium_success_on_route_charge",
        "total_energy_received_medium",
    ]
    heavy_cols = [
        "block_id",
        "soc_left_heavy_percent_on_route_charge",
        "heavy_success_on_route_charge",
        "total_energy_received_heavy",
    ]

    if not medium_report.empty:
        summary = summary.merge(
            medium_report[[c for c in medium_cols if c in medium_report.columns]],
            on="block_id",
            how="left",
        )
    if not heavy_report.empty:
        summary = summary.merge(
            heavy_report[[c for c in heavy_cols if c in heavy_report.columns]],
            on="block_id",
            how="left",
        )

    output_cols = [
        "block_id",
        "line_group",
        "block_number",
        "service_id",
        "service_day",
        "depot_code",
        "asset_class",
        "asset_class_new",
        "total_distance_km",
        "total_energy_require_medium_kwh",
        "total_energy_require_heavy_kwh",
        "avg_kwh_per_km_medium",
        "avg_kwh_per_km_heavy",
        "soc_left_medium_percent_depot_only",
        "soc_left_heavy_percent_depot_only",
        "medium_success_depot_only",
        "heavy_success_depot_only",
        "soc_left_medium_percent_on_route_charge",
        "medium_success_on_route_charge",
        "soc_left_heavy_percent_on_route_charge",
        "heavy_success_on_route_charge",
        "total_energy_received_medium",
        "total_energy_received_heavy",
    ]
    return summary[[c for c in output_cols if c in summary.columns]].copy()


@st.cache_data(show_spinner=False)
def _ensure_exclude_p1_outputs() -> tuple[str, str, str]:
    paths = (EVENTS_EXCLUDE_P1_PATH, DISPENSERS_EXCLUDE_P1_PATH, SUMMARY_EXCLUDE_P1_PATH)
    if all(path.exists() for path in paths):
        return tuple(str(path) for path in paths)

    blocks = _load_blocks_for_export()
    candidate_df = load_candidate_stop_map(CANDIDATE_STOP_MAP_PATH)

    heavy_blocks = _filter_p1_blocks(blocks, "heavy")
    medium_blocks = _filter_p1_blocks(blocks, "medium")

    heavy_events, disp_final = build_final_proposed_dispensers(
        blocks_df=heavy_blocks,
        candidate_df=candidate_df,
        gtfs_dir=GTFS_DIR,
        mode="heavy",
    )
    medium_events, _ = build_final_proposed_dispensers(
        blocks_df=medium_blocks,
        candidate_df=candidate_df,
        gtfs_dir=GTFS_DIR,
        mode="medium",
    )

    proposed_map = (
        dict(zip(disp_final["candidate_name"], disp_final["final_proposed_dispensers"]))
        if not disp_final.empty
        else {}
    )

    heavy_assigned, heavy_report, _ = simulate_all_blocks_with_allocation_by_service_id(
        blocks_df=blocks,
        events_df=heavy_events,
        installed_disp=proposed_map,
        mode="heavy",
    )
    medium_assigned, medium_report, _ = simulate_all_blocks_with_allocation_by_service_id(
        blocks_df=blocks,
        events_df=medium_events,
        installed_disp=proposed_map,
        mode="medium",
    )

    charging_output = pd.concat(
        [
            _assigned_events_for_export(heavy_assigned),
            _assigned_events_for_export(medium_assigned),
        ],
        ignore_index=True,
    )
    if not charging_output.empty:
        charging_output["opportunity_id"] = range(1, len(charging_output) + 1)

    summary_output = _summary_for_export(blocks, medium_report, heavy_report)

    charging_output.to_csv(EVENTS_EXCLUDE_P1_PATH, index=False)
    disp_final.to_csv(DISPENSERS_EXCLUDE_P1_PATH, index=False)
    summary_output.to_csv(SUMMARY_EXCLUDE_P1_PATH, index=False)

    return tuple(str(path) for path in paths)


def _attach_block_category(events_df: pd.DataFrame, block_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge block-level success fields into charging-event records and create P-category.
    Category is based on HEAVY success columns, same as old version.
    """
    need_cols = {
        "block_id",
        "heavy_success_depot_only",
        "heavy_success_on_route_charge",
        "medium_success_depot_only",
        "medium_success_on_route_charge",
    }
    _require_cols(block_df, need_cols, "block_success_summary_on_route")

    base = block_df.loc[:, list(need_cols)].copy()
    base["block_id"] = base["block_id"].astype(str)
    base = base.drop_duplicates(subset=["block_id"]).copy()

    out = events_df.copy()
    out["block_id"] = out["block_id"].astype(str)
    out = out.merge(base, on="block_id", how="left")

    cond_p1 = (
        out["heavy_success_depot_only"].eq("SUCCESS")
        & out["heavy_success_on_route_charge"].eq("SUCCESS")
    )
    cond_p2 = (
        out["heavy_success_depot_only"].eq("FAILURE")
        & out["heavy_success_on_route_charge"].eq("SUCCESS")
    )
    cond_p3 = (
        out["heavy_success_depot_only"].eq("FAILURE")
        & out["heavy_success_on_route_charge"].eq("FAILURE")
    )

    out["block_category"] = np.select(
        [cond_p1, cond_p2, cond_p3],
        [
            "P1 blocks (depot only)",
            "P2 blocks (depot fail on-route success)",
            "P3 blocks (both fail)",
        ],
        default="Other / unmatched",
    )

    return out


def _format_hour_window(h: int) -> str:
    return f"{h}:00:00 - {h+1}:00:00"


def _format_minute_label(t0_sec: int, minute_idx: int) -> str:
    t = t0_sec + minute_idx * 60
    h = t // 3600
    m = (t % 3600) // 60
    return f"{int(h)}:{int(m):02d}"


def _build_constant_segments(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["seg_start", "seg_end", "conc"])

    starts = events["start_sec"].to_numpy(dtype=float)
    ends = events["end_sec"].to_numpy(dtype=float)

    marks = []
    for s, e in zip(starts, ends):
        marks.append((float(s), +1, 1))
        marks.append((float(e), -1, 0))

    marks.sort(key=lambda x: (x[0], x[2]))

    cur = 0
    prev_t = None
    segs = []

    for t, delta, _ in marks:
        if prev_t is not None and t > prev_t and cur > 0:
            segs.append((prev_t, t, cur))
        cur += delta
        prev_t = t

    return pd.DataFrame(segs, columns=["seg_start", "seg_end", "conc"])


def _minute_peak_busyness_any_window(events: pd.DataFrame, window_start_sec: int, window_end_sec: int) -> pd.DataFrame:
    if events.empty or window_end_sec <= window_start_sec:
        return pd.DataFrame({"minute_idx": np.array([], dtype=int), "active_sessions": np.array([], dtype=int)})

    ev = events[(events["start_sec"] < window_end_sec) & (events["end_sec"] > window_start_sec)].copy()
    n_minutes = int(np.ceil((window_end_sec - window_start_sec) / 60.0))

    if ev.empty:
        return pd.DataFrame({
            "minute_idx": np.arange(n_minutes, dtype=int),
            "active_sessions": np.zeros(n_minutes, dtype=int)
        })

    ev["start_sec"] = ev["start_sec"].clip(lower=window_start_sec, upper=window_end_sec)
    ev["end_sec"] = ev["end_sec"].clip(lower=window_start_sec, upper=window_end_sec)

    segs = _build_constant_segments(ev)
    peaks = np.zeros(n_minutes, dtype=int)

    if segs.empty:
        return pd.DataFrame({"minute_idx": np.arange(n_minutes, dtype=int), "active_sessions": peaks})

    t0 = float(window_start_sec)

    for a, b, c in segs.itertuples(index=False):
        i0 = int(np.floor((a - t0) / 60.0))
        i1 = int(np.ceil((b - t0) / 60.0)) - 1
        if i1 < 0 or i0 >= n_minutes:
            continue
        i0 = max(i0, 0)
        i1 = min(i1, n_minutes - 1)
        if i0 <= i1:
            peaks[i0:i1 + 1] = np.maximum(peaks[i0:i1 + 1], int(c))

    return pd.DataFrame({"minute_idx": np.arange(n_minutes, dtype=int), "active_sessions": peaks})


def _hour_stats(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["hour", "label", "peak_active", "total_active_sessions_minutes", "sessions_in_hour"])

    h_min = int(np.floor(events["start_sec"].min() / 3600.0))
    h_max = int(np.ceil(events["end_sec"].max() / 3600.0))
    hours = list(range(h_min, h_max))

    rows = []
    for h in hours:
        hs = h * 3600
        he = hs + 3600
        sub = events[(events["start_sec"] < he) & (events["end_sec"] > hs)]
        minute_df = _minute_peak_busyness_any_window(sub, hs, he)
        peak_active = int(minute_df["active_sessions"].max()) if not minute_df.empty else 0
        total_active = int(minute_df["active_sessions"].sum()) if not minute_df.empty else 0

        rows.append({
            "hour": h,
            "label": _format_hour_window(h),
            "peak_active": peak_active,
            "total_active_sessions_minutes": total_active,
            "sessions_in_hour": int(len(sub)),
        })

    return pd.DataFrame(rows)


def _required_dispensers_for_events(events: pd.DataFrame) -> int:
    if events.empty:
        return 0

    t0 = int(np.floor(events["start_sec"].min() / 60.0) * 60)
    t1 = int(np.ceil(events["end_sec"].max() / 60.0) * 60)

    minute_df = _minute_peak_busyness_any_window(events, t0, t1)
    if minute_df.empty:
        return 0

    return int(minute_df["active_sessions"].max())


def _build_block_label(df: pd.DataFrame) -> pd.Series:
    if "line_group" in df.columns and "block_number" in df.columns:
        return "LG " + df["line_group"].astype(int).astype(str) + "-BLK " + df["block_number"].astype(int).astype(str)
    return df["block_id"].astype(str)


def _fcfs_sessions_supported(events: pd.DataFrame, k: int) -> tuple[int, int]:
    if events.empty or k <= 0:
        return 0, 0 if events.empty else int(len(events))

    df = events.loc[:, ["start_sec", "end_sec"]].copy()
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    df = df.dropna(subset=["start_sec", "end_sec"])
    df = df[df["end_sec"] > df["start_sec"]].copy()

    total_sessions = len(df)
    if total_sessions == 0:
        return 0, 0

    df = df.sort_values(["start_sec", "end_sec"], kind="mergesort").reset_index(drop=True)

    heap = []
    served = 0

    for _, row in df.iterrows():
        s = float(row["start_sec"])
        e = float(row["end_sec"])

        while heap and heap[0] <= s:
            heappop(heap)

        if len(heap) < k:
            heappush(heap, e)
            served += 1

    return served, total_sessions


def _coverage_curve_sessions_fcfs(events: pd.DataFrame, max_k: int) -> pd.DataFrame:
    rows = []
    for k in range(1, max_k + 1):
        served, total = _fcfs_sessions_supported(events, k)
        rows.append({
            "dispensers": k,
            "served_sessions": served,
            "total_sessions": total,
            "coverage_pct_sessions": (100.0 * served / total) if total else np.nan,
        })
    return pd.DataFrame(rows)


def _fcfs_blocks_supported(events: pd.DataFrame, k: int) -> tuple[int, int]:
    if events.empty or k <= 0:
        return 0, int(events["block_id"].nunique()) if "block_id" in events.columns else 0

    df = events.loc[:, ["block_id", "start_sec", "end_sec"]].copy()
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    df = df.dropna(subset=["block_id", "start_sec", "end_sec"])
    df = df[df["end_sec"] > df["start_sec"]].copy()

    all_blocks = set(df["block_id"].astype(str).unique())
    total_blocks = len(all_blocks)
    if total_blocks == 0:
        return 0, 0

    df["block_id"] = df["block_id"].astype(str)
    df = df.sort_values(["start_sec", "end_sec", "block_id"], kind="mergesort")

    heap = []
    dropped_blocks = set()

    for r in df.itertuples(index=False):
        s = float(r.start_sec)
        e = float(r.end_sec)
        b = r.block_id

        while heap and heap[0] <= s:
            heappop(heap)

        if len(heap) < k:
            heappush(heap, e)
        else:
            dropped_blocks.add(b)

    supported_blocks = len(all_blocks - dropped_blocks)
    return supported_blocks, total_blocks


def _coverage_curve_blocks_fcfs(events: pd.DataFrame, max_k: int) -> pd.DataFrame:
    rows = []
    for k in range(1, max_k + 1):
        sup, tot = _fcfs_blocks_supported(events, k)
        rows.append({
            "dispensers": k,
            "blocks_supported": sup,
            "blocks_total": tot,
            "coverage_pct_blocks": (100.0 * sup / tot) if tot else np.nan,
        })
    return pd.DataFrame(rows)


def _get_final_proposed_dispensers(disp_df: pd.DataFrame, candidate_name: str) -> int:
    row = disp_df[disp_df["candidate_name"] == candidate_name]
    if row.empty:
        return 0
    return int(row.iloc[0].get("final_proposed_dispensers", 0))


SERVICE_DAY_TO_MAP = {
    "MF": DATA_DIR / "terminal_stations_grouped_map_MF_service_id_1.html",
    "SAT": DATA_DIR / "terminal_stations_grouped_map_Sat_service_id_2.html",
    "SUN": DATA_DIR / "terminal_stations_grouped_map_Sun_service_id_3.html",
}


def _render_service_day_map(service_day: str):
    """
    Render the pre-generated terminal station map that matches the selected service day.
    """
    map_path = SERVICE_DAY_TO_MAP.get(str(service_day))
    st.markdown("### Terminal station heat map")

    if map_path is None:
        st.info(f"No map is configured for service day: {service_day}")
        return

    if not map_path.exists():
        st.warning(
            f"Map file not found for {service_day}: {map_path.name}. "
            "Please make sure the HTML map has been generated and saved under /data."
        )
        return

    st.iframe(map_path, height=700, width="stretch")



# =========================
# App
# =========================


def render_dispenser_explorer_panel():
    # =========================
    # Sidebar filters
    # =========================
    st.sidebar.header("Scenario filters")
    p1_scope = st.sidebar.radio(
        "P1 block handling",
        ["Exclude P1 blocks", "Include P1 blocks"],
        index=0,
        horizontal=True,
        key="dispenser_p1_scope",
    )
    exclude_p1 = p1_scope.startswith("Exclude")
    p1_scope_key = "exclude_p1" if exclude_p1 else "include_p1"

    if exclude_p1:
        try:
            with st.spinner("Preparing exclude-P1 dispenser explorer outputs."):
                _ensure_exclude_p1_outputs()
        except Exception as exc:
            st.error(f"Could not prepare exclude-P1 output files: {exc}")
            st.stop()

    events_path = EVENTS_EXCLUDE_P1_PATH if exclude_p1 else EVENTS_PATH
    dispensers_path = DISPENSERS_EXCLUDE_P1_PATH if exclude_p1 else DISPENSERS_PATH
    summary_path = SUMMARY_EXCLUDE_P1_PATH if exclude_p1 else SUMMARY_PATH

    events_df = pd.read_csv(events_path)
    disp_df = pd.read_csv(dispensers_path)
    block_df = pd.read_csv(summary_path)

    _require_cols(
        events_df,
        {"candidate_name", "start_sec", "end_sec", "block_id", "service_day", "mode", "assigned"},
        events_path.name,
    )
    _require_cols(
        disp_df,
        {"candidate_name", "final_proposed_dispensers"},
        dispensers_path.name,
    )

    events_df = events_df.copy()
    events_df["candidate_name"] = events_df["candidate_name"].astype(str)
    events_df["block_id"] = events_df["block_id"].astype(str)
    events_df["service_day"] = events_df["service_day"].astype(str)
    events_df["mode"] = events_df["mode"].astype(str).str.lower()
    events_df["start_sec"] = pd.to_numeric(events_df["start_sec"], errors="coerce")
    events_df["end_sec"] = pd.to_numeric(events_df["end_sec"], errors="coerce")

    events_df = events_df.dropna(subset=["candidate_name", "start_sec", "end_sec"])
    events_df = events_df[events_df["end_sec"] > events_df["start_sec"]].copy()

    if MIN_SESSION_MINUTES and MIN_SESSION_MINUTES > 0:
        min_sec = MIN_SESSION_MINUTES * 60.0
        events_df = events_df[(events_df["end_sec"] - events_df["start_sec"]) >= min_sec].copy()

    events_df = _attach_block_category(events_df, block_df)

    mode_options = sorted(events_df["mode"].dropna().unique().tolist())
    sel_mode = st.sidebar.selectbox("Duty mode", mode_options, index=0)

    service_day_options = sorted(events_df["service_day"].dropna().unique().tolist())
    sel_service_day = st.sidebar.selectbox("Service day", service_day_options, index=0)

    category_options = [
        "P1 blocks (depot only)",
        "P2 blocks (depot fail on-route success)",
        "P3 blocks (both fail)",
    ]
    default_categories = category_options[1:] if exclude_p1 else category_options
    sel_categories = st.sidebar.multiselect(
        "Block categories",
        options=category_options,
        default=default_categories,
        key=f"dispenser_categories_{p1_scope_key}",
    )

    show_assigned_only = st.sidebar.checkbox("Assigned sessions only", value=False)

    # scenario base: KPI should use this, BEFORE assigned-only filter
    scenario_base_df = events_df[
        (events_df["mode"] == sel_mode) &
        (events_df["service_day"] == sel_service_day)
    ].copy()

    if sel_categories:
        scenario_base_df = scenario_base_df[
            scenario_base_df["block_category"].isin(sel_categories)
        ].copy()
    else:
        scenario_base_df = scenario_base_df.iloc[0:0].copy()

    # display df: this can respond to assigned-only
    filtered_for_display = scenario_base_df.copy()
    if show_assigned_only:
        filtered_for_display = filtered_for_display[
            filtered_for_display["assigned"] == True
        ].copy()

    candidates = sorted(filtered_for_display["candidate_name"].dropna().unique().tolist())
    if not candidates:
        st.warning("No charging sessions found for the selected filters.")
        st.stop()

    _render_service_day_map(sel_service_day)

    sel_candidate = st.selectbox(
        "Select a candidate on-route charging location",
        candidates,
        index=0,
    )

    # KPI source: scenario logic before assigned-only
    cand_events_kpi = scenario_base_df[
        scenario_base_df["candidate_name"] == sel_candidate
    ].copy()

    # display source: after assigned-only
    cand_events = filtered_for_display[
        filtered_for_display["candidate_name"] == sel_candidate
    ].copy()


    # =========================
    # KPIs
    # =========================
    scenario_disp_needed = _required_dispensers_for_events(cand_events_kpi)
    disp_proposed = _get_final_proposed_dispensers(disp_df, sel_candidate)
    total_sessions = int(len(cand_events))
    unique_blocks = int(cand_events["block_id"].nunique())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Scenario dispensers needed", scenario_disp_needed)
    kpi2.metric("Dispensers proposed", disp_proposed)
    kpi3.metric("Charging sessions", total_sessions)
    kpi4.metric("Unique blocks", unique_blocks)

    st.caption(
        f"Mode: {sel_mode} | Service day: {sel_service_day} | "
        f"P1 handling: {p1_scope} | "
        f"Categories: {', '.join(sel_categories) if sel_categories else 'None'}"
    )

    # =========================
    # Coverage vs dispensers (blocks)
    # =========================
    st.markdown("### Block Coverage vs. number of dispensers")

    if cand_events.empty:
        st.info("No sessions for this candidate under the selected scenario.")
    else:
        max_k = max(1, scenario_disp_needed, disp_proposed)
        max_k = min(int(max_k), 40)

        curve_blocks = _coverage_curve_blocks_fcfs(cand_events, max_k=max_k)

        fig_cov_blocks = px.line(
            curve_blocks,
            x="dispensers",
            y="coverage_pct_blocks",
            markers=True,
            hover_data={
                "blocks_supported": True,
                "blocks_total": True,
                "coverage_pct_blocks": ":.2f",
            },
        )
        fig_cov_blocks.update_layout(
            xaxis_title="Number of dispensers at this location",
            yaxis_title="% of blocks fully supported",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 102]),
        )
        fig_cov_blocks.update_xaxes(dtick=1)

        if disp_proposed > 0:
            fig_cov_blocks.add_vline(
                x=disp_proposed,
                line_dash="dash",
                annotation_text="Proposed",
                annotation_position="top left",
            )

        st.plotly_chart(fig_cov_blocks, width="stretch")

    # =========================
    # Coverage vs dispensers (sessions)
    # =========================
    st.markdown("### Charging Session Coverage vs. number of dispensers")

    if cand_events.empty:
        st.info("No sessions for this candidate.")
    else:
        max_k = max(1, scenario_disp_needed, disp_proposed)
        max_k = min(int(max_k), 30)

        curve_df = _coverage_curve_sessions_fcfs(cand_events, max_k=max_k)

        fig_cov = px.line(
            curve_df,
            x="dispensers",
            y="coverage_pct_sessions",
            markers=True,
            hover_data={
                "served_sessions": True,
                "total_sessions": True,
                "coverage_pct_sessions": ":.2f",
            },
        )
        fig_cov.update_layout(
            xaxis_title="Number of dispensers at this location",
            yaxis_title="% of charging sessions that can be covered",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 102]),
        )
        fig_cov.update_xaxes(dtick=1)

        if disp_proposed > 0:
            fig_cov.add_vline(
                x=disp_proposed,
                line_dash="dash",
                annotation_text="Proposed",
                annotation_position="top left",
            )

        st.plotly_chart(fig_cov, width="stretch")

    st.divider()

    # =========================
    # Hour selection
    # =========================
    hour_tbl = _hour_stats(cand_events)
    if hour_tbl.empty:
        st.warning("No valid hourly windows for this candidate.")
        st.stop()

    hour_tbl = hour_tbl.sort_values(
        ["peak_active", "total_active_sessions_minutes", "sessions_in_hour", "hour"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    sel_label = st.selectbox(
        "Select an hourly time window (sorted by busiest → emptiest)",
        hour_tbl["label"].tolist(),
        index=0,
    )

    sel_hour = int(hour_tbl.loc[hour_tbl["label"] == sel_label, "hour"].iloc[0])
    window_start = sel_hour * 3600
    window_end = window_start + 3600
    window_title = sel_label

    win_events = cand_events[(cand_events["start_sec"] < window_end) & (cand_events["end_sec"] > window_start)].copy()
    minute_df = _minute_peak_busyness_any_window(win_events, window_start, window_end)

    # =========================
    # Gantt
    # =========================
    st.markdown(f"### Gantt — {window_title}")

    if win_events.empty:
        st.info("No sessions overlap the selected hour.")
    else:
        plot_df = win_events.copy()
        plot_df["clip_start"] = plot_df["start_sec"].clip(lower=window_start, upper=window_end)
        plot_df["clip_end"] = plot_df["end_sec"].clip(lower=window_start, upper=window_end)
        plot_df["block_label"] = _build_block_label(plot_df)

        plot_df["start_ts"] = BASE_DAY + pd.to_timedelta(plot_df["clip_start"].astype(float), unit="s")
        plot_df["end_ts"] = BASE_DAY + pd.to_timedelta(plot_df["clip_end"].astype(float), unit="s")

        block_order_df = (
            plot_df.groupby("block_label", as_index=False)["start_ts"]
            .min()
            .sort_values(["start_ts", "block_label"], ascending=[True, True])
            .reset_index(drop=True)
        )
        order = block_order_df["block_label"].tolist()

        plot_df["block_label"] = pd.Categorical(
            plot_df["block_label"],
            categories=order,
            ordered=True
        )

        plot_df = plot_df.sort_values(
            ["block_label", "start_ts", "end_ts", "depot_code"],
            ascending=[True, True, True, True]
        ).reset_index(drop=True)

        hover_cols = [c for c in [
            "block_id", "line_group", "block_number",
            "depot_code", "service_day", 
            "event_type", 
            "soc_start_pct", "soc_end_pct", "charged_kwh",
            "prev_route_short_name", "next_route_short_name",
            "prev_trip_end_stop_name", "next_trip_start_stop_name",
        ] if c in plot_df.columns]

        fig_gantt = px.timeline(
            plot_df,
            x_start="start_ts",
            x_end="end_ts",
            y="block_label",
            color="depot_code" if "depot_code" in plot_df.columns else None,
            hover_data=hover_cols,
            category_orders={"block_label": order},
            title=None,
        )

        x0 = BASE_DAY + pd.to_timedelta(window_start, unit="s")
        x1 = BASE_DAY + pd.to_timedelta(window_end, unit="s")

        fig_gantt.update_layout(
            xaxis_title="Time",
            yaxis_title="Block",
            height=min(950, 140 + 22 * plot_df["block_label"].nunique()),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_gantt.update_xaxes(
            tickformat="%H:%M",
            range=[x0, x1],
            rangeslider_visible=False,
        )
        fig_gantt.update_yaxes(
            autorange="reversed",
            categoryorder="array",
            categoryarray=order,
        )

        st.plotly_chart(fig_gantt, width="stretch")

    # =========================
    # Busyness
    # =========================
    st.markdown(f"### Busyness (per-minute peak concurrency) — {window_title}")

    if minute_df.empty:
        st.info("No minutes to display in this hour.")
    else:
        minute_plot = minute_df.copy()
        minute_plot["time_hhmm"] = minute_plot["minute_idx"].map(lambda i: _format_minute_label(window_start, int(i)))
        minute_plot["over_capacity"] = (minute_plot["active_sessions"] > disp_proposed) if disp_proposed > 0 else False

        if disp_proposed > 0:
            over_minutes = int(minute_plot["over_capacity"].sum())
            # st.caption(
            #     f"Peak concurrent sessions: **{int(minute_plot['active_sessions'].max())}** | "
            #     f"Minutes over proposed capacity ({disp_proposed} dispensers): **{over_minutes}/60**"
            # )

        fig_busy = px.bar(
            minute_plot,
            x="minute_idx",
            y="active_sessions",
            hover_data={"time_hhmm": True, "active_sessions": True, "over_capacity": True},
            title=None,
        )
        # fig_busy.update_traces(
        #     hovertemplate="Time: %{customdata[0]}<br>Peak concurrent sessions: %{y}<br>Over proposed capacity: %{customdata[2]}<extra></extra>"
        # )
        fig_busy.update_layout(
            xaxis_title="Minute within hour (0–59)",
            yaxis_title="Peak concurrent sessions in minute",
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_busy.update_xaxes(dtick=5)

        if disp_proposed > 0:
            fig_busy.add_hline(
                y=disp_proposed,
                line_dash="dash",
                annotation_text=f"Proposed capacity ({disp_proposed})",
                annotation_position="top left",
            )
            ymax = max(int(minute_plot["active_sessions"].max()), disp_proposed)
            fig_busy.add_hrect(
                y0=disp_proposed,
                y1=ymax,
                opacity=0.08,
                line_width=0,
                annotation_text="Over capacity",
                annotation_position="top right",
            )

        st.plotly_chart(fig_busy, width="stretch")

    # =========================
    # Raw records
    # =========================
    raw_scope = st.radio(
        "Raw charging-session records scope",
        ["Selected hour", "Whole day"],
        index=0,
        horizontal=True,
        key="dispenser_raw_records_scope",
    )
    raw_events = win_events if raw_scope == "Selected hour" else cand_events
    raw_scope_note = (
        f"Selected hour: {window_title}"
        if raw_scope == "Selected hour"
        else "Whole selected service day"
    )

    st.markdown(f"### Raw charging-session records ({raw_scope.lower()})")
    st.caption(
        f"{raw_scope_note} | Candidate: {sel_candidate} | Mode: {sel_mode} | "
        f"Service day: {sel_service_day} | P1 handling: {p1_scope} | "
        f"Categories: {', '.join(sel_categories) if sel_categories else 'None'}"
    )
    if raw_events.empty:
        st.info(
            "No records in the selected hour."
            if raw_scope == "Selected hour"
            else "No records for the selected day and filters."
        )
    else:
        raw_cols = [
            "candidate_name",
            "block_category",
            "block_id",
            "line_group",
            "block_number",
            "depot_code",
            "service_day",
            "mode",
            "assigned",
            "assigned_dispenser",
            "heavy_success_depot_only",
            "heavy_success_on_route_charge",
            "medium_success_depot_only",
            "medium_success_on_route_charge",
            "start_dt",
            "end_dt",
            "event_type",
            "soc_start_pct",
            "soc_end_pct",
            "charged_kwh",
            "duration_min",
            "prev_route_short_name",
            "next_route_short_name",
            "prev_trip_end_stop_name",
            "next_trip_start_stop_name",
            "start_sec", 
            "end_sec"
        ]
        raw_cols = [c for c in raw_cols if c in raw_events.columns]
        raw_df = raw_events.loc[:, raw_cols].copy()
        raw_df = raw_df.sort_values(["start_sec", "end_sec"], ascending=[True, True]).reset_index(drop=True)
        st.dataframe(raw_df, width="stretch", height=320)

    with st.expander("Show hour ranking table"):
        st.dataframe(hour_tbl, width="stretch")

    with st.expander("Show candidate plan table"):
        st.dataframe(disp_df.sort_values("candidate_name").reset_index(drop=True), width="stretch")
