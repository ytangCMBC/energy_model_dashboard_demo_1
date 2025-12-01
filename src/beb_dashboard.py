
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import base64
import altair as alt
import plotly.graph_objects as go


# ---------------------- Paths (match run_pipeline.py) ----------------------
OUT_ROOT = Path("../data/processed")

# Elevation
ELEV_DIR_SERVICE   = OUT_ROOT / "elevation"
ELEV_DIR_DEADHEAD  = OUT_ROOT / "deadhead" / "elevation"

# Simulation timeseries (per mode)
SIM_SERVICE = {
    "heavy": OUT_ROOT / "sim" / "heavy",
    "medium": OUT_ROOT / "sim" / "medium",
}
SIM_DEADHEAD = {
    "heavy": OUT_ROOT / "deadhead" / "sim" / "heavy",
    "medium": OUT_ROOT / "deadhead" / "sim" / "medium",
}

GTFS_DIR = Path("../data/processed/gtfs_bus_only")
TRIPS_CSV = GTFS_DIR / "trips.txt"
ROUTES_CSV = GTFS_DIR / "routes.txt"
STOP_TIMES_CSV = GTFS_DIR / "stop_times.txt"
STOPS_CSV = GTFS_DIR / "stops.txt"

BLOCK_INV_CSV = Path("block_trip_inventory_2.csv")

# ---------------------- Cache helpers ----------------------
@st.cache_data(show_spinner=False)
def _load_gtfs_meta():
    trips = pd.read_csv(TRIPS_CSV) if TRIPS_CSV.exists() else None
    routes = pd.read_csv(ROUTES_CSV) if ROUTES_CSV.exists() else None
    stop_times = pd.read_csv(STOP_TIMES_CSV) if STOP_TIMES_CSV.exists() else None
    return trips, routes, stop_times

@st.cache_data(show_spinner=False)
def _load_stops():
    if not STOPS_CSV.exists():
        return None
    return pd.read_csv(STOPS_CSV)


@st.cache_data(show_spinner=False)
def _load_block_inventory() -> pd.DataFrame:
    if not BLOCK_INV_CSV.exists():
        st.error(f"Block inventory not found: {BLOCK_INV_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(BLOCK_INV_CSV)
    return df


@st.cache_data(show_spinner=False)
def _read_elev_cached(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    lat_col  = "lat"
    lon_col  = "lon"
    dist_col = "dist_m"
    elev_col = "elev_m_smooth"
    if lat_col is None or lon_col is None:
        raise ValueError(f"Missing lat/lon columns in {path}")
    cols = [lat_col, lon_col] + ([dist_col] if dist_col else []) + ([elev_col] if elev_col else [])
    out = df[cols].copy()
    new_cols = ["lat","lon"] + (["dist_m"] if dist_col else []) + (["elev_m_smooth"] if elev_col else [])
    out.columns = new_cols
    if "dist_m" in out.columns:
        out = out.sort_values("dist_m").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
        d = [0.0]
        for i in range(1, len(out)):
            d.append(_hav(out.loc[i-1,"lat"], out.loc[i-1,"lon"], out.loc[i,"lat"], out.loc[i,"lon"]))
        out["dist_m"] = np.cumsum(d)
    return out


def _build_elev_full_for_trip(shape_id: str, trip_type: str) -> pd.DataFrame:
    """
    Return a stable, full elevation dataframe for a trip with cols:
    ['trip_cumul_m', 'elev_m_smooth', 'dist_km'] (NaNs dropped).
    """
    path = _elev_path(shape_id, trip_type)
    if not path.exists():
        return pd.DataFrame(columns=["trip_cumul_m","elev_m_smooth","dist_km"])

    df = pd.read_parquet(path)

    # ensure lat/lon
    if not {"lat","lon"}.issubset(df.columns):
        return pd.DataFrame(columns=["trip_cumul_m","elev_m_smooth","dist_km"])

    # ensure dist_m
    if "dist_m" not in df.columns:
        d = [0.0]
        for i in range(1, len(df)):
            d.append(_hav(df.loc[i-1,"lat"], df.loc[i-1,"lon"], df.loc[i,"lat"], df.loc[i,"lon"]))
        df["dist_m"] = np.cumsum(d)

    # require smoothed elevation
    if "elev_m_smooth" not in df.columns:
        return pd.DataFrame(columns=["trip_cumul_m","elev_m_smooth","dist_km"])

    out = df[["dist_m","elev_m_smooth"]].copy()
    out["trip_cumul_m"] = out["dist_m"] - float(out["dist_m"].iloc[0])
    out = out.loc[out["elev_m_smooth"].notna(), ["trip_cumul_m","elev_m_smooth"]]
    out = out.sort_values("trip_cumul_m").reset_index(drop=True)
    if len(out) >= 1:
        out["dist_km"] = out["trip_cumul_m"] / 1000.0
    else:
        out["dist_km"] = []
    return out




# ---------------------- Geometry ----------------------
RAD = 6371000.0

def _hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return float(2*RAD*np.arctan2(np.sqrt(a), np.sqrt(1-a)))

# ---------------------- Build geometry ----------------------

@st.cache_data(show_spinner=False)
def _build_master(sequence: List[Dict], trips, routes, stop_times, max_pts: int = 1800) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows, trips_rows = [], []
    cumul_m = 0.0
    for i, item in enumerate(sequence, start=1):
        ttype = str(item.get("type","in_service")).lower()
        shape_id = str(item["shape_id"]) 
        trip_id = str(item.get("trip_id", shape_id))
        path = _elev_path(shape_id, ttype)
        if not path.exists():
            trips_rows.append({
                "trip_idx": i, "trip_id": trip_id if ttype=="in_service" else shape_id,
                "type": ttype, "shape_id": shape_id,
                "route_short_name": None if ttype!="in_service" else None,
                "start_time": None if ttype!="in_service" else item.get("start_time"),
                "end_time": None, "trip_distance_km": 0.0,
                "cumul_distance_km": cumul_m/1000.0,
                "start_cumul_m": cumul_m, "end_cumul_m": cumul_m
            })
            continue
        df = _read_elev_cached(str(path))
        df["trip_cumul_m"] = df["dist_m"] - float(df["dist_m"].iloc[0])
        # light subsampling to cap points per trip (simple stride)
        n = len(df)
        stride = 1 if n <= max_pts else max(1, int(np.ceil(n/max_pts)))
        df_s = df.iloc[::stride].copy()
        trip_len_m = float(df_s["trip_cumul_m"].iloc[-1])
        if ttype == "in_service":
            rshort, stime, etime = _trip_meta_from_gtfs(trip_id, trips, routes, stop_times)
        else:
            rshort, stime, etime = (None, None, None)

        for _, r in df_s.iterrows():
            row = {
                "trip_idx": i,
                "trip_id": trip_id if ttype=="in_service" else shape_id,
                "type": ttype,
                "shape_id": shape_id,
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "trip_cumul_m": float(r["trip_cumul_m"]),
                "cumul_m": cumul_m + float(r["trip_cumul_m"]),
            }
            if "elev_m_smooth" in df_s.columns:
                val = r["elev_m_smooth"]
                row["elev_m_smooth"] = float(val) if pd.notna(val) else np.nan
            all_rows.append(row)

        trips_rows.append({
            "trip_idx": i, "trip_id": trip_id if ttype=="in_service" else shape_id,
            "type": ttype, "shape_id": shape_id,
            "route_short_name": rshort, "start_time": item.get("start_time") if (stime is None) else stime,
            "end_time": etime, "trip_distance_km": trip_len_m/1000.0,
            "cumul_distance_km": (cumul_m+trip_len_m)/1000.0,
            "start_cumul_m": cumul_m, "end_cumul_m": cumul_m + trip_len_m
        })
        cumul_m += trip_len_m
    return pd.DataFrame(all_rows), pd.DataFrame(trips_rows)

# ---------------------- Utility ----------------------

def _elev_path(shape_id: str, trip_type: str) -> Path:
    return (ELEV_DIR_DEADHEAD if trip_type.lower() in {"pull_in","pull_out","interline"} else ELEV_DIR_SERVICE) / f"elev_{shape_id}.parquet"

def _sim_path(shape_id: str, trip_type: str, mode: str) -> Path:
    """
    Simulation parquet for this trip & mode (heavy/medium).

    Service:   ../data/processed/sim/{mode}/sim_{mode}_{shape_id}.parquet
    Deadhead:  ../data/processed/deadhead/sim/{mode}/sim_{mode}_{shape_id}.parquet
    """
    is_deadhead = trip_type.lower() in {"pull_in", "pull_out", "interline"}
    base = SIM_DEADHEAD[mode] if is_deadhead else SIM_SERVICE[mode]
    return base / f"sim_{mode}_{shape_id}.parquet"


@st.cache_data(show_spinner=False)
def _load_sim_for_trip(shape_id: str, trip_type: str, mode: str) -> pd.DataFrame:
    """
    Load sim timeseries for a trip and compute a raw distance column.
    Returns DataFrame including:
      - time_seconds
      - speed_meters_per_second
      - pwr_out_electrical_watts
      - soc
      - dist_m_raw  (cumulative distance from speed & time)
    """
    path = _sim_path(shape_id, trip_type, mode)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)

    if "time_seconds" not in df.columns or "speed_meters_per_second" not in df.columns:
        return pd.DataFrame()

    df = df.sort_values("time_seconds").reset_index(drop=True)
    dt = df["time_seconds"].diff().fillna(0.0)
    df["dist_m_raw"] = (df["speed_meters_per_second"].clip(lower=0.0) * dt).cumsum()
    return df


def _trip_meta_from_gtfs(trip_id: str, trips, routes, stop_times):
    if trips is None or routes is None or stop_times is None:
        return None, None, None
    trow = trips[trips["trip_id"].astype(str) == str(trip_id)]
    if trow.empty:
        return None, None, None
    route_id = trow["route_id"].iloc[0]
    rrow = routes[routes["route_id"] == route_id]
    rshort = rrow["route_short_name"].iloc[0] if not rrow.empty else None
    st_trip = stop_times[stop_times["trip_id"].astype(str) == str(trip_id)]
    if st_trip.empty:
        return rshort, None, None
    stime = st_trip["departure_time"].dropna().astype(str).sort_values().head(1).iloc[0] if "departure_time" in st_trip.columns else None
    etime = st_trip["arrival_time"].dropna().astype(str).sort_values().tail(1).iloc[0] if "arrival_time" in st_trip.columns else None
    return rshort, stime, etime

def _lookup_stop_name(code_or_id: str, stops: pd.DataFrame | None) -> str:
    """
    Try to resolve a stop code or ID to a stop_name using GTFS stops.txt.
    If no match is found, return the original string (for depot codes like HTC/BTC).
    """
    s = str(code_or_id)
    if "stop_id" in stops.columns:
        m = stops[stops["stop_id"].astype(str) == s]
        if not m.empty:
            return str(m["stop_name"].iloc[0])

    if "stop_code" in stops.columns:
        m = stops[stops["stop_code"].astype(int).astype(str) == s]
        if not m.empty:
            return str(m["stop_name"].iloc[0])

    return s


def _trip_start_end_names(cur_row: pd.Series,
                          stop_times: pd.DataFrame | None,
                          stops: pd.DataFrame | None) -> tuple[str, str]:
    """
    For in-service trips:
      - Use stop_times to get first/last stop, then map through stops.txt.

    For not-in-service (pull_in / pull_out / interline):
      - Parse shape_id like 'BTC_58249' or '58249_50123'.
      - If a part is numeric → treat as stop_code and look up GTFS stop_name.
      - If non-numeric → treat as depot code and use raw string.
    """
    ttype = str(cur_row["type"]).lower()

    # ---------- In-service ----------
    if ttype == "in_service" and stop_times is not None:
        st_trip = stop_times[stop_times["trip_id"].astype(str) == str(cur_row["trip_id"])]
        if not st_trip.empty:
            if "stop_sequence" in st_trip.columns:
                st_trip = st_trip.sort_values("stop_sequence")
            row0 = st_trip.iloc[0]
            rowN = st_trip.iloc[-1]

            start_code = row0.get("stop_id", row0.get("stop_code", None))
            end_code   = rowN.get("stop_id", rowN.get("stop_code", None))
            start_name = _lookup_stop_name(start_code, stops) if start_code is not None else "—"
            end_name   = _lookup_stop_name(end_code, stops)   if end_code is not None else "—"

            return start_name, end_name

    # ---------- Not in service ----------
    shape = str(cur_row["shape_id"])
    parts = shape.split("_")

    if len(parts) == 2:
        raw_start, raw_end = parts[0].strip(), parts[1].strip()

        # If a part is numeric → lookup in stops.txt
        if raw_start.isdigit():  
            start_name = _lookup_stop_name(raw_start, stops)
        else:
            start_name = raw_start  # depot / non-numeric code

        if raw_end.isdigit():
            end_name = _lookup_stop_name(raw_end, stops)
        else:
            end_name = raw_end
    else:
        # fallback: show shape_id
        start_name = shape
        end_name = shape

    return start_name, end_name


# put near the top
EPS = 1e-6

def _locate_trip_row(cur_m_val: float, trips: pd.DataFrame) -> pd.Series:
    t = trips.sort_values("start_cumul_m").reset_index(drop=True)

    # Before the first trip starts → pick first
    if cur_m_val <= float(t.loc[0, "start_cumul_m"]) + EPS:
        return t.loc[0]

    # Middle trips: [start, end) and boundary goes to NEXT trip
    for i in range(len(t) - 1):
        s = float(t.loc[i,   "start_cumul_m"])
        e = float(t.loc[i,   "end_cumul_m"])
        s_next = float(t.loc[i+1, "start_cumul_m"])
        # inside
        if (cur_m_val >= s - EPS) and (cur_m_val < e - EPS):
            return t.loc[i]
        # exactly on boundary → choose next trip
        if abs(cur_m_val - e) < EPS or abs(cur_m_val - s_next) < EPS:
            return t.loc[i+1]

    # Last trip: [start, end] inclusive
    return t.loc[len(t) - 1]

# ---------------- Energy mode selector ----------------
energy_mode = st.radio(
    "Select energy model",
    ["Heavy-duty", "Medium-duty"],
    index=0,
    horizontal=True,
)

# Internal key used in code
if energy_mode.startswith("Heavy"):
    energy_mode_key = "heavy"
else:
    energy_mode_key = "medium"


# ---------------------- UI ----------------------
st.set_page_config(page_title="Block Level Summary Dashboard", layout="wide")
st.title("Block Level Summary Dashboard")

# Load metadata
block_inv = _load_block_inventory()

with st.sidebar:
    st.markdown("### Filters")

    # 1 Duty / energy mode (controls which success column and which sim files)
    energy_mode = st.radio(
        "Duty / Energy mode",
        ["Heavy-duty", "Medium-duty"],
        index=0,
    )
    energy_mode_key = "heavy" if energy_mode.startswith("Heavy") else "medium"
    success_col = "heavy_success" if energy_mode_key == "heavy" else "medium_success"

    # Guard: empty inventory
    if block_inv.empty:
        st.stop()

    # 2 Depot selection
    depot_options = sorted(block_inv["depot_code"].dropna().unique())
    depot = st.selectbox("Depot", depot_options)

    df_depot = block_inv[block_inv["depot_code"] == depot].copy()

    # 3 Success / failure filter for the chosen mode
    status_label = st.radio(
        "Simulation result",
        ["Success", "Failure"],
        index=0,
        horizontal=True,
    )
    status_value = "SUCCESS" if status_label == "Success" else "FAILURE"

    df_mode = df_depot[df_depot[success_col] == status_value].copy()

    if df_mode.empty:
        st.warning("No blocks match this depot + result combination.")
        st.stop()

    # 4 Block ID selection
    block_ids = sorted(df_mode["block_id"].unique())
    block_id = st.selectbox("Block ID", block_ids)

    # The row that defines this block's combined sequence
    row_block = df_mode[df_mode["block_id"] == block_id].iloc[0]

    # Parse combined_sequence_json for this block
    try:
        sequence = json.loads(row_block["combined_sequence_json"])
        if not isinstance(sequence, list):
            raise ValueError("combined_sequence_json is not a list")
    except Exception as e:
        st.error(f"Unable to parse combined_sequence_json for block {block_id}: {e}")
        st.stop()

    # (Optional) small debug view of the sequence if you want it
    with st.expander("Show raw sequence JSON", expanded=False):
        st.code(json.dumps(sequence, indent=2))



selection_key = f"{energy_mode_key}|{depot}|{status_value}|{block_id}"

def _reset_state_for_new_selection():
    st.session_state.selection_key = selection_key

    # Playback state
    st.session_state.cursor_m = 0.0
    st.session_state.anim_running = False
    st.session_state.last_tick = time.time()
    st.session_state.pause_until = None
    st.session_state.finished = False

    # Trip index
    st.session_state.current_trip_idx = 0

    # Elevation / sim cache (per-trip)
    st.session_state.elev_trip_key = None
    st.session_state.elev_full_df = None
    st.session_state.sim_trip_key = None
    st.session_state.sim_df_trip = None

    # Energy / SOC cache (per block & mode)
    st.session_state.energy_trip_done = 0
    st.session_state.energy_offset = 0.0
    st.session_state.energy_cumul_kwh = 0.0
    st.session_state.SOC_OFFSETS = {}
    st.session_state.SOC_END_GLOBAL = {}
    st.session_state.TRIP_KWH = {}
    st.session_state.CUMUL_KWH = {}
    st.session_state.TRIP_KWH_PER_KM = {}



if (
    "selection_key" not in st.session_state
    or st.session_state.selection_key != selection_key
):
    _reset_state_for_new_selection()


# Build data
trips, routes, stop_times = _load_gtfs_meta()
stops = _load_stops()
points_df, trips_df = _build_master(sequence, trips, routes, stop_times)
if points_df.empty:
    st.warning("No geometry loaded. Check elevation parquet paths and shape_ids.")
    st.stop()

# ---- Battery / SOC model (global cumulative SOC view) ----
BATTERY_USABLE_KWH = 376.0      # usable pack energy
START_SOC = 0.90                # 90% start
INIT_KWH = BATTERY_USABLE_KWH * START_SOC 


def _ensure_energy_upto(
    target_trip_idx: int,
    trips_df: pd.DataFrame,
    points_df: pd.DataFrame,
    mode: str,
) -> None:
    """
    Lazily compute energy up to target_trip_idx and derive GLOBAL SOC
    from a simple battery model:

        SOC_global = (INIT_KWH - cumulative_energy_used) / BATTERY_USABLE_KWH

    Stores results in st.session_state:
      - TRIP_KWH[trip_idx]
      - CUMUL_KWH[trip_idx]
      - TRIP_KWH_PER_KM[trip_idx]
      - SOC_END_GLOBAL[trip_idx]   (fraction 0–1 at END of trip)
    """
    # Initialise if missing (defensive)
    if "energy_trip_done" not in st.session_state:
        st.session_state.energy_trip_done = 0
        st.session_state.energy_cumul_kwh = 0.0
        st.session_state.SOC_END_GLOBAL = {}
        st.session_state.TRIP_KWH = {}
        st.session_state.CUMUL_KWH = {}
        st.session_state.TRIP_KWH_PER_KM = {}

    done = int(st.session_state.energy_trip_done)
    if done >= target_trip_idx:
        return

    trips_sorted = trips_df.sort_values("trip_idx")

    for _, row in trips_sorted.iterrows():
        trip_idx = int(row["trip_idx"])
        if trip_idx <= done or trip_idx > target_trip_idx:
            continue

        shape_id = str(row["shape_id"])
        ttype = str(row["type"]).lower()

        # Sim timeseries for this trip
        sim_df = _load_sim_for_trip(shape_id, ttype, mode)

        # Trip distance from geometry (for kWh/km)
        pts = points_df[points_df["trip_idx"] == trip_idx]
        if not pts.empty:
            d_trip_m = float(pts["cumul_m"].iloc[-1] - pts["cumul_m"].iloc[0])
            d_trip_km = d_trip_m / 1000.0
        else:
            d_trip_km = float("nan")

        this_kwh = float("nan")

        if not sim_df.empty:
            # Integrate electrical power to get trip energy in kWh
            if (
                "pwr_out_electrical_watts" in sim_df.columns
                and "time_seconds" in sim_df.columns
            ):
                dt = sim_df["time_seconds"].diff().fillna(0.0)
                p = sim_df["pwr_out_electrical_watts"].fillna(0.0)
                this_kwh = float((p * dt).sum() / 3_600_000.0)  # J → kWh

        # Update cumulative kWh across this block
        if np.isfinite(this_kwh):
            st.session_state.energy_cumul_kwh += this_kwh

        st.session_state.TRIP_KWH[trip_idx] = this_kwh
        st.session_state.CUMUL_KWH[trip_idx] = st.session_state.energy_cumul_kwh

        # Battery-based global SOC at END of this trip (fraction 0–1)
        if np.isfinite(st.session_state.energy_cumul_kwh):
            soc_end_g = (INIT_KWH - st.session_state.energy_cumul_kwh) / BATTERY_USABLE_KWH
        else:
            soc_end_g = float("nan")

        st.session_state.SOC_END_GLOBAL[trip_idx] = soc_end_g

        # Trip kWh/km
        if d_trip_km > 0 and np.isfinite(this_kwh):
            st.session_state.TRIP_KWH_PER_KM[trip_idx] = this_kwh / d_trip_km
        else:
            st.session_state.TRIP_KWH_PER_KM[trip_idx] = float("nan")

        st.session_state.energy_trip_done = trip_idx




# State
if "cursor_m" not in st.session_state:
    st.session_state.cursor_m = 0.0

# ---------------- Cursor & segmentation (500 m segments per trip, clamped at trip ends) ----------------
SEG_LEN_M = 500.0
trips_sorted = trips_df.sort_values("start_cumul_m").reset_index(drop=True)
segment_points: list[float] = []

for _, trow in trips_sorted.iterrows():
    s = float(trow["start_cumul_m"])
    e = float(trow["end_cumul_m"])

    # Skip degenerate trips
    if e <= s + 1e-6:
        continue

    pos = s
    while pos < e - 1e-6:
        segment_points.append(pos)
        pos += SEG_LEN_M
    segment_points.append(e)


segment_points = sorted(set(segment_points))

if not segment_points:
    segment_points = [0.0]

prev_cursor = st.session_state.get("cursor_m", segment_points[0])
default_cursor = min(segment_points, key=lambda v: abs(v - prev_cursor))

st.markdown("### Cursor along block (500 m segments)")

cursor_val = st.select_slider(
    "Position along block",
    options=segment_points,
    value=default_cursor,
    format_func=lambda v: f"{v/1000.0:.2f} km",
    key="cursor_m",
)

follow_bus = st.checkbox("Follow bus (pan camera)", value=True)


cur_m = st.session_state.cursor_m
cur_row = _locate_trip_row(cur_m, trips_df)
cur_idx = int(cur_row["trip_idx"])
start_stop_name, end_stop_name = _trip_start_end_names(cur_row, stop_times, stops)

trip_start = float(cur_row["start_cumul_m"]) 
trip_end   = float(cur_row["end_cumul_m"]) 

# Static elevation line: (re)build only when trip changes
this_trip_key = f"{cur_row['type']}|{cur_row['shape_id']}"
if st.session_state.elev_trip_key != this_trip_key:
    elev_full = _build_elev_full_for_trip(str(cur_row["shape_id"]), str(cur_row["type"]).lower())
    st.session_state.elev_trip_key = this_trip_key
    st.session_state.elev_full_df = elev_full if len(elev_full) >= 2 else None

    # Prebuild the line chart once per trip (static)
    if st.session_state.elev_full_df is not None:
        st.session_state.elev_line_chart = (
            alt.Chart(st.session_state.elev_full_df)
            .mark_line()
            .encode(
                x=alt.X("dist_km:Q", title="Distance (km)"),
                y=alt.Y("elev_m_smooth:Q", title="Elevation (m)")
            )
            .properties(height=180, width="container")
        )
    else:
        st.session_state.elev_line_chart = None

# Static simulation arrays: power & SOC vs distance (aligned to elevation)
sim_trip_key = f"{energy_mode_key}|{cur_row['type']}|{cur_row['shape_id']}"
if st.session_state.sim_trip_key != sim_trip_key:
    trip_idx = int(cur_row["trip_idx"])
    _ensure_energy_upto(trip_idx, trips_df, points_df, energy_mode_key)

    sim_df = _load_sim_for_trip(
        str(cur_row["shape_id"]),
        str(cur_row["type"]).lower(),
        energy_mode_key,
    )
    elev_full_df = st.session_state.elev_full_df

    if (not sim_df.empty) and (elev_full_df is not None) and (len(elev_full_df) >= 2):
        # Align distance to elevation
        elev_len = float(elev_full_df["trip_cumul_m"].iloc[-1])
        max_raw = float(sim_df["dist_m_raw"].iloc[-1]) if sim_df["dist_m_raw"].iloc[-1] > 0 else 0.0
        scale = (elev_len / max_raw) if (max_raw > 0.0 and elev_len > 0.0) else 1.0

        sim_df["dist_m"] = sim_df["dist_m_raw"] * scale
        sim_df["dist_km"] = sim_df["dist_m"] / 1000.0

        # Electrical power in kW
        if "pwr_out_electrical_watts" in sim_df.columns:
            sim_df["pwr_out_elec_kw"] = sim_df["pwr_out_electrical_watts"] / 1000.0

        # Cumulative electrical energy in kWh (for energy vs distance plot)
        if (
            "pwr_out_electrical_watts" in sim_df.columns
            and "time_seconds" in sim_df.columns
        ):
            dt = sim_df["time_seconds"].diff().fillna(0.0)
            p = sim_df["pwr_out_electrical_watts"].fillna(0.0)
            energy_j = (p * dt).cumsum()
            sim_df["cum_energy_kwh"] = energy_j / 3_600_000.0  # J → kWh

            prev_cumul_kwh = st.session_state.CUMUL_KWH.get(trip_idx - 1, 0.0)
            sim_df["soc_global"] = (INIT_KWH - (prev_cumul_kwh + sim_df["cum_energy_kwh"])) / BATTERY_USABLE_KWH

        st.session_state.sim_df_trip = sim_df
    else:
        st.session_state.sim_df_trip = None

    st.session_state.sim_trip_key = sim_trip_key




# --- Recompute current trip for drawing ---

seg = points_df[points_df["trip_idx"] == cur_idx].copy()
seg["cumul_m"] = trip_start + seg["trip_cumul_m"]

cursor_eff = min(st.session_state.cursor_m, trip_end)
s = seg["cumul_m"].to_numpy()
k = int(np.searchsorted(s, cursor_eff + 1e-6, side="right"))
past = seg.iloc[:k].copy()

if len(past) < 2:
    past = seg.head(2)

# Markers
seg_start = seg.head(1).iloc[0]
seg_endpt = seg.tail(1).iloc[0]
start_marker = {"lon": float(seg_start["lon"]), "lat": float(seg_start["lat"]) }
end_marker   = {"lon": float(seg_endpt["lon"]),   "lat": float(seg_endpt["lat"]) }

# Bus dot at tail of past
last = past.tail(1).iloc[0]
lat, lon = float(last["lat"]), float(last["lon"]) 

# Map
path_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": past[["lon","lat"]].values.tolist()}],
    get_path="path",
    get_width=4,
    width_min_pixels=2,
    opacity=0.95,
)
start_layer = pdk.Layer(
    "ScatterplotLayer",
    data=[start_marker],
    get_position="[lon, lat]",
    get_radius=24,
    radius_min_pixels=6,
    get_fill_color=[0, 170, 60],   
    get_line_color=[0, 120, 45],   
    pickable=False,
)

end_layer = pdk.Layer(
    "ScatterplotLayer",
    data=[end_marker],
    get_position="[lon, lat]",
    get_radius=24,
    radius_min_pixels=6,
    get_fill_color=[0, 170, 60],   
    get_line_color=[0, 120, 45],  
    pickable=False,
)


# --- Bus icon (inline base64; row carries full icon dict) ---
@st.cache_data(show_spinner=False)
def _bus_icon():
    icon_path = Path(__file__).parent / "bus_icon.png"
    icon_data_uri = "data:image/png;base64," + base64.b64encode(icon_path.read_bytes()).decode("utf-8")
    return {
        "url": icon_data_uri,
        "width": 512,
        "height": 512,
        "anchorY": 512,
    }
bus_icon = _bus_icon()
bus_layer = pdk.Layer(
    "IconLayer",
    data=pd.DataFrame([{"lon": lon, "lat": lat, "icon": bus_icon}]),
    get_icon="icon",            
    get_position='[lon, lat]',
    get_size=4,                   
    size_scale=6,                 
    pickable=False,
)



if follow_bus:
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=13.2, bearing=0, pitch=0)
else:
    t0 = seg_start
    view_state = pdk.ViewState(latitude=float(t0["lat"]), longitude=float(t0["lon"]), zoom=13.2, bearing=0, pitch=0)

st.pydeck_chart(
    pdk.Deck(layers=[path_layer, start_layer, end_layer, bus_layer], initial_view_state=view_state, map_style=None),
    use_container_width=True, height=560
)


# Trip info
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown("### Trip Information")

# Custom CSS for trip info cards (only needs to be injected once)
st.markdown(
    """
    <style>
    .trip-card {
        background: #f9fafb;
        border-radius: 12px;
        padding: 8px 10px;
        border: 1px solid #e1e5ea;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
        margin-bottom: 6px;
    }
    .trip-card-title {
        font-size: 0.90rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4b5563;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .trip-card-value {
        font-size: 1.0rem;
        font-weight: 500;
        color: #111827;
        word-wrap: break-word;
    }

    /* KPI cards follow the same visual style, slightly more accent on title */
    .kpi-card {
        background: #f9fafb;
        border-radius: 12px;
        padding: 8px 10px;
        border: 1px solid #d1e0ff;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
        margin-bottom: 6px;
    }
    .kpi-card-title {
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4b5563;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .kpi-card-value {
        font-size: 1.0rem;
        font-weight: 500;
        color: #111827;
        word-wrap: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Trip info (styled cards)
c1, c2, c3, c4 = st.columns(4)
c5, c6, c7, c8 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="trip-card">
            <div class="trip-card-title">Trip Index</div>
            <div class="trip-card-value">{int(cur_row["trip_idx"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="trip-card">
            <div class="trip-card-title">Type</div>
            <div class="trip-card-value">{str(cur_row["type"]).replace("_"," ").title()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="trip-card">
            <div class="trip-card-title">Trip ID</div>
            <div class="trip-card-value">{str(cur_row["trip_id"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="trip-card">
            <div class="trip-card-title">Route</div>
            <div class="trip-card-value">{cur_row.get("route_short_name") or "—"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"""
        <div class="trip-card">
            <div class="trip-card-title">Start Time</div>
            <div class="trip-card-value">{cur_row.get("start_time") or "—"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c6:
    st.markdown(
        f"""
        <div class="trip-card">
            <div class="trip-card-title">End Time</div>
            <div class="trip-card-value">{cur_row.get("end_time") or "—"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c7:
    st.markdown(
        f"""
        <div class="trip-card">
            <div class="trip-card-title">Start Stop</div>
            <div class="trip-card-value">{start_stop_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c8:
    st.markdown(
        f"""
        <div class="trip-card">
            <div class="trip-card-title">End Stop</div>
            <div class="trip-card-value">{end_stop_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Trip-level Energy KPIs ----------------
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown(f"### Trip Energy KPIs ({energy_mode} model)")

trip_idx = int(cur_row["trip_idx"])
_ensure_energy_upto(trip_idx, trips_df, points_df, energy_mode_key)

trip_kwh_val = st.session_state.TRIP_KWH.get(trip_idx, float("nan"))
cumul_kwh_val = st.session_state.CUMUL_KWH.get(trip_idx, float("nan"))
trip_kwh_per_km_val = st.session_state.TRIP_KWH_PER_KM.get(trip_idx, float("nan"))
soc_end_global_val = st.session_state.SOC_END_GLOBAL.get(trip_idx, float("nan"))


def _fmt(val, fmt_str, default="N/A"):
    return default if (val is None or not np.isfinite(val)) else fmt_str.format(val)

trip_dist_val = _fmt(float(cur_row["trip_distance_km"]), "{:.3f}")
cumul_dist_val = _fmt(float(cur_row["cumul_distance_km"]), "{:.3f}")
v_trip_kwh = _fmt(trip_kwh_val, "{:,.1f}")
v_cumul_kwh = _fmt(cumul_kwh_val, "{:,.1f}")
v_kwh_per_km = _fmt(trip_kwh_per_km_val, "{:.2f}")
v_soc_cumul = _fmt(soc_end_global_val * 100.0 if np.isfinite(soc_end_global_val) else np.nan, "{:.1f}")

col1, col2, col3, col4 = st.columns(4)
col5, col6, col7, col8 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-card-title">Trip Energy (kWh)</div>
            <div class="kpi-card-value">{v_trip_kwh}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-card-title">Cumulative Energy (kWh)</div>
            <div class="kpi-card-value">{v_cumul_kwh}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-card-title">Trip kWh/km</div>
            <div class="kpi-card-value">{v_kwh_per_km}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-card-title">Cumulative SOC (%)</div>
            <div class="kpi-card-value">{v_soc_cumul}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-card-title">Trip Distance (km)</div>
            <div class="kpi-card-value">{trip_dist_val}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col6:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-card-title">Cumulative Distance (km)</div>
            <div class="kpi-card-value">{cumul_dist_val}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



PLOT_WIDTH = 1600
PLOT_HEIGHT = 500


# ---------------- Elevation chart (static line per trip + bus icon per frame) ----------------
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown("### Elevation vs Distance")

elev_full_df = st.session_state.elev_full_df

if elev_full_df is not None and len(elev_full_df) >= 2:
    cursor_eff = min(float(st.session_state.cursor_m), float(trip_end))
    rel_m = max(0.0, cursor_eff - float(trip_start))

    t_all = elev_full_df["trip_cumul_m"].to_numpy()
    kk = int(np.searchsorted(t_all, rel_m + 1e-6, side="left"))
    kk = max(0, min(kk, len(elev_full_df) - 1))

    marker_row = elev_full_df.iloc[kk:kk+1][["dist_km", "elev_m_smooth"]]
    dot_x = float(marker_row["dist_km"].iloc[0])
    dot_y = float(marker_row["elev_m_smooth"].iloc[0])

    # Build Plotly figure
    fig = go.Figure()

    # Elevation line
    fig.add_trace(
        go.Scatter(
            x=elev_full_df["dist_km"],
            y=elev_full_df["elev_m_smooth"],
            mode="lines",
            line=dict(width=2),
            name="Elevation (m)"
        )
    )

    # Moving red dot
    fig.add_trace(
        go.Scatter(
            x=[dot_x],
            y=[dot_y],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Position"
        )
    )

    # Layout styling
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="Distance (km)",
        yaxis_title="Elevation (m)",
        height=PLOT_HEIGHT,
        width=PLOT_WIDTH,
        plot_bgcolor="white",
        paper_bgcolor="white",
        transition=dict(duration=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="left",
            x=0
        )        
    )


    # Border + subtle grid
    fig.update_xaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True,
        showgrid=True, gridcolor="rgba(0,0,0,0.05)"
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True,
        showgrid=True, gridcolor="rgba(0,0,0,0.05)"
    )

    st.plotly_chart(fig, use_container_width=False)

else:
    st.info("Elevation profile unavailable for this trip.", icon="ℹ️")


# ---------------- Electrical Energy vs Distance ----------------
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown(f"### Energy vs Distance ({energy_mode} model)")

sim_df_trip = st.session_state.sim_df_trip

if sim_df_trip is not None and "cum_energy_kwh" in sim_df_trip.columns and len(sim_df_trip) >= 2:
    cursor_eff = min(float(st.session_state.cursor_m), float(trip_end))
    rel_m = max(0.0, cursor_eff - float(trip_start))

    t_sim = sim_df_trip["dist_m"].to_numpy()
    jj = int(np.searchsorted(t_sim, rel_m + 1e-6, side="left"))
    jj = max(0, min(jj, len(sim_df_trip) - 1))

    dot_x_e = float(sim_df_trip["dist_km"].iloc[jj])
    dot_y_e = float(sim_df_trip["cum_energy_kwh"].iloc[jj])

    fig_e = go.Figure()
    fig_e.add_trace(
        go.Scatter(
            x=sim_df_trip["dist_km"],
            y=sim_df_trip["cum_energy_kwh"],
            mode="lines",
            line=dict(width=2),
            name="Cumulative Energy (kWh)"
        )
    )
    fig_e.add_trace(
        go.Scatter(
            x=[dot_x_e],
            y=[dot_y_e],
            mode="markers",
            marker=dict(size=9, color="red"),
            name="Position"
        )
    )

    fig_e.update_layout(
        margin=dict(l=10, r=10, t=10, b=30),   
        xaxis_title="Distance (km)",
        yaxis_title="Cumulative Energy (kWh)",
        height=PLOT_HEIGHT,
        width=PLOT_WIDTH,
        plot_bgcolor="white",
        paper_bgcolor="white",
        transition=dict(duration=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="left",
            x=0
        )
    )

    fig_e.update_xaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True,
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
    )
    fig_e.update_yaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True,
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
    )

    st.plotly_chart(fig_e, use_container_width=False)
else:
    st.info("Simulation energy profile unavailable for this trip.", icon="ℹ️")


# ---------------- Cumulative SOC vs Distance ----------------
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown(f"### Cumulative SOC vs Distance ({energy_mode} model)")

if sim_df_trip is not None and "soc_global" in sim_df_trip.columns and len(sim_df_trip) >= 2:
    cursor_eff = min(float(st.session_state.cursor_m), float(trip_end))
    rel_m = max(0.0, cursor_eff - float(trip_start))

    t_sim2 = sim_df_trip["dist_m"].to_numpy()
    jj2 = int(np.searchsorted(t_sim2, rel_m + 1e-6, side="left"))
    jj2 = max(0, min(jj2, len(sim_df_trip) - 1))

    dot_x_s = float(sim_df_trip["dist_km"].iloc[jj2])
    dot_y_s = float(sim_df_trip["soc_global"].iloc[jj2] * 100.0)

    fig_s = go.Figure()
    fig_s.add_trace(
        go.Scatter(
            x=sim_df_trip["dist_km"],
            y=sim_df_trip["soc_global"] * 100.0,
            mode="lines",
            line=dict(width=2),
            name="SOC (%)"
        )
    )
    fig_s.add_trace(
        go.Scatter(
            x=[dot_x_s],
            y=[dot_y_s],
            mode="markers",
            marker=dict(size=9, color="red"),
            name="Position"
        )
    )

    fig_s.update_layout(
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="Distance (km)",
        yaxis_title="SOC (%)",
        height=PLOT_HEIGHT,
        width=PLOT_WIDTH,
        plot_bgcolor="white",
        paper_bgcolor="white",
        transition=dict(duration=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="left",
            x=0
        )
    )


    fig_s.update_xaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True,
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
    )
    fig_s.update_yaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True,
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
    )

    st.plotly_chart(fig_s, use_container_width=False)
else:
    st.info("SOC profile unavailable for this trip.", icon="ℹ️")





