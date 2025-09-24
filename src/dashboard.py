import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ---------------------------
# Paths
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUT_ROOT = DATA_DIR / "processed"
EDGES_DIR  = OUT_ROOT / "edges"
EVENTS_DIR = OUT_ROOT / "events"
ELEV_DIR   = OUT_ROOT / "elevation"
SIM_DIR    = OUT_ROOT / "sim"
MANDATORY_DIR = SIM_DIR / "mandatory"
SIM_SUMMARY = SIM_DIR / "sim_summary_final.csv"


# ---------------------------
# Robust boolean coercion
# ---------------------------
TRUE_SET  = {"1","true","t","yes","y"}
FALSE_SET = {"0","false","f","no","n",""}

def as_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=bool)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0).astype(float) != 0.0)
    ss = s.astype(str).str.strip().str.lower()
    return ss.map(lambda v: (v in TRUE_SET)).fillna(False)

# ---------------------------
# Elevation-based spans
# ---------------------------
def spans_from_elev_mask(elev_df: pd.DataFrame, mask_col: str = "bridge_mask"):
    if mask_col not in elev_df.columns:
        return []
    m = as_bool_series(elev_df[mask_col]).to_numpy()
    if not m.any():
        return []
    edges = np.where(np.diff(np.r_[False, m, False]) != 0)[0]
    spans = [(int(edges[i]), int(edges[i+1])) for i in range(0, len(edges), 2)]
    return spans

# ---------------------------
# Edges → step series
# ---------------------------
def ensure_step_distances(edges_df: pd.DataFrame, dist_along: np.ndarray) -> pd.DataFrame:
    df = edges_df.copy()
    have_x = {"x0_m", "x1_m"}.issubset(df.columns)
    have_idx = {"begin_shape_index", "end_shape_index"}.issubset(df.columns)
    if not have_x and not have_idx:
        raise ValueError("edges parquet must have either x0_m/x1_m or begin/end indices.")
    if not have_x and have_idx:
        i0 = np.clip(df["begin_shape_index"].astype(int).to_numpy(), 0, len(dist_along)-1)
        i1 = np.clip(df["end_shape_index"].astype(int).to_numpy(), 0, len(dist_along)-1)
        df["x0_m"] = dist_along[i0]
        df["x1_m"] = dist_along[i1]
    return df

def to_steps(df: pd.DataFrame, ycol: str):
    d = df.sort_values("x0_m").reset_index(drop=True)
    xs = np.r_[d["x0_m"].to_numpy(), d["x1_m"].to_numpy()[-1]]
    ys = d[ycol].to_numpy()
    return xs, ys


def snap_events_to_coords(events_df: pd.DataFrame, elev_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    # Use nearest match on dist_m
    snapped = []
    for _, ev in events_df.iterrows():
        d = ev["dist_m"]
        i = (elev_df["dist_m"] - d).abs().idxmin()
        snapped.append({
            "event": ev["event"],
            "name": ev.get("name", ""),
            "will_stop": ev["will_stop"],
            "dist_m": d,
            "lat": elev_df.loc[i, "lat"],
            "lon": elev_df.loc[i, "lon"]
        })
    return pd.DataFrame(snapped)
# ---------------------------
# Cached I/O
# ---------------------------
@st.cache_data(show_spinner=False)
def load_summary():
    return pd.read_csv(SIM_SUMMARY)

@st.cache_data(show_spinner=False)
def load_edges(shape_id: int):
    p = EDGES_DIR / f"edges_{shape_id}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_elev(shape_id: int):
    return pd.read_parquet(ELEV_DIR / f"elev_{shape_id}.parquet")

@st.cache_data(show_spinner=False)
def load_mandatory(shape_id: int):
    p = MANDATORY_DIR / f"mandatory_{shape_id}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(layout="wide", page_title="Transit Energy Model Dashboard")
st.title("Transit Energy Model Dashboard -- Route Level")

summary_df = load_summary()
if summary_df.empty:
    st.error("No simulation summary found.")
    st.stop()

with st.sidebar:
    depot = st.selectbox("Depot", options=sorted(summary_df["Depot Code"].dropna().unique()))
    route = st.selectbox("Route", options=sorted(summary_df.loc[summary_df["Depot Code"]==depot,"route_short_name"].dropna().unique()))
    direction = st.selectbox("Direction", options=sorted(summary_df.loc[summary_df["route_short_name"]==route,"route_direction"].dropna().unique()))
    duty = st.selectbox("Duty mode", options=["heavy","medium"])
    sid = st.selectbox("Shape ID", options=sorted(summary_df.loc[
        (summary_df["route_short_name"]==route) &
        (summary_df["route_direction"]==direction),
        "shape_id"].unique()))
    show_stops = st.checkbox("Show stop events", value=True)

meta_df = summary_df[summary_df["shape_id"] == sid]
if meta_df.empty:
    st.error(f"No summary entry found for shape_id {sid}.")
    st.stop()
else:
    meta = meta_df.iloc[0]

meta = summary_df[summary_df["shape_id"]==sid].iloc[0]

# Load artifacts
elev_df   = load_elev(sid)
edges_df  = load_edges(sid)
mandatory_df = load_mandatory(sid) if show_stops else pd.DataFrame()

coords_latlon = list(zip(elev_df["lat"].astype(float), elev_df["lon"].astype(float)))
dist_along = elev_df["dist_m"].astype(float).to_numpy()
center = coords_latlon[0] if coords_latlon else (0, 0)

bridge_spans_idx = spans_from_elev_mask(elev_df, "bridge_mask")

# ---------------------------
# MAP (Folium)
# ---------------------------
st.subheader("Map")
m = folium.Map(location=center, zoom_start=13, control_scale=True)
folium.PolyLine(coords_latlon, color="blue", weight=3, opacity=0.9, tooltip="Matched route").add_to(m)

if bridge_spans_idx:
    bridges_fg = folium.FeatureGroup(name="Bridges (elevation)", show=True)
    for s, e in bridge_spans_idx:
        seg = coords_latlon[s:max(s+1, e)]
        if len(seg) >= 2:
            folium.PolyLine(seg, color="red", weight=7, opacity=0.95, tooltip="Bridge").add_to(bridges_fg)
    bridges_fg.add_to(m)


# total = 0
if show_stops and not mandatory_df.empty:
    mandatory_snapped = snap_events_to_coords(mandatory_df, elev_df)
    stops_fg = folium.FeatureGroup(name="Bus Stops (mandatory events)", show=True)
    for _, r in mandatory_snapped.iterrows():
        ev = str(r.get("event", "")).lower()
        if duty == "medium":
            if "will_stop" in r and not r["will_stop"]:
                continue  
        if ev in ("start", "gtfs_stop", "end"):
            color = "blue"
        elif ev == "traffic_signal":
            color = "green"
        elif ev == "stop_sign":
            color = "orange"
        else:
            color = "gray"

        folium.CircleMarker(
            (float(r["lat"]), float(r["lon"])),
            radius=3, color=color, fill=True,
            popup=f"{ev} ({duty})"
        ).add_to(stops_fg)
        # total+=1

    stops_fg.add_to(m)

# print(f"total:{total}")

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, height=520, width=None)

# ---------------------------
# KPIs
# ---------------------------
st.markdown(
    """
    <style>
    /* Target metric label text */
    div[data-testid="stMetric"] label {
        font-size: 20px !important;
        font-weight: 700 !important;  /* bold */
    }

    /* Optional: make the metric value (the number) bold/larger too */
    div[data-testid="stMetric"] > div:nth-child(1) {
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.subheader("KPIs")
colA, colB, colC, colD = st.columns(4)
with colB:
    if duty=="heavy":
        st.metric("Energy Consumption (kWh)", f"{meta['pack_used_kwh_heavy_duty']:.2f}")
    else:
        st.metric("Energy Consumption (kWh)", f"{meta['pack_used_kwh_medium_duty']:.2f}")
with colC:
    if duty=="heavy":
        st.metric("kWh per km", f"{meta['kwh_per_km_heavy_duty']:.2f}")
    else:
        st.metric("kWh per km", f"{meta['kwh_per_km_medium_duty']:.2f}")
with colD:
    st.metric("Original Bus Type", str(meta.get("Asset Class","N/A")))
with colA:
    st.metric("Travel Distance (km)", f"{meta['distance_km']:.2f}")



# ---------------------------
# CHARTS (Plotly)
# ---------------------------
col1, col2 = st.columns(2)

with col2:
    st.subheader("Elevation vs Distance")
    y_elev = elev_df["elev_m_smooth"] if "elev_m_smooth" in elev_df.columns else elev_df["elev_m"]
    x_km = elev_df["dist_m"].to_numpy() / 1000.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_km, y=y_elev, mode="lines", name="Elevation"))
    for s, e in bridge_spans_idx:
        a_m = float(elev_df["dist_m"].iloc[s]); b_m = float(elev_df["dist_m"].iloc[max(s, e-1)])
        fig.add_vrect(x0=a_m/1000.0, x1=b_m/1000.0, fillcolor="purple", opacity=0.15, line_width=0)
    fig.update_layout(xaxis_title="Distance (km)", yaxis_title="Elevation (m)", margin=dict(l=30, r=10, t=30, b=30), height=320)
    st.plotly_chart(fig, use_container_width=True)

with col1:
    st.subheader("Speed Limit vs Distance")
    if edges_df.empty:
        st.info("No edges file for this shape; skipping speed plot.")
    else:
        try:
            edges_for_steps = ensure_step_distances(edges_df, dist_along)
            ycol = "speed_limit_kph_smooth"
            xs_m, ys = to_steps(edges_for_steps, ycol)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=xs_m/1000.0, y=np.r_[ys, ys[-1]], mode="lines", line=dict(shape="hv"), name="Speed limit (kph)"))
            fig2.update_layout(xaxis_title="Distance (km)", yaxis_title="Speed limit (kph)", margin=dict(l=30, r=10, t=30, b=30), height=320)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as ex:
            st.warning(f"Speed plot skipped: {ex}")

# ---------------------------
# Time Series (Speed, Grade, SOC)
# ---------------------------
mode_dir = SIM_DIR / ("heavy" if duty=="heavy" else "medium")
ts_path = mode_dir / f"sim_{'heavy' if duty=='heavy' else 'medium'}_{sid}.parquet"

if ts_path.exists():
    ts_df = pd.read_parquet(ts_path)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Speed vs Time")
        if "speed_meters_per_second" in ts_df.columns and "time_seconds" in ts_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_df["time_seconds"], y=ts_df["speed_meters_per_second"]*3.6, mode="lines", name="Speed (km/h)"))
            fig.update_layout(xaxis_title="Time (s)", yaxis_title="Speed (km/h)", margin=dict(l=30, r=10, t=30, b=30), height=320)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Grade vs Time")
        if "grade" in ts_df.columns and "time_seconds" in ts_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_df["time_seconds"], y=ts_df["grade"], mode="lines", name="Grade"))
            fig.update_layout(xaxis_title="Time (s)", yaxis_title="Grade (fraction)", margin=dict(l=30, r=10, t=30, b=30), height=320)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("SOC vs Time")
    if "soc" in ts_df.columns and "time_seconds" in ts_df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_df["time_seconds"], y=ts_df["soc"], mode="lines", name="SOC"))
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="SOC (%)", margin=dict(l=30, r=10, t=30, b=30), height=320)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"No timeseries parquet found for shape {sid} ({duty}).")
