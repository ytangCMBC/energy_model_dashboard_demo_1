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

DATA_DIR  = PROJECT_ROOT / "data"
OUT_ROOT  = DATA_DIR / "processed"

# In-service
EDGES_DIR      = OUT_ROOT / "edges"
EVENTS_DIR     = OUT_ROOT / "events"
ELEV_DIR       = OUT_ROOT / "elevation"
SIM_DIR        = OUT_ROOT / "sim"
MANDATORY_DIR  = SIM_DIR / "mandatory"
SIM_SUMMARY    = SIM_DIR / "sim_summary_final.csv"

# Deadhead (Not In Service)
DEADHEAD_ROOT           = OUT_ROOT / "deadhead"
DEADHEAD_EDGES_DIR      = DEADHEAD_ROOT / "edges"
DEADHEAD_EVENTS_DIR     = DEADHEAD_ROOT / "events"
DEADHEAD_ELEV_DIR       = DEADHEAD_ROOT / "elevation"
DEADHEAD_SIM_DIR        = DEADHEAD_ROOT / "sim"
DEADHEAD_MANDATORY_DIR  = DEADHEAD_SIM_DIR / "mandatory"
DEADHEAD_SUMMARY_FINAL  = DEADHEAD_SIM_DIR / "sim_summary_deadhead_final.csv"

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
    snapped = []
    for _, ev in events_df.iterrows():
        d = ev["dist_m"]
        i = (elev_df["dist_m"] - d).abs().idxmin()
        snapped.append({
            "event": ev["event"],
            "name": ev.get("name", ""),
            "will_stop": ev.get("will_stop", True),
            "dist_m": d,
            "lat": elev_df.loc[i, "lat"],
            "lon": elev_df.loc[i, "lon"]
        })
    return pd.DataFrame(snapped)


def ensure_cursor_defaults(elev_df: pd.DataFrame, key: str = "cursor"):
    """Only set defaults if missing. Do NOT overwrite widget keys."""
    dmin = float(elev_df["dist_m"].min()) / 1000.0
    # initialize once, before the slider is created
    if f"{key}_km" not in st.session_state:
        st.session_state[f"{key}_km"] = dmin  # safe: slider not created yet


def compute_cursor_derived(elev_df: pd.DataFrame, key: str = "cursor"):
    """Read km and compute derived values without modifying the widget key."""
    dmin = float(elev_df["dist_m"].min()) / 1000.0
    dmax = float(elev_df["dist_m"].max()) / 1000.0
    km = float(st.session_state.get(f"{key}_km", dmin))
    # clamp locally; do NOT write back to the widget key
    km = max(min(km, dmax), dmin)
    m = km * 1000.0
    idx = int((elev_df["dist_m"] - m).abs().idxmin())
    latlon = (float(elev_df.loc[idx, "lat"]), float(elev_df.loc[idx, "lon"]))
    st.session_state[f"{key}_m"] = m
    st.session_state[f"{key}_idx"] = idx
    st.session_state[f"{key}_latlon"] = latlon


# ---------------------------
# Cached I/O (parameterized by trip type)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_edges(path_dir: Path, shape_id: str):
    p = path_dir / f"edges_{shape_id}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_elev(path_dir: Path, shape_id: str):
    return pd.read_parquet(path_dir / f"elev_{shape_id}.parquet")


@st.cache_data(show_spinner=False)
def load_mandatory(path_dir: Path, shape_id: str):
    p = path_dir / f"mandatory_{shape_id}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# ---------------------------
# Streamlit Panel Renderer
# ---------------------------
def render_trip_panel():
    st.markdown("## Route Level Summary Panel")

    # Trip type selector
    with st.sidebar:
        trip_type = st.radio("Trip type", ["In Service", "Not In Service (Deadhead)"], index=0)

    # Load proper summary + set directories
    if trip_type == "In Service":
        summary_df = load_summary(SIM_SUMMARY)
        edges_dir     = EDGES_DIR
        elev_dir      = ELEV_DIR
        mandatory_dir = MANDATORY_DIR
        sim_dir       = SIM_DIR
    else:
        summary_df = load_summary(DEADHEAD_SUMMARY_FINAL)
        edges_dir     = DEADHEAD_EDGES_DIR
        elev_dir      = DEADHEAD_ELEV_DIR
        mandatory_dir = DEADHEAD_MANDATORY_DIR
        sim_dir       = DEADHEAD_SIM_DIR

    if summary_df.empty:
        st.error("No simulation summary found for the selected trip type.")
        st.stop()

    # ---------------------------
    # Sidebar filters (different by trip type)
    # ---------------------------
    with st.sidebar:
        duty = st.selectbox("Duty mode", options=["heavy", "medium"])

        if trip_type == "In Service":
            depot_opts = sorted(summary_df.get("Depot Code", pd.Series(dtype=str)).dropna().unique())
            depot = st.selectbox("Depot", options=depot_opts) if depot_opts else None

            route_opts = sorted(
                summary_df.loc[summary_df.get("Depot Code","")==depot, "route_short_name"].dropna().unique()
            ) if depot else []
            route = st.selectbox("Route", options=route_opts) if route_opts else None

            dir_mask = (summary_df["route_short_name"] == route) if route else np.zeros(len(summary_df), dtype=bool)
            direction_opts = sorted(
                summary_df.loc[dir_mask, "route_direction"].dropna().unique()
            ) if route else []
            direction = st.selectbox("Direction", options=direction_opts) if direction_opts else None

            sid_opts = summary_df.loc[
                (summary_df.get("Depot Code","")==depot) &
                (summary_df["route_short_name"]==route) &
                (summary_df["route_direction"]==direction),
                "shape_id"
            ].astype(str).unique() if (depot and route and direction) else []
            sid = st.selectbox("Shape ID", options=sorted(sid_opts)) if len(sid_opts) > 0 else None

            show_stops = st.checkbox("Show stop events", value=True)
        else:
            # Deadhead: Depot → Record Type → From Stop → To Stop
            depot_col = "depot" if "depot" in summary_df.columns else (
                "depot_code" if "depot_code" in summary_df.columns else None
            )
            if not depot_col:
                st.error("No depot column found in deadhead summary (expected 'depot' or 'depot_code').")
                st.stop()

            depot_opts = sorted(summary_df.get(depot_col, pd.Series(dtype=str)).dropna().unique())
            depot = st.selectbox("Depot", options=depot_opts) if depot_opts else None

            # Filter by depot first
            df_d = summary_df.loc[summary_df.get(depot_col, "") == depot] if depot else summary_df

            # Record type
            rec_opts = sorted(df_d.get("record_type", pd.Series(dtype=str)).dropna().unique())
            record_type = st.selectbox("Record Type", options=rec_opts) if rec_opts else None

            df_r = df_d.loc[df_d["record_type"] == record_type] if record_type else df_d

            # From stop name
            from_opts = sorted(df_r.get("from_stop_name", pd.Series(dtype=str)).dropna().unique())
            from_name = st.selectbox("From Stop Name", options=from_opts) if from_opts else None

            df_f = df_r.loc[df_r["from_stop_name"] == from_name] if from_name else df_r

            # To stop name
            to_opts = sorted(df_f.get("to_stop_name", pd.Series(dtype=str)).dropna().unique())
            to_name = st.selectbox("To Stop Name", options=to_opts) if to_opts else None

            df_t = df_f.loc[df_f["to_stop_name"] == to_name] if to_name else df_f

            # Pick the first matching trip silently (don't show shape_id)
            if df_t.empty:
                st.info("No trip found for the selected filters.")
                st.stop()

            # If multiple rows match, choose the first deterministically
            meta_df = df_t.sort_values(
                ["distance_km", "kwh_per_km_medium_duty", "kwh_per_km_heavy_duty"],
                ascending=[False, True, True],
            ).head(1)
            sid = str(meta_df.iloc[0]["shape_id"])

            show_stops = st.checkbox("Show control events (signals/stop signs)", value=True)

    # Guard
    if not sid:
        st.info("Please select filters to view a shape.")
        st.stop()

    # Select the row
    meta_df = summary_df[summary_df["shape_id"].astype(str) == str(sid)]
    if meta_df.empty:
        st.error(f"No summary entry found for shape_id {sid}.")
        st.stop()
    meta = meta_df.iloc[0]

    # Load artifacts
    elev_df   = load_elev(elev_dir, str(sid))
    edges_df  = load_edges(edges_dir, str(sid))
    mandatory_df = load_mandatory(mandatory_dir, str(sid)) if show_stops else pd.DataFrame()

    if "last_sid" not in st.session_state or st.session_state["last_sid"] != sid:
        dmin = float(elev_df["dist_m"].min()) / 1000.0
        st.session_state["cursor_km"] = dmin
    st.session_state["last_sid"] = sid

    coords_latlon = list(zip(elev_df["lat"].astype(float), elev_df["lon"].astype(float)))
    dist_along = elev_df["dist_m"].astype(float).to_numpy()
    center = coords_latlon[0] if coords_latlon else (0, 0)

    bridge_spans_idx = spans_from_elev_mask(elev_df, "bridge_mask")

    # ---------------------------
    # MAP (Folium)
    # ---------------------------
    # Make sure cursor state exists (even before drawing the slider)
    ensure_cursor_defaults(elev_df, key="cursor")
    compute_cursor_derived(elev_df, key="cursor")

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

    if show_stops and not mandatory_df.empty:
        mandatory_snapped = snap_events_to_coords(mandatory_df, elev_df)
        stops_fg = folium.FeatureGroup(name="Events", show=True)
        for _, r in mandatory_snapped.iterrows():
            ev = str(r.get("event", "")).lower()
            if duty == "medium" and ("will_stop" in r) and (not bool(r["will_stop"])):
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
        stops_fg.add_to(m)
    cur_latlon = st.session_state.get("cursor_latlon")
    if cur_latlon:
        folium.CircleMarker(
            location=cur_latlon,
            radius=6,
            color="red",
            fill=True,
            fill_opacity=0.95,
            tooltip=f"{st.session_state['cursor_km']:.3f} km",
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=520, width=None)

    # ===== Cursor UI BETWEEN MAP AND KPI =====
    dmin = float(elev_df["dist_m"].min()) / 1000.0
    dmax = float(elev_df["dist_m"].max()) / 1000.0
    st.slider(
        "Move Dot (Distance along route, km)",
        min_value=round(dmin, 3),
        max_value=round(dmax, 3),
        step=0.001,
        key="cursor_km",
    )
    # After user moves the slider, recompute derived values
    compute_cursor_derived(elev_df, key="cursor")

    # ---------------------------
    # KPIs
    # ---------------------------
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] label { font-size: 20px !important; font-weight: 700 !important; }
        div[data-testid="stMetric"] > div:nth-child(1) { font-size: 24px !important; font-weight: 700 !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.subheader("KPIs")
    if trip_type == "In Service":
        colA, colB, colC, colD = st.columns(4)
        with colB:
            if duty == "heavy":
                st.metric("Energy Consumption (kWh)", f"{meta['pack_used_kwh_heavy_duty']:.2f}")
            else:
                st.metric("Energy Consumption (kWh)", f"{meta['pack_used_kwh_medium_duty']:.2f}")
        with colC:
            if duty == "heavy":
                st.metric("kWh per km", f"{meta['kwh_per_km_heavy_duty']:.2f}")
            else:
                st.metric("kWh per km", f"{meta['kwh_per_km_medium_duty']:.2f}")
        with colD:
            st.metric("Original Bus Type", str(meta.get("Asset Class","N/A")))
        with colA:
            st.metric("Travel Distance (km)", f"{meta['distance_km']:.2f}")
    else:
        # Not In Service (Deadhead): KPIs = Distance, Energy, kWh/km
        dist_km = float(meta.get("distance_km", np.nan))
        if duty == "heavy":
            energy_kwh = float(meta.get("pack_used_kwh_heavy_duty", np.nan))
            kwh_per_km = float(meta.get("kwh_per_km_heavy_duty", np.nan))
        else:
            energy_kwh = float(meta.get("pack_used_kwh_medium_duty", np.nan))
            kwh_per_km = float(meta.get("kwh_per_km_medium_duty", np.nan))

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Travel Distance (km)", f"{dist_km:.2f}" if np.isfinite(dist_km) else "N/A")
        with col5:
            st.metric("Energy Consumption (kWh)", f"{energy_kwh:.2f}" if np.isfinite(energy_kwh) else "N/A")
        with col6:
            st.metric("kWh per km", f"{kwh_per_km:.2f}" if np.isfinite(kwh_per_km) else "N/A")

    # ---------------------------
    # CHARTS (Plotly)
    # ---------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Posted Speed Limit vs Distance")
        edges_for_steps = ensure_step_distances(edges_df, dist_along)
        ycol = "speed_limit_kph_smooth"
        xs_m, ys = to_steps(edges_for_steps, ycol)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=xs_m/1000.0, y=np.r_[ys, ys[-1]],
            mode="lines", line=dict(shape="hv"),
            name="Speed limit (kph)"
        ))

        ckm = float(st.session_state["cursor_km"])
        cm = float(st.session_state["cursor_m"])
        i = int(np.clip(np.searchsorted(xs_m, cm, side="right") - 1, 0, len(ys)-1))
        y_at_cursor = float(ys[i]) if np.isfinite(ys[i]) else None
        if y_at_cursor is not None:
            fig2.add_trace(go.Scatter(
                x=[ckm], y=[y_at_cursor],
                mode="markers+text",
                marker=dict(size=10, color="red"),
                text=[f"{y_at_cursor:.0f} kph"],
                textposition="top center",
                name="Cursor",
                showlegend=False
            ))

        fig2.update_layout(
            xaxis_title="Distance (km)",
            yaxis_title="Speed limit (kph)",
            margin=dict(l=30, r=10, t=30, b=30),
            height=320
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, width="stretch")

    with col2:
        st.subheader("Elevation vs Distance")
        y_elev = pd.to_numeric(
            elev_df["elev_m_smooth"],
            errors="coerce",
        )
        x_km = pd.to_numeric(elev_df["dist_m"], errors="coerce") / 1000.0

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_km, y=y_elev, mode="lines", name="Elevation"))

        # Find cursor point
        ckm = float(st.session_state["cursor_km"])
        cidx = int(st.session_state["cursor_idx"])
        y_at_cursor = float(y_elev.iloc[cidx]) if pd.notna(y_elev.iloc[cidx]) else None
        if y_at_cursor is not None:
            fig.add_trace(go.Scatter(
                x=[ckm], y=[y_at_cursor],
                mode="markers+text",
                marker=dict(size=10, color="red"),
                text=[f"{y_at_cursor:.1f} m"],
                textposition="top center",
                name="Cursor",
                showlegend=False
            ))
        for s, e in bridge_spans_idx:
            a_m = float(elev_df["dist_m"].iloc[s]); b_m = float(elev_df["dist_m"].iloc[max(s, e-1)])
            fig.add_vrect(x0=a_m/1000.0, x1=b_m/1000.0, fillcolor="purple", opacity=0.15, line_width=0)
        fig.update_layout(
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            margin=dict(l=30, r=10, t=30, b=30),
            height=320,
            hovermode="x"
        )
        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, width="stretch")

    # ---------------------------
    # Time Series (Speed, Grade, SOC) — NOW VS DISTANCE
    # ---------------------------
    mode_dir = (SIM_DIR if trip_type == "In Service" else DEADHEAD_SIM_DIR) / (
        "heavy" if duty == "heavy" else "medium"
    )
    ts_path = mode_dir / f"sim_{'heavy' if duty=='heavy' else 'medium'}_{sid}.parquet"

    if ts_path.exists():
        ts_df = pd.read_parquet(ts_path).copy()

        # We need time + speed to build distance
        if "time_seconds" in ts_df.columns and "speed_meters_per_second" in ts_df.columns:
            ts_df = ts_df.sort_values("time_seconds").reset_index(drop=True)

            # 1. Build raw cumulative distance from time & speed
            dt = ts_df["time_seconds"].diff().fillna(0.0)
            speed_mps = ts_df["speed_meters_per_second"].clip(lower=0.0)
            ts_df["dist_m_raw"] = (speed_mps * dt).cumsum()

            # 2. Align to elevation distance (so cursor & plots share the same x-axis)
            start_m = float(elev_df["dist_m"].iloc[0])
            elev_len = float(elev_df["dist_m"].iloc[-1] - start_m)
            max_raw = float(ts_df["dist_m_raw"].iloc[-1]) if len(ts_df) > 0 else 0.0

            if elev_len > 0.0 and max_raw > 0.0:
                scale = elev_len / max_raw
            else:
                scale = 1.0

            ts_df["dist_m"] = start_m + ts_df["dist_m_raw"] * scale
            ts_df["dist_km"] = ts_df["dist_m"] / 1000.0
        else:
            # If we don't have speed/time, just fall back to no distance plots
            ts_df["dist_m"] = np.nan
            ts_df["dist_km"] = np.nan

        # Helper: get value at cursor along distance
        def _value_at_cursor(df: pd.DataFrame, ycol: str):
            if df.empty or "dist_m" not in df.columns or ycol not in df.columns:
                return None, None
            cm = float(st.session_state.get("cursor_m", 0.0))
            xs = df["dist_m"].to_numpy()
            if len(xs) == 0:
                return None, None
            idx = int(np.clip(np.searchsorted(xs, cm, side="right") - 1, 0, len(xs) - 1))
            y = df[ycol].iloc[idx]
            if pd.isna(y):
                return None, None
            return float(df["dist_km"].iloc[idx]), float(y)

        ckm = float(st.session_state.get("cursor_km", 0.0))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Speed vs Distance")
            if "speed_meters_per_second" in ts_df.columns and "dist_km" in ts_df.columns:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=ts_df["dist_km"],
                        y=ts_df["speed_meters_per_second"] * 3.6,
                        mode="lines",
                        name="Speed (km/h)",
                    )
                )

                x_c, y_c = _value_at_cursor(ts_df, "speed_meters_per_second")
                if x_c is not None and y_c is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[x_c],
                            y=[y_c * 3.6],
                            mode="markers+text",
                            marker=dict(size=10, color="red"),
                            text=[f"{y_c*3.6:.1f} km/h"],
                            textposition="top center",
                            name="Cursor",
                            showlegend=False,
                        )
                    )

                fig.update_layout(
                    xaxis_title="Distance (km)",
                    yaxis_title="Speed (km/h)",
                    margin=dict(l=30, r=10, t=30, b=30),
                    height=320,
                    showlegend=False
                )
                st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Grade vs Distance")
            if "grade" in ts_df.columns and "dist_km" in ts_df.columns:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=ts_df["dist_km"],
                        y=ts_df["grade"],
                        mode="lines",
                        name="Grade",
                    )
                )

                x_c, y_c = _value_at_cursor(ts_df, "grade")
                if x_c is not None and y_c is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[x_c],
                            y=[y_c],
                            mode="markers+text",
                            marker=dict(size=10, color="red"),
                            text=[f"{y_c:.3f}"],
                            textposition="top center",
                            name="Cursor",
                            showlegend=False,
                        )
                    )

                fig.update_layout(
                    xaxis_title="Distance (km)",
                    yaxis_title="Grade (fraction)",
                    margin=dict(l=30, r=10, t=30, b=30),
                    height=320,
                    showlegend=False
                )
                st.plotly_chart(fig, width="stretch")

        st.subheader("SOC vs Distance")
        if "soc" in ts_df.columns and "dist_km" in ts_df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ts_df["dist_km"],
                    y=ts_df["soc"],
                    mode="lines",
                    name="SOC",
                )
            )

            x_c, y_c = _value_at_cursor(ts_df, "soc")
            if x_c is not None and y_c is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[x_c],
                        y=[y_c],
                        mode="markers+text",
                        marker=dict(size=10, color="red"),
                        text=[f"{y_c:.3f}"],
                        textposition="top center",
                        name="Cursor",
                        showlegend=False,
                    )
                )

            fig.update_layout(
                xaxis_title="Distance (km)",
                yaxis_title="SOC",
                margin=dict(l=30, r=10, t=30, b=30),
                height=320,
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info(f"No timeseries parquet found for shape {sid} ({duty}).")


# ---------------------------
# Standalone entry point
# ---------------------------
if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
        page_title="Energy Model--Route Level Summary Panel",
    )
    render_trip_panel()
