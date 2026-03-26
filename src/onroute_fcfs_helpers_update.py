from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Charging curve and constants
# ---------------------------------------------------------------------
t_sec = np.array([0, 360, 600, 1800, 2400, 2900, 3800, 4200, 4800, 5200], dtype=float)
p_kw = np.array([250, 250.2, 251.9, 258.7, 270, 230, 140, 120, 100, 80], dtype=float)
soc_pct_curve = np.array([8, 21.9, 26.6, 49.7, 60, 70, 78, 84, 88, 90], dtype=float)

BATTERY_KWH = 564.0
MAX_SOC_FRAC = 0.90
MAX_ENERGY_KWH = BATTERY_KWH * MAX_SOC_FRAC
MIN_CHARGE_SEC = 180
CHARGE_TRIGGER_SOC_PCT = 70.0
SOC_THRESHOLD_PERCENT = 20.0
INTERLINE_TYPES = {"interline"}


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def parse_combined_sequence_json(raw: Any) -> List[Dict[str, Any]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, list):
        return raw
    s = str(raw).strip()
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)



def time_to_sec(tstr: str) -> int:
    h, m, s = map(int, str(tstr).split(":"))
    return h * 3600 + m * 60 + s



def sec_to_hhmmss_no_wrap(sec: int) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"



def soc_to_time(s: float) -> float:
    s_clipped = np.clip(s, soc_pct_curve.min(), soc_pct_curve.max())
    return float(np.interp(s_clipped, soc_pct_curve, t_sec))



def time_to_soc(t: float) -> float:
    t_clipped = np.clip(t, t_sec.min(), t_sec.max())
    return float(np.interp(t_clipped, t_sec, soc_pct_curve))



def charge_session(start_soc_pct: float, duration_sec: float, max_soc_pct: float = 90.0, n_steps: int = 200):
    if duration_sec <= 0 or start_soc_pct >= max_soc_pct:
        return 0.0, float(min(start_soc_pct, max_soc_pct)), None, None

    t0 = soc_to_time(start_soc_pct)
    t_max = soc_to_time(max_soc_pct)
    t1 = min(t0 + duration_sec, t_max)
    if t1 <= t0:
        return 0.0, float(time_to_soc(t0)), None, None

    ts = np.linspace(t0, t1, n_steps)
    ps = np.interp(ts, t_sec, p_kw)
    energy_kwh = np.trapezoid(ps, ts) / 3600.0
    end_soc = time_to_soc(t1)
    return float(energy_kwh), float(end_soc), ts, ps



def _code_in_matched(code: Any, matched_codes: Set[int]) -> bool:
    if code is None:
        return False
    if isinstance(code, int):
        return code in matched_codes
    if isinstance(code, float):
        if pd.isna(code):
            return False
        return int(code) in matched_codes
    if isinstance(code, str):
        s = code.strip()
        return s.isdigit() and int(s) in matched_codes
    return False



def _normalize_stop_code(code: Any) -> Optional[int]:
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    if isinstance(code, int):
        return code
    if isinstance(code, float):
        return int(code)
    s = str(code).strip()
    return int(s) if s.isdigit() else None



def _choose_charge_stop(prev_end_code: Any, next_start_code: Any, matched_codes: Set[int]) -> Optional[int]:
    p = _normalize_stop_code(prev_end_code)
    n = _normalize_stop_code(next_start_code)
    if p is not None and p in matched_codes:
        return p
    if n is not None and n in matched_codes:
        return n
    return None



def _infer_trip_distance_km(trip: Dict[str, Any]) -> float:
    if "trip_distance_km" in trip and trip["trip_distance_km"] is not None:
        return float(trip["trip_distance_km"])
    if trip.get("start_time") and trip.get("end_time"):
        t0 = time_to_sec(trip["start_time"])
        t1 = time_to_sec(trip["end_time"])
        dur_h = max(0.0, (t1 - t0) / 3600.0)
        return dur_h * 30.0
    return 1.0



def _hash_dict(d: Dict[str, int]) -> str:
    text = "|".join(f"{k}:{d[k]}" for k in sorted(d))
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def _concat_nonempty(dfs: List[pd.DataFrame], default_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Concatenate only non-empty, non-all-NA DataFrames to avoid pandas FutureWarning
    on dtype inference with empty/all-NA entries.
    """
    valid = []
    for df in dfs:
        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            continue
        if df.empty:
            continue

        # Drop columns that are entirely NA before concat
        cleaned = df.dropna(axis=1, how="all").copy()
        if cleaned.empty and len(cleaned.columns) == 0:
            continue

        valid.append(cleaned)

    if not valid:
        return pd.DataFrame(columns=default_columns or [])

    return pd.concat(valid, ignore_index=True)



# ---------------------------------------------------------------------
# GTFS / candidate loaders
# ---------------------------------------------------------------------
def load_candidate_stop_map(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df = df.dropna(subset=["candidate_name", "stop_code"]).copy()
    df["candidate_name"] = df["candidate_name"].astype(str).str.strip()
    df["stop_code"] = pd.to_numeric(df["stop_code"], errors="coerce")
    df = df.dropna(subset=["stop_code"]).copy()
    df["stop_code"] = df["stop_code"].astype(int)
    return df



def build_candidate_dict(candidate_df: pd.DataFrame) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {}
    if candidate_df.empty:
        return out
    for name, g in candidate_df.groupby("candidate_name"):
        out[str(name)] = set(pd.to_numeric(g["stop_code"], errors="coerce").dropna().astype(int).tolist())
    return out



def build_stop_to_candidate(candidate_df: pd.DataFrame) -> Dict[int, str]:
    tmp = candidate_df[["stop_code", "candidate_name"]].dropna().drop_duplicates(subset=["stop_code"]).copy()
    tmp["stop_code"] = tmp["stop_code"].astype(int)
    tmp["candidate_name"] = tmp["candidate_name"].astype(str)
    return dict(zip(tmp["stop_code"], tmp["candidate_name"]))



def build_trip_to_route_short_name(gtfs_dir: str | Path) -> Dict[str, str]:
    gtfs_dir = Path(gtfs_dir)
    trips = pd.read_csv(gtfs_dir / "trips.txt", dtype=str, low_memory=False)[["trip_id", "route_id"]].dropna()
    routes = pd.read_csv(gtfs_dir / "routes.txt", dtype=str, low_memory=False)[["route_id", "route_short_name"]].dropna()
    t2r = trips.merge(routes, on="route_id", how="left")
    return dict(zip(t2r["trip_id"], t2r["route_short_name"]))



def build_stop_code_to_stop_name(gtfs_dir: str | Path) -> Dict[int, str]:
    gtfs_dir = Path(gtfs_dir)
    stops = pd.read_csv(gtfs_dir / "stops.txt", dtype=str, low_memory=False)
    stops = stops.dropna(subset=["stop_code", "stop_name"]).copy()
    stops["stop_code_int"] = pd.to_numeric(stops["stop_code"], errors="coerce")
    stops = stops.dropna(subset=["stop_code_int"])
    stops["stop_code_int"] = stops["stop_code_int"].astype(int)
    return dict(zip(stops["stop_code_int"], stops["stop_name"]))

def build_events_for_service_id(
    blocks_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    gtfs_dir: str | Path,
    service_id: int,
    mode: str = "heavy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = blocks_df[blocks_df["service_id"] == service_id].copy()
    return build_events_for_blocks(sub, candidate_df, gtfs_dir, mode=mode, persist_dir=None)


def build_final_proposed_dispensers(
    blocks_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    gtfs_dir: str | Path,
    mode: str = "heavy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekday_events, weekday_disp = build_events_for_service_id(blocks_df, candidate_df, gtfs_dir, 1, mode=mode)
    saturday_events, saturday_disp = build_events_for_service_id(blocks_df, candidate_df, gtfs_dir, 2, mode=mode)
    sunday_events, sunday_disp = build_events_for_service_id(blocks_df, candidate_df, gtfs_dir, 3, mode=mode)

    weekday_disp = weekday_disp.rename(columns={"dispensers_needed": "weekday_dispensers_needed"})
    saturday_disp = saturday_disp.rename(columns={"dispensers_needed": "saturday_dispensers_needed"})
    sunday_disp = sunday_disp.rename(columns={"dispensers_needed": "sunday_dispensers_needed"})

    disp_final = (
        weekday_disp[["candidate_name", "weekday_dispensers_needed"]]
        .merge(saturday_disp[["candidate_name", "saturday_dispensers_needed"]], on="candidate_name", how="outer")
        .merge(sunday_disp[["candidate_name", "sunday_dispensers_needed"]], on="candidate_name", how="outer")
        .fillna(0)
    )

    for c in ["weekday_dispensers_needed", "saturday_dispensers_needed", "sunday_dispensers_needed"]:
        disp_final[c] = disp_final[c].astype(int)

    disp_final["final_proposed_dispensers"] = disp_final[
        ["weekday_dispensers_needed", "saturday_dispensers_needed", "sunday_dispensers_needed"]
    ].max(axis=1)

    event_columns = [
        "block_id", "mode", "line_group", "block_number", "asset_class_new",
        "depot_code", "service_id", "service_day",
        "prev_trip_id", "next_trip_id",
        "prev_route_short_name", "next_route_short_name",
        "prev_trip_end_stop_code", "next_trip_start_stop_code",
        "prev_trip_end_stop_name", "next_trip_start_stop_name",
        "stop_code", "candidate_name",
        "start_sec", "end_sec", "event_type",
        "soc_start_pct", "soc_end_pct", "charged_kwh",
        "duration_sec", "duration_min", "start_dt", "end_dt",
    ]

    all_events = _concat_nonempty(
        [weekday_events, saturday_events, sunday_events],
        default_columns=event_columns,
    )

    return all_events, disp_final

# ---------------------------------------------------------------------
# Baseline profile logic (no capacity limit; used for depot-only and path)
# ---------------------------------------------------------------------
def build_block_profile_with_charging(
    block_trips: List[Dict[str, Any]],
    matched_codes: Set[int],
    mode: str = "medium",
    layover_assume_min: int = 8,
    prep_time_min: int = 3,
    charge_trigger_soc_pct: float = CHARGE_TRIGGER_SOC_PCT,
) -> pd.DataFrame:
    if not block_trips:
        return pd.DataFrame(
            columns=[
                "dist_km", "soc_pct", "net_energy_kwh", "phase",
                "cum_used_kwh", "cum_charged_kwh", "stop_code", "charge_kwh",
                "charge_duration_sec",
            ]
        )

    energy_key = "energy_medium_kwh" if mode == "medium" else "energy_heavy_kwh"
    buffer_sec = int(prep_time_min * 60)

    def clamp_kwh(x: float) -> float:
        return min(float(MAX_ENERGY_KWH), float(x))

    def soc_from_kwh(kwh_val: float) -> float:
        return 100.0 * (kwh_val / float(BATTERY_KWH))

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

    kwh = clamp_kwh(MAX_ENERGY_KWH)
    total_used = 0.0
    total_charged = 0.0
    soc_pct_curr = soc_from_kwh(kwh)
    dist_cum_km = 0.0

    xs, socs, net_energy, phase = [], [], [], []
    used_hist, charged_hist, stop_code_hist = [], [], []
    charge_kwh_hist, charge_dur_s_hist = [], []

    def record_point(dist: float, soc_pct_val: float, tag: str, stop_code=None, charge_kwh=0.0, charge_duration_sec=0.0):
        net_kwh = MAX_ENERGY_KWH - total_used + total_charged
        xs.append(float(dist))
        socs.append(float(soc_pct_val))
        net_energy.append(float(net_kwh))
        phase.append(str(tag))
        used_hist.append(float(total_used))
        charged_hist.append(float(total_charged))
        stop_code_hist.append(str(stop_code).strip() if stop_code is not None else None)
        charge_kwh_hist.append(float(charge_kwh or 0.0))
        charge_dur_s_hist.append(float(charge_duration_sec or 0.0))

    def apply_charge_here(duration_sec: int, chosen_stop_code: Optional[int]) -> bool:
        nonlocal kwh, total_charged, soc_pct_curr
        if duration_sec < MIN_CHARGE_SEC or soc_pct_curr >= charge_trigger_soc_pct:
            return False
        delta_e, _, _, _ = charge_session(soc_pct_curr, duration_sec)
        kwh_before = kwh
        kwh = clamp_kwh(kwh + delta_e)
        actual_delta = kwh - kwh_before
        if actual_delta <= 0:
            return False
        total_charged += actual_delta
        soc_pct_curr = soc_from_kwh(kwh)
        record_point(dist_cum_km, soc_pct_curr, "charge", chosen_stop_code, actual_delta, duration_sec)
        return True

    record_point(dist_cum_km, soc_pct_curr, "start")

    for i, trip in enumerate(trips):
        ttype = trip.get("type")
        energy_key = "energy_medium_kwh" if mode == "medium" else "energy_heavy_kwh"
        use = float(trip.get(energy_key, 0.0) or 0.0)
        dist_km = _infer_trip_distance_km(trip)
        dist_end = dist_cum_km + dist_km

        if ttype in INTERLINE_TYPES and 0 < i < len(trips) - 1:
            prev_trip = trips[i - 1]
            next_trip = trips[i + 1]
            if prev_trip.get("type") == "in_service" and next_trip.get("type") == "in_service":
                inter_start_code = trip.get("start_stop_code")
                inter_end_code = trip.get("end_stop_code")
                has_start = _code_in_matched(inter_start_code, matched_codes)
                has_end = _code_in_matched(inter_end_code, matched_codes)
                if has_start or has_end:
                    if has_start and not has_end:
                        apply_charge_here(max(0, layover_assume_min * 60 - buffer_sec), _normalize_stop_code(inter_start_code))
                    total_used += use
                    kwh = clamp_kwh(kwh - use)
                    soc_pct_curr = soc_from_kwh(kwh)
                    record_point(dist_end, soc_pct_curr, "drive")
                    dist_cum_km = dist_end
                    if has_end:
                        apply_charge_here(max(0, layover_assume_min * 60 - buffer_sec), _normalize_stop_code(inter_end_code))
                    continue

        total_used += use
        kwh = clamp_kwh(kwh - use)
        soc_pct_curr = soc_from_kwh(kwh)
        record_point(dist_end, soc_pct_curr, "drive")
        dist_cum_km = dist_end

        if ttype == "pull_out" and first_in_idx is not None:
            next_trip = trips[first_in_idx]
            prev_end_code = trip.get("end_stop_code")
            next_start_code = next_trip.get("start_stop_code")
            eligible = _code_in_matched(prev_end_code, matched_codes) or _code_in_matched(next_start_code, matched_codes)
            if eligible:
                chosen_stop = _choose_charge_stop(prev_end_code, next_start_code, matched_codes)
                apply_charge_here(max(0, layover_assume_min * 60 - buffer_sec), chosen_stop)
            continue

        if ttype == "in_service" and i < len(trips) - 1:
            next_trip = trips[i + 1]
            if next_trip.get("type") == "pull_in":
                continue
            if next_trip.get("type") == "in_service":
                layover = (next_trip["start_sec"] - trip["end_sec"]) if (trip["end_sec"] is not None and next_trip["start_sec"] is not None) else 0
                prev_end_code = trip.get("end_stop_code")
                next_start_code = next_trip.get("start_stop_code")
                eligible = _code_in_matched(prev_end_code, matched_codes) or _code_in_matched(next_start_code, matched_codes)
                if eligible:
                    chosen_stop = _choose_charge_stop(prev_end_code, next_start_code, matched_codes)
                    apply_charge_here(max(0, layover - buffer_sec), chosen_stop)
                continue

    return pd.DataFrame({
        "dist_km": xs,
        "soc_pct": socs,
        "net_energy_kwh": net_energy,
        "phase": phase,
        "cum_used_kwh": used_hist,
        "cum_charged_kwh": charged_hist,
        "stop_code": stop_code_hist,
        "charge_kwh": charge_kwh_hist,
        "charge_duration_sec": charge_dur_s_hist,
    })


# ---------------------------------------------------------------------
# Event extraction from on_route_charge_assembly_part2 logic
# ---------------------------------------------------------------------
def extract_charge_events_for_block(
    block_id: str,
    block_trips: List[Dict[str, Any]],
    matched_codes: Set[int],
    stop_to_candidate: Dict[int, str],
    mode: str = "heavy",
    layover_assume_min: int = 8,
    prep_time_min: int = 3,
    charge_trigger_soc_pct: float = CHARGE_TRIGGER_SOC_PCT,
    block_meta: Optional[Dict[str, Any]] = None,
    trip_to_route_short: Optional[Dict[str, str]] = None,
    stop_code_to_name: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    buffer_total = int(prep_time_min * 60)
    buffer_half = buffer_total // 2

    def clamp_kwh(x: float) -> float:
        return min(float(MAX_ENERGY_KWH), float(x))

    def soc_from_kwh(kwh_val: float) -> float:
        return 100.0 * (kwh_val / float(BATTERY_KWH))

    def code_ok(code: Any) -> bool:
        return _code_in_matched(code, matched_codes)

    def stop_name_from_code(code: Any):
        code_int = _normalize_stop_code(code)
        if stop_code_to_name is None or code_int is None:
            return None
        return stop_code_to_name.get(code_int)

    def get_route_short_name_from_trip(t: Optional[Dict[str, Any]]):
        if trip_to_route_short is None or t is None:
            return None
        tid = t.get("trip_id")
        return None if tid is None else trip_to_route_short.get(str(tid))

    trips = []
    for t in block_trips:
        t2 = dict(t)
        if t2.get("start_time") and t2.get("end_time"):
            t2["start_sec"] = time_to_sec(t2["start_time"])
            t2["end_sec"] = time_to_sec(t2["end_time"])
        else:
            t2["start_sec"] = None
            t2["end_sec"] = None
        trips.append(t2)

    in_idxs = [i for i, t in enumerate(trips) if t.get("type") == "in_service"]
    first_in_idx = in_idxs[0] if in_idxs else None
    kwh = clamp_kwh(MAX_ENERGY_KWH)
    soc_pct_curr = soc_from_kwh(kwh)
    events: List[Dict[str, Any]] = []
    meta = block_meta or {}

    def maybe_add_and_apply_charge(location_code, s0, s1, event_type, prev_trip=None, next_trip=None, prev_end_stop_code=None, next_start_stop_code=None):
        nonlocal kwh, soc_pct_curr
        if soc_pct_curr < SOC_THRESHOLD_PERCENT:
            return
        if location_code is None or s0 is None or s1 is None or s1 <= s0:
            return
        duration_sec = int(s1 - s0)
        if duration_sec < MIN_CHARGE_SEC or soc_pct_curr >= charge_trigger_soc_pct:
            return
        code_int = int(location_code)
        cand = stop_to_candidate.get(code_int)
        if cand is None:
            return
        event = {
            "block_id": block_id,
            "mode": mode,
            "line_group": meta.get("line_group"),
            "block_number": meta.get("block_number"),
            "asset_class_new": meta.get("asset_class_new"),
            "depot_code": meta.get("depot_code"),
            "service_id": meta.get("service_id"),
            "service_day": meta.get("service_day"),
            "prev_trip_id": None if prev_trip is None else prev_trip.get("trip_id"),
            "next_trip_id": None if next_trip is None else next_trip.get("trip_id"),
            "prev_route_short_name": get_route_short_name_from_trip(prev_trip),
            "next_route_short_name": get_route_short_name_from_trip(next_trip),
            "prev_trip_end_stop_code": _normalize_stop_code(prev_end_stop_code),
            "next_trip_start_stop_code": _normalize_stop_code(next_start_stop_code),
            "prev_trip_end_stop_name": stop_name_from_code(prev_end_stop_code),
            "next_trip_start_stop_name": stop_name_from_code(next_start_stop_code),
            "stop_code": code_int,
            "candidate_name": cand,
            "start_sec": int(s0),
            "end_sec": int(s1),
            "event_type": event_type,
            "soc_start_pct": float(soc_pct_curr),
        }
        delta_e, _, _, _ = charge_session(soc_pct_curr, duration_sec)
        kwh_before = kwh
        kwh = clamp_kwh(kwh + delta_e)
        soc_pct_curr = soc_from_kwh(kwh)
        event["soc_end_pct"] = float(soc_pct_curr)
        event["charged_kwh"] = float(kwh - kwh_before)
        event["duration_sec"] = duration_sec
        event["duration_min"] = duration_sec / 60.0
        events.append(event)

    for i, trip in enumerate(trips):
        ttype = trip.get("type")
        energy_key = "energy_medium_kwh" if mode == "medium" else "energy_heavy_kwh"
        use = float(trip.get(energy_key, 0.0) or 0.0)

        if ttype in INTERLINE_TYPES and 0 < i < len(trips) - 1:
            prev_trip = trips[i - 1]
            next_trip = trips[i + 1]
            if prev_trip.get("type") == "in_service" and next_trip.get("type") == "in_service":
                inter_start_code = trip.get("start_stop_code")
                inter_end_code = trip.get("end_stop_code")
                has_start = code_ok(inter_start_code)
                has_end = code_ok(inter_end_code)
                assumed_charge = max(0, layover_assume_min * 60 - buffer_total)
                prev_end_code_ctx = prev_trip.get("end_stop_code")
                next_start_code_ctx = next_trip.get("start_stop_code")
                if has_start and not has_end and assumed_charge > 0 and trip.get("start_sec") is not None:
                    s0 = trip["start_sec"] + buffer_half
                    s1 = s0 + assumed_charge
                    maybe_add_and_apply_charge(inter_start_code, s0, s1, "interline_start", prev_trip, next_trip, prev_end_code_ctx, next_start_code_ctx)
                kwh = clamp_kwh(kwh - use)
                soc_pct_curr = soc_from_kwh(kwh)
                if has_end and assumed_charge > 0 and trip.get("end_sec") is not None:
                    s1 = trip["end_sec"] - buffer_half
                    s0 = s1 - assumed_charge
                    maybe_add_and_apply_charge(inter_end_code, s0, s1, "interline_end", prev_trip, next_trip, prev_end_code_ctx, next_start_code_ctx)
                continue

        kwh = clamp_kwh(kwh - use)
        soc_pct_curr = soc_from_kwh(kwh)

        if ttype == "pull_out" and first_in_idx is not None:
            next_trip = trips[first_in_idx]
            prev_end_code = trip.get("end_stop_code")
            next_start_code = next_trip.get("start_stop_code")
            eligible = code_ok(prev_end_code) or code_ok(next_start_code)
            assumed_charge = max(0, layover_assume_min * 60 - buffer_total)
            if eligible and assumed_charge > 0 and trip.get("end_sec") is not None:
                s0 = trip["end_sec"] + buffer_half
                s1 = s0 + assumed_charge
                loc = prev_end_code if code_ok(prev_end_code) else next_start_code
                maybe_add_and_apply_charge(loc, s0, s1, "pull_out_assumed", trip, next_trip, prev_end_code, next_start_code)
            continue

        if ttype == "in_service" and i < len(trips) - 1:
            next_trip = trips[i + 1]
            if next_trip.get("type") == "pull_in":
                continue
            if next_trip.get("type") == "in_service":
                prev_end_code = trip.get("end_stop_code")
                next_start_code = next_trip.get("start_stop_code")
                eligible = code_ok(prev_end_code) or code_ok(next_start_code)
                if eligible and (trip.get("end_sec") is not None) and (next_trip.get("start_sec") is not None):
                    s0 = trip["end_sec"] + buffer_half
                    s1 = next_trip["start_sec"] - buffer_half
                    loc = prev_end_code if code_ok(prev_end_code) else next_start_code
                    maybe_add_and_apply_charge(loc, s0, s1, "in_service_layover", trip, next_trip, prev_end_code, next_start_code)
            continue

    return pd.DataFrame(events)

def simulate_all_blocks_with_allocation_by_service_id(
    blocks_df: pd.DataFrame,
    events_df: pd.DataFrame,
    installed_disp: Dict[str, int],
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assigned_all = []
    report_all = []
    profiles_all = []

    for sid in [1, 2, 3]:
        blocks_sid = blocks_df[blocks_df["service_id"] == sid].copy()
        events_sid = events_df[events_df["service_id"] == sid].copy() if not events_df.empty else pd.DataFrame()

        if blocks_sid.empty:
            continue

        assigned_sid = allocate_sessions_fcfs(events_sid, installed_disp)
        report_sid, profiles_sid = simulate_all_blocks_with_allocation(blocks_sid, assigned_sid, mode=mode)

        assigned_all.append(assigned_sid)
        report_all.append(report_sid)
        if profiles_sid is not None and not profiles_sid.empty:
            profiles_all.append(profiles_sid)

    assigned_events_df = _concat_nonempty(assigned_all)
    report_df = _concat_nonempty(report_all)
    profiles_df = _concat_nonempty(profiles_all)

    return assigned_events_df, report_df, profiles_df


def build_events_for_blocks(
    blocks_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    gtfs_dir: str | Path,
    mode: str = "heavy",
    persist_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stop_to_candidate = build_stop_to_candidate(candidate_df)
    matched_codes = set(stop_to_candidate.keys())
    trip_to_route_short = build_trip_to_route_short_name(gtfs_dir)
    stop_code_to_name = build_stop_code_to_stop_name(gtfs_dir)

    all_events: List[pd.DataFrame] = []
    for _, row in blocks_df.iterrows():
        block_id = row["block_id"]
        trips = row["block_trips"]
        meta = {
            "line_group": row.get("line_group"),
            "block_number": row.get("block_number"),
            "asset_class_new": row.get("asset_class_new"),
            "depot_code": row.get("depot_code"),
            "service_id": row.get("service_id"),
            "service_day": row.get("service_day"),
        }
        ev = extract_charge_events_for_block(
            block_id,
            trips,
            matched_codes,
            stop_to_candidate=stop_to_candidate,
            mode=mode,
            block_meta=meta,
            trip_to_route_short=trip_to_route_short,
            stop_code_to_name=stop_code_to_name,
        )
        if not ev.empty:
            all_events.append(ev)

    event_columns = [
        "block_id", "mode", "line_group", "block_number", "asset_class_new",
        "depot_code", "service_id", "service_day",
        "prev_trip_id", "next_trip_id",
        "prev_route_short_name", "next_route_short_name",
        "prev_trip_end_stop_code", "next_trip_start_stop_code",
        "prev_trip_end_stop_name", "next_trip_start_stop_name",
        "stop_code", "candidate_name",
        "start_sec", "end_sec", "event_type",
        "soc_start_pct", "soc_end_pct", "charged_kwh",
        "duration_sec", "duration_min",
    ]

    events_df = _concat_nonempty(all_events, default_columns=event_columns)
    if not events_df.empty:
        events_df["mode"] = mode
        events_df["start_dt"] = events_df["start_sec"].apply(sec_to_hhmmss_no_wrap)
        events_df["end_dt"] = events_df["end_sec"].apply(sec_to_hhmmss_no_wrap)

    disp_df = dispensers_needed_by_candidate(events_df)

    if persist_dir is not None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        events_path = persist_dir / f"charging_events_fcfs_{mode}.parquet"
        disp_path = persist_dir / f"dispensers_needed_by_candidate_fcfs_{mode}.parquet"
        events_df.to_parquet(events_path, index=False)
        disp_df.to_parquet(disp_path, index=False)

    return events_df, disp_df



def dispensers_needed_by_candidate(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["candidate_name", "dispensers_needed", "num_sessions", "num_unique_blocks"])
    out_rows = []
    for cand, g in events_df.groupby("candidate_name"):
        marks = []
        for _, r in g.iterrows():
            marks.append((int(r["start_sec"]), +1))
            marks.append((int(r["end_sec"]), -1))
        marks.sort(key=lambda x: (x[0], x[1]))
        cur = 0
        mx = 0
        for _, delta in marks:
            cur += delta
            mx = max(mx, cur)
        out_rows.append({
            "candidate_name": cand,
            "dispensers_needed": int(mx),
            "num_sessions": int(len(g)),
            "num_unique_blocks": int(g["block_id"].nunique()),
        })
    return pd.DataFrame(out_rows).sort_values(["dispensers_needed", "candidate_name"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------------
# FCFS allocation
# ---------------------------------------------------------------------
def allocate_sessions_fcfs(events_df: pd.DataFrame, installed_disp: Dict[str, int]) -> pd.DataFrame:
    if events_df.empty:
        return events_df.assign(assigned=False, dispenser_idx=pd.NA, reject_reason="no_events")

    assigned_chunks: List[pd.DataFrame] = []
    for cand, g in events_df.groupby("candidate_name", dropna=False):
        g = g.sort_values(["start_sec", "end_sec", "block_id"], kind="stable").copy()
        n = int(installed_disp.get(cand, 0))
        if n <= 0:
            g["assigned"] = False
            g["dispenser_idx"] = pd.NA
            g["reject_reason"] = "no_dispenser"
            assigned_chunks.append(g)
            continue

        avail = [0] * n
        assigns = []
        disp_idxs = []
        reasons = []
        for _, r in g.iterrows():
            start_sec = int(r["start_sec"])
            eligible = [i for i, ready in enumerate(avail) if ready <= start_sec]
            if eligible:
                i_sel = min(eligible, key=lambda i: (avail[i], i))
                avail[i_sel] = int(r["end_sec"])
                assigns.append(True)
                disp_idxs.append(i_sel + 1)
                reasons.append("")
            else:
                assigns.append(False)
                disp_idxs.append(pd.NA)
                reasons.append("busy")
        g["assigned"] = assigns
        g["dispenser_idx"] = disp_idxs
        g["reject_reason"] = reasons
        assigned_chunks.append(g)

    out = _concat_nonempty(assigned_chunks)
    out["installed_dispensers"] = out["candidate_name"].map(installed_disp).fillna(0).astype(int)
    return out


# ---------------------------------------------------------------------
# Block simulation using accepted events only
# ---------------------------------------------------------------------
def _extract_accepted_event_points(assigned_events_df: pd.DataFrame) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    if assigned_events_df.empty:
        return {}
    accepted = assigned_events_df[assigned_events_df["assigned"] == True].copy()
    if accepted.empty:
        return {}
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for (block_id, mode), g in accepted.groupby(["block_id", "mode"], dropna=False):
        pts = []
        for _, r in g.sort_values(["start_sec", "end_sec"], kind="stable").iterrows():
            pts.append({
                "start_sec": int(r["start_sec"]),
                "end_sec": int(r["end_sec"]),
                "duration_sec": int(r.get("duration_sec", int(r["end_sec"]) - int(r["start_sec"]))),
                "stop_code": _normalize_stop_code(r.get("stop_code")),
                "candidate_name": r.get("candidate_name"),
                "event_type": r.get("event_type"),
                "charged_kwh_request": float(r.get("charged_kwh", np.nan)) if pd.notna(r.get("charged_kwh", np.nan)) else np.nan,
            })
        out[(str(block_id), str(mode))] = pts
    return out



def build_block_profile_with_assigned_events(
    block_trips: List[Dict[str, Any]],
    assigned_points: List[Dict[str, Any]],
    mode: str = "heavy",
    layover_assume_min: int = 8,
    prep_time_min: int = 3,
    charge_trigger_soc_pct: float = CHARGE_TRIGGER_SOC_PCT,
) -> pd.DataFrame:
    """
    Re-simulate one block using only FCFS-accepted charging events, while
    preserving the same structural charging order used in assembly_part2.

    Key idea:
    - Walk through pull_out / in_service / interline in the same sequence as
      extract_charge_events_for_block().
    - At each charging opportunity, only apply charge if an accepted FCFS event
      matches that exact opportunity.
    - Do NOT replay accepted events by generic timestamp insertion.
    """

    if not block_trips:
        return pd.DataFrame(
            columns=[
                "dist_km",
                "soc_pct",
                "net_energy_kwh",
                "phase",
                "cum_used_kwh",
                "cum_charged_kwh",
                "stop_code",
                "charge_kwh",
                "charge_duration_sec",
                "candidate_name",
                "event_type",
                "start_sec",
                "end_sec",
            ]
        )

    energy_key = "energy_medium_kwh" if mode == "medium" else "energy_heavy_kwh"
    buffer_total = int(prep_time_min * 60)
    buffer_half = buffer_total // 2

    def clamp_kwh(x: float) -> float:
        return min(float(MAX_ENERGY_KWH), float(x))

    def soc_from_kwh(kwh_val: float) -> float:
        return 100.0 * (kwh_val / float(BATTERY_KWH))

    # ------------------------------------------------------------------
    # Normalize trips
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Normalize accepted FCFS events into an exact-match lookup
    # ------------------------------------------------------------------
    def _norm_event_key(pt: Dict[str, Any]) -> Tuple[Any, ...]:
        stop_code = _normalize_stop_code(pt.get("stop_code"))
        start_sec = None if pt.get("start_sec") is None else int(pt.get("start_sec"))
        end_sec = None if pt.get("end_sec") is None else int(pt.get("end_sec"))
        duration_sec = (
            int(pt.get("duration_sec"))
            if pt.get("duration_sec") is not None
            else (None if start_sec is None or end_sec is None else int(end_sec - start_sec))
        )
        return (
            pt.get("event_type"),
            stop_code,
            start_sec,
            end_sec,
            duration_sec,
        )

    accepted_lookup: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for pt in assigned_points or []:
        key = _norm_event_key(pt)
        accepted_lookup.setdefault(key, []).append(dict(pt))

    def pop_matching_assigned_event(
        event_type: str,
        stop_code: Any,
        s0: Any,
        s1: Any,
    ) -> Optional[Dict[str, Any]]:
        stop_code = _normalize_stop_code(stop_code)
        start_sec = None if s0 is None else int(s0)
        end_sec = None if s1 is None else int(s1)
        duration_sec = None if (start_sec is None or end_sec is None) else int(end_sec - start_sec)
        key = (event_type, stop_code, start_sec, end_sec, duration_sec)
        bucket = accepted_lookup.get(key)
        if bucket:
            return bucket.pop(0)
        return None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    kwh = clamp_kwh(MAX_ENERGY_KWH)
    total_used = 0.0
    total_charged = 0.0
    soc_pct_curr = soc_from_kwh(kwh)
    dist_cum_km = 0.0

    rows = []

    def record_point(
        dist: float,
        soc_pct_val: float,
        tag: str,
        stop_code=None,
        charge_kwh=0.0,
        charge_duration_sec=0.0,
        candidate_name=None,
        event_type=None,
        start_sec=None,
        end_sec=None,
    ):
        net_kwh = MAX_ENERGY_KWH - total_used + total_charged
        rows.append(
            {
                "dist_km": float(dist),
                "soc_pct": float(soc_pct_val),
                "net_energy_kwh": float(net_kwh),
                "phase": str(tag),
                "cum_used_kwh": float(total_used),
                "cum_charged_kwh": float(total_charged),
                "stop_code": str(stop_code).strip() if stop_code is not None else None,
                "charge_kwh": float(charge_kwh or 0.0),
                "charge_duration_sec": float(charge_duration_sec or 0.0),
                "candidate_name": candidate_name,
                "event_type": event_type,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )

    def apply_assigned_charge_if_matched(
        location_code: Any,
        s0: Any,
        s1: Any,
        event_type: str,
    ) -> bool:
        """
        Apply charge only if there is an accepted FCFS event matching this exact
        opportunity from the original extraction logic.
        """
        nonlocal kwh, total_charged, soc_pct_curr

        if soc_pct_curr < SOC_THRESHOLD_PERCENT:
            return False
        if location_code is None or s0 is None or s1 is None or s1 <= s0:
            return False

        accepted_evt = pop_matching_assigned_event(event_type, location_code, s0, s1)
        if accepted_evt is None:
            return False

        duration_sec = int(accepted_evt.get("duration_sec", int(s1 - s0)))
        if duration_sec < MIN_CHARGE_SEC or soc_pct_curr >= charge_trigger_soc_pct:
            return False

        delta_e, _, _, _ = charge_session(soc_pct_curr, duration_sec)
        kwh_before = kwh
        kwh = clamp_kwh(kwh + delta_e)
        actual_delta = kwh - kwh_before
        if actual_delta <= 0:
            return False

        total_charged += actual_delta
        soc_pct_curr = soc_from_kwh(kwh)

        record_point(
            dist_cum_km,
            soc_pct_curr,
            "charge",
            stop_code=_normalize_stop_code(location_code),
            charge_kwh=actual_delta,
            charge_duration_sec=duration_sec,
            candidate_name=accepted_evt.get("candidate_name"),
            event_type=event_type,
            start_sec=int(s0),
            end_sec=int(s1),
        )
        return True

    def apply_assigned_charge_by_candidate_stops(
        candidate_stop_codes: List[Any],
        s0: Any,
        s1: Any,
        event_type: str,
    ) -> bool:
        for code in candidate_stop_codes:
            ok = apply_assigned_charge_if_matched(code, s0, s1, event_type)
            if ok:
                return True
        return False

    # ------------------------------------------------------------------
    # Replay block in assembly_part2 order
    # ------------------------------------------------------------------
    record_point(dist_cum_km, soc_pct_curr, "start")

    for i, trip in enumerate(trips):
        ttype = trip.get("type")
        use = float(trip.get(energy_key, 0.0) or 0.0)
        dist_km = _infer_trip_distance_km(trip)
        dist_end = dist_cum_km + dist_km

        # --------------------------------------------------------------
        # Interline special case
        # --------------------------------------------------------------
        if ttype in INTERLINE_TYPES and 0 < i < len(trips) - 1:
            prev_trip = trips[i - 1]
            next_trip = trips[i + 1]

            if prev_trip.get("type") == "in_service" and next_trip.get("type") == "in_service":
                inter_start_code = trip.get("start_stop_code")
                inter_end_code = trip.get("end_stop_code")
                assumed_charge = max(0, layover_assume_min * 60 - buffer_total)

                # interline_start charge first
                if assumed_charge > 0 and trip.get("start_sec") is not None:
                    s0 = trip["start_sec"] + buffer_half
                    s1 = s0 + assumed_charge
                    apply_assigned_charge_by_candidate_stops(
                        [inter_start_code],
                        s0,
                        s1,
                        "interline_start",
                    )

                # then consume interline energy
                total_used += use
                kwh = clamp_kwh(kwh - use)
                soc_pct_curr = soc_from_kwh(kwh)
                record_point(dist_end, soc_pct_curr, "drive")
                dist_cum_km = dist_end

                # interline_end charge after drive
                if assumed_charge > 0 and trip.get("end_sec") is not None:
                    s1 = trip["end_sec"] - buffer_half
                    s0 = s1 - assumed_charge
                    apply_assigned_charge_by_candidate_stops(
                        [inter_end_code],
                        s0,
                        s1,
                        "interline_end",
                    )
                continue

        # --------------------------------------------------------------
        # Default trip energy consumption
        # --------------------------------------------------------------
        total_used += use
        kwh = clamp_kwh(kwh - use)
        soc_pct_curr = soc_from_kwh(kwh)
        record_point(dist_end, soc_pct_curr, "drive")
        dist_cum_km = dist_end

        # --------------------------------------------------------------
        # pull_out -> first in_service assumed charge
        # --------------------------------------------------------------
        if ttype == "pull_out" and first_in_idx is not None:
            next_trip = trips[first_in_idx]
            prev_end_code = trip.get("end_stop_code")
            next_start_code = next_trip.get("start_stop_code")
            assumed_charge = max(0, layover_assume_min * 60 - buffer_total)

            if assumed_charge > 0 and trip.get("end_sec") is not None:
                s0 = trip["end_sec"] + buffer_half
                s1 = s0 + assumed_charge
                apply_assigned_charge_by_candidate_stops(
                    [prev_end_code, next_start_code],
                    s0,
                    s1,
                    "pull_out_assumed",
                )
            continue

        # --------------------------------------------------------------
        # in_service layover -> next in_service
        # --------------------------------------------------------------
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

                    apply_assigned_charge_by_candidate_stops(
                        [prev_end_code, next_start_code],
                        s0,
                        s1,
                        "in_service_layover",
                    )
                continue

    return pd.DataFrame(rows)



def simulate_all_blocks_with_allocation(
    blocks_df: pd.DataFrame,
    assigned_events_df: pd.DataFrame,
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    accepted_map = _extract_accepted_event_points(assigned_events_df)
    results = []
    profile_rows = []
    for _, row in blocks_df.iterrows():
        block_id = str(row["block_id"])
        trips = row["block_trips"]
        rec = {"block_id": row["block_id"], "block_distance_km": block_total_distance_km(trips)}

        profile = build_block_profile_with_assigned_events(
            trips,
            accepted_map.get((block_id, mode), []),
            mode=mode,
        )
        if profile.empty:
            soc_min = math.nan
            soc_end = math.nan
            total_recv = math.nan
            success = "FAILURE"
        else:
            soc_min = float(profile["soc_pct"].min())
            soc_end = float(profile["soc_pct"].iloc[-1])
            total_recv = float(profile["cum_charged_kwh"].iloc[-1])
            success = "SUCCESS" if soc_min >= SOC_THRESHOLD_PERCENT else "FAILURE"
            tmp = profile.copy()
            tmp["block_id"] = row["block_id"]
            tmp["mode"] = mode
            profile_rows.append(tmp)

        rec[f"soc_min_{mode}_on_route"] = soc_min
        rec[f"soc_left_{mode}_percent_on_route_charge"] = soc_end
        rec[f"total_energy_received_{mode}"] = total_recv
        rec[f"{mode}_success_on_route_charge"] = success
        results.append(rec)

    results_df = pd.DataFrame(results)
    report_df = blocks_df.merge(results_df, on="block_id", how="left")
    profiles_df = _concat_nonempty(profile_rows)
    return report_df, profiles_df


# ---------------------------------------------------------------------
# KPI helpers / persistence
# ---------------------------------------------------------------------
def summarize_assignment(assigned_events_df: pd.DataFrame) -> pd.DataFrame:
    if assigned_events_df.empty:
        return pd.DataFrame(columns=["candidate_name", "installed_dispensers", "assigned_sessions", "rejected_sessions", "coverage_pct", "unique_blocks_assigned", "energy_assigned_kwh"])
    g = assigned_events_df.groupby("candidate_name", dropna=False)
    out = g.agg(
        installed_dispensers=("installed_dispensers", "max"),
        total_sessions=("candidate_name", "size"),
        assigned_sessions=("assigned", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        rejected_sessions=("assigned", lambda s: int((~pd.Series(s).fillna(False).astype(bool)).sum())),
        unique_blocks_assigned=("block_id", lambda s: int(assigned_events_df.loc[s.index][assigned_events_df.loc[s.index, "assigned"] == True]["block_id"].nunique())),
        energy_assigned_kwh=("charged_kwh", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())),
    ).reset_index()
    out["coverage_pct"] = np.where(out["total_sessions"] > 0, out["assigned_sessions"] / out["total_sessions"] * 100.0, np.nan)
    return out.sort_values(["installed_dispensers", "candidate_name"], ascending=[False, True]).reset_index(drop=True)

def block_total_distance_km(block_trips: List[Dict[str, Any]]) -> float:
    if not block_trips:
        return 0.0
    total = 0.0
    for trip in block_trips:
        total += _infer_trip_distance_km(trip)
    return float(total)

def compute_depot_summary(block_inv: pd.DataFrame, duty: str | None = None) -> pd.DataFrame:
    if block_inv.empty:
        return pd.DataFrame()
    tmp = block_inv.copy()

    if duty in {"medium", "heavy"}:
        dep_col = f"{duty}_success_depot_only"
        on_col = f"{duty}_success_on_route_charge"
        tmp["depot_ok"] = (tmp[dep_col] == "SUCCESS").astype(int)
        tmp["on_ok"] = (tmp[on_col] == "SUCCESS").astype(int)
        summary = (
            tmp.groupby("depot_code", dropna=False)
            .agg(
                total_blocks=("block_id", "size"),
                success_blocks_depot_only=("depot_ok", "sum"),
                success_blocks_on_route=("on_ok", "sum"),
            )
            .reset_index()
        )
        summary["success_rate_depot_only_%"] = summary["success_blocks_depot_only"] / summary["total_blocks"] * 100.0
        summary["success_rate_on_route_%"] = summary["success_blocks_on_route"] / summary["total_blocks"] * 100.0
        summary["duty"] = duty
        return summary.sort_values("depot_code", kind="stable")

    tmp["med_dep_ok"] = (tmp["medium_success_depot_only"] == "SUCCESS").astype(int)
    tmp["med_on_ok"] = (tmp["medium_success_on_route_charge"] == "SUCCESS").astype(int)
    tmp["hev_dep_ok"] = (tmp["heavy_success_depot_only"] == "SUCCESS").astype(int)
    tmp["hev_on_ok"] = (tmp["heavy_success_on_route_charge"] == "SUCCESS").astype(int)
    g = tmp.groupby("depot_code", dropna=False)
    summary = g.agg(
        total_blocks=("block_id", "size"),
        med_success_blocks_depot_only=("med_dep_ok", "sum"),
        med_success_blocks_on_route=("med_on_ok", "sum"),
        heavy_success_blocks_depot_only=("hev_dep_ok", "sum"),
        heavy_success_blocks_on_route=("hev_on_ok", "sum"),
    ).reset_index()
    for src, dst in [
        ("med_success_blocks_depot_only", "med_success_rate_depot_only_%"),
        ("med_success_blocks_on_route", "med_success_rate_on_route_%"),
        ("heavy_success_blocks_depot_only", "heavy_success_rate_depot_only_%"),
        ("heavy_success_blocks_on_route", "heavy_success_rate_on_route_%"),
    ]:
        summary[dst] = summary[src] / summary["total_blocks"] * 100.0
    return summary.sort_values("depot_code", kind="stable")


def compute_service_day_summary(block_inv: pd.DataFrame, duty: str | None = None) -> pd.DataFrame:
    if block_inv.empty:
        return pd.DataFrame()
    tmp = block_inv.copy()

    if duty in {"medium", "heavy"}:
        dep_col = f"{duty}_success_depot_only"
        on_col = f"{duty}_success_on_route_charge"
        tmp["depot_ok"] = (tmp[dep_col] == "SUCCESS").astype(int)
        tmp["on_ok"] = (tmp[on_col] == "SUCCESS").astype(int)
        summary = (
            tmp.groupby("service_day", dropna=False)
            .agg(
                total_blocks=("block_id", "size"),
                success_blocks_depot_only=("depot_ok", "sum"),
                success_blocks_on_route=("on_ok", "sum"),
            )
            .reset_index()
        )
        summary["success_rate_depot_only_%"] = summary["success_blocks_depot_only"] / summary["total_blocks"] * 100.0
        summary["success_rate_on_route_%"] = summary["success_blocks_on_route"] / summary["total_blocks"] * 100.0
        summary["duty"] = duty
        return summary

    tmp["med_dep_ok"] = (tmp["medium_success_depot_only"] == "SUCCESS").astype(int)
    tmp["med_on_ok"] = (tmp["medium_success_on_route_charge"] == "SUCCESS").astype(int)
    tmp["hev_dep_ok"] = (tmp["heavy_success_depot_only"] == "SUCCESS").astype(int)
    tmp["hev_on_ok"] = (tmp["heavy_success_on_route_charge"] == "SUCCESS").astype(int)
    g = tmp.groupby("service_day", dropna=False)
    summary = g.agg(
        total_blocks=("block_id", "size"),
        med_success_blocks_depot_only=("med_dep_ok", "sum"),
        med_success_blocks_on_route=("med_on_ok", "sum"),
        heavy_success_blocks_depot_only=("hev_dep_ok", "sum"),
        heavy_success_blocks_on_route=("hev_on_ok", "sum"),
    ).reset_index()
    for src, dst in [
        ("med_success_blocks_depot_only", "med_success_rate_depot_only_%"),
        ("med_success_blocks_on_route", "med_success_rate_on_route_%"),
        ("heavy_success_blocks_depot_only", "heavy_success_rate_depot_only_%"),
        ("heavy_success_blocks_on_route", "heavy_success_rate_on_route_%"),
    ]:
        summary[dst] = summary[src] / summary["total_blocks"] * 100.0
    return summary



def scenario_output_dir(base_dir: str | Path, installed_disp: Dict[str, int]) -> Path:
    base_dir = Path(base_dir)
    key = _hash_dict(installed_disp)
    out_dir = base_dir / "onroute_fcfs_scenarios" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir



def persist_scenario_artifacts(
    out_dir,
    assigned_events_df: pd.DataFrame,
    report_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    assignment_summary_df: pd.DataFrame,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 1. Assigned events
    # -------------------------------
    if assigned_events_df is not None and not assigned_events_df.empty:
        assigned_events_df.to_parquet(out_dir / "assigned_events.parquet", index=False)
        assigned_events_df.to_csv(out_dir / "assigned_events.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "assigned_events.csv", index=False)

    # -------------------------------
    # 2. Block report
    #    block_trips is a nested object column and cannot be safely
    #    written to parquet without schema normalization.
    # -------------------------------
    report_to_save = report_df.copy()

    if "block_trips" in report_to_save.columns:
        report_to_save["block_trips_json"] = report_to_save["block_trips"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else (
                None if pd.isna(x) else str(x)
            )
        )
        report_to_save = report_to_save.drop(columns=["block_trips"])

    report_to_save.to_parquet(out_dir / "block_report.parquet", index=False)
    report_to_save.to_csv(out_dir / "block_report.csv", index=False)

    # -------------------------------
    # 3. Profiles
    # -------------------------------
    if profiles_df is not None and not profiles_df.empty:
        profiles_to_save = profiles_df.copy()

        if "block_trips" in profiles_to_save.columns:
            profiles_to_save["block_trips_json"] = profiles_to_save["block_trips"].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else (
                    None if pd.isna(x) else str(x)
                )
            )
            profiles_to_save = profiles_to_save.drop(columns=["block_trips"])

        profiles_to_save.to_parquet(out_dir / "profiles.parquet", index=False)
        profiles_to_save.to_csv(out_dir / "profiles.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "profiles.csv", index=False)

    # -------------------------------
    # 4. Assignment summary
    # -------------------------------
    if assignment_summary_df is not None and not assignment_summary_df.empty:
        assignment_summary_df.to_parquet(out_dir / "assignment_summary.parquet", index=False)
        assignment_summary_df.to_csv(out_dir / "assignment_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "assignment_summary.csv", index=False)
