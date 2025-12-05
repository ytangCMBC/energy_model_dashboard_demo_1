import pandas as pd
import os
import re
import numpy as np
from collections import defaultdict

import json
import ast
import math
from typing import Any, Dict, List, Set, Tuple



t_sec  = np.array([0, 360, 600, 1800, 2400, 2900, 3800, 4200, 4800, 5200])
p_kw   = np.array([250, 250.2, 251.9, 258.7, 270, 230, 140, 120, 100, 80])
soc_pct = np.array([8, 21.9, 26.6, 49.7, 60, 70, 78, 84, 88, 90])

def time_to_sec(tstr):
    h, m, s = map(int, tstr.split(':'))
    return h*3600 + m*60 + s
def soc_to_time(s):
    s_clipped = np.clip(s, soc_pct.min(), soc_pct.max())
    return float(np.interp(s_clipped, soc_pct, t_sec))

def time_to_soc(t):
    t_clipped = np.clip(t, t_sec.min(), t_sec.max())
    return float(np.interp(t_clipped, t_sec, soc_pct))

def charge_session(start_soc_pct, duration_sec, max_soc_pct=90.0, n_steps=200):
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

def _code_in_matched(code, matched_codes):
    """
    Robust stop-code membership check, same idea as in simulate_block.
    """
    if code is None:
        return False
    if isinstance(code, (int, float)):
        return int(code) in matched_codes
    if isinstance(code, str) and code.strip().isdigit():
        return int(code.strip()) in matched_codes
    return False


def _infer_trip_distance_km(trip):
    """
    For now: if you don't have trip_distance_km in combine_df,
    approximate distance from (end_time - start_time) using an
    average speed (e.g., 30 km/h). You can replace this with
    true distances from your combine_df later.
    """
    if "trip_distance_km" in trip and trip["trip_distance_km"] is not None:
        return float(trip["trip_distance_km"])

    if trip.get("start_time") and trip.get("end_time"):
        t0 = time_to_sec(trip["start_time"])
        t1 = time_to_sec(trip["end_time"])
        dur_h = max(0.0, (t1 - t0) / 3600.0)
        avg_speed_kmh = 30.0  # tweak if you want
        return dur_h * avg_speed_kmh

    return 1.0


BATTERY_KWH = 376.0
MAX_SOC_FRAC = 0.90
MAX_ENERGY_KWH = BATTERY_KWH * MAX_SOC_FRAC 
INTERLINE_TYPES = {"interline"}

def build_block_profile_with_charging(block_trips,
                                      matched_codes,
                                      mode="medium"):
    """
    Build distance-based energy/SOC profile for a single block, including
    on-route charging at:
      - pull_out → first in_service (fixed 6 min)
      - in_service → in_service (layover - 60 s)
      - interline between in_service trips:
          * if only start is a charger: charge at START of interline
          * if end is a charger (end-only or both): charge at END of interline

    Returns a DataFrame with columns:
      - dist_km
      - soc_pct
      - net_energy_kwh   (INIT_KWH - cumulative_used + cumulative_charged)
      - phase            ('drive' or 'charge')
    """
    energy_key = "energy_medium_kwh" if mode == "medium" else "energy_heavy_kwh"

    # Copy + attach start_sec / end_sec
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

    # Index of first in-service
    in_idxs = [i for i, t in enumerate(trips) if t.get("type") == "in_service"]
    first_in_idx = in_idxs[0] if in_idxs else None

    # Battery state
    kwh = MAX_ENERGY_KWH
    total_used = 0.0
    total_charged = 0.0
    soc_pct_curr = MAX_SOC_FRAC * 100.0

    # Distance axis
    dist_cum_km = 0.0

    xs = []           # distance
    socs = []         # SOC (%)
    net_energy = []   # net kWh remaining relative to INIT_KWH
    phase = []   # 'start', 'drive', or 'charge'       
    used_hist = [] # cumulative used kWh
    charged_hist = []   # cumulative charged kWh    

    def record_point(dist, soc_pct, total_used, total_charged, tag):
        net_kwh = MAX_ENERGY_KWH - total_used + total_charged
        xs.append(dist)
        socs.append(soc_pct)
        net_energy.append(net_kwh)
        phase.append(tag)
        used_hist.append(total_used)
        charged_hist.append(total_charged)

    # Start point
    record_point(dist_cum_km, soc_pct_curr, total_used, total_charged, "start")

    # ---------- Main trip loop ----------
    for i, trip in enumerate(trips):
        ttype = trip.get("type")

        use = float(trip.get(energy_key, 0.0) or 0.0)
        dist_km = _infer_trip_distance_km(trip)

        dist_start = dist_cum_km
        dist_end = dist_cum_km + dist_km

        # ---------- Special handling: interline NIS between in_service trips ----------
        if ttype in INTERLINE_TYPES and 0 < i < len(trips) - 1:
            prev_trip = trips[i - 1]
            next_trip = trips[i + 1]

            if (
                prev_trip.get("type") == "in_service"
                and next_trip.get("type") == "in_service"
            ):
                inter_start_code = trip.get("start_stop_code")
                inter_end_code = trip.get("end_stop_code")

                has_start = _code_in_matched(inter_start_code, matched_codes)
                has_end = _code_in_matched(inter_end_code, matched_codes)

                # Only do special logic if at least one end is a charger
                if has_start or has_end:
                    # 1) start-only: charge at BEGINNING of interline
                    if has_start and not has_end:
                        duration_sec = 6 * 60
                        delta_e, _, _, _ = charge_session(soc_pct_curr, duration_sec)

                        kwh_before = kwh
                        kwh += delta_e
                        if kwh > MAX_ENERGY_KWH:
                            kwh = MAX_ENERGY_KWH
                        actual_delta = kwh - kwh_before
                        total_charged += actual_delta
                        soc_pct_curr = 100.0 * (kwh / BATTERY_KWH)

                        # charge point at dist_start
                        record_point(dist_start, soc_pct_curr,
                                     total_used, total_charged, "charge")

                    # 2) drive the interline
                    total_used += use
                    kwh -= use
                    kwh = min(MAX_ENERGY_KWH, kwh)
                    soc_pct_curr = 100.0 * (kwh / BATTERY_KWH)

                    record_point(dist_end, soc_pct_curr,
                                 total_used, total_charged, "drive")
                    dist_cum_km = dist_end

                    # 3) end-only OR both: charge at END of interline
                    if has_end:
                        duration_sec = 6 * 60
                        delta_e, _, _, _ = charge_session(soc_pct_curr, duration_sec)

                        kwh_before = kwh
                        kwh += delta_e
                        if kwh > MAX_ENERGY_KWH:
                            kwh = MAX_ENERGY_KWH
                        actual_delta = kwh - kwh_before
                        total_charged += actual_delta
                        soc_pct_curr = 100.0 * (kwh / BATTERY_KWH)

                        # charge point at dist_end (current dist_cum_km)
                        record_point(dist_cum_km, soc_pct_curr,
                                     total_used, total_charged, "charge")

                    # we've fully handled this interline trip; skip generic logic below
                    continue

        # ---------- Generic driving for all other trips (and interlines with no chargers) ----------
        total_used += use
        kwh -= use
        kwh = min(MAX_ENERGY_KWH, kwh)
        soc_pct_curr = 100.0 * (kwh / BATTERY_KWH)

        record_point(dist_end, soc_pct_curr, total_used, total_charged, "drive")
        dist_cum_km = dist_end

        # ---------- Pull-out → first in-service charging ----------
        if ttype == "pull_out" and first_in_idx is not None:
            next_trip = trips[first_in_idx]
            prev_end_code = trip.get("end_stop_code")
            next_start_code = next_trip.get("start_stop_code")

            eligible = (
                _code_in_matched(prev_end_code, matched_codes) or
                _code_in_matched(next_start_code, matched_codes)
            )
            if eligible:
                duration_sec = 6 * 60  # fixed 6 min
                if duration_sec > 0:
                    delta_e, _, _, _ = charge_session(soc_pct_curr, duration_sec)

                    kwh_before = kwh
                    kwh += delta_e
                    if kwh > MAX_ENERGY_KWH:
                        kwh = MAX_ENERGY_KWH
                    actual_delta = kwh - kwh_before
                    total_charged += actual_delta
                    soc_pct_curr = 100.0 * (kwh / BATTERY_KWH)

                    record_point(dist_cum_km, soc_pct_curr,
                                 total_used, total_charged, "charge")

            continue  # pull_out has no further layover logic

        # ---------- In-service charging: in_service → in_service ----------
        if ttype == "in_service" and i < len(trips) - 1:
            next_trip = trips[i + 1]

            if next_trip.get("type") == "pull_in":
                continue

            if next_trip.get("type") == "in_service":
                if trip["end_sec"] is not None and next_trip["start_sec"] is not None:
                    layover = next_trip["start_sec"] - trip["end_sec"]
                else:
                    layover = 0

                prev_end_code = trip.get("end_stop_code")
                next_start_code = next_trip.get("start_stop_code")

                eligible = (
                    _code_in_matched(prev_end_code, matched_codes) or
                    _code_in_matched(next_start_code, matched_codes)
                )

                if eligible:
                    duration_sec = max(0, layover - 60)  # minus 1 min prep
                    if duration_sec > 0:
                        delta_e, _, _, _ = charge_session(soc_pct_curr, duration_sec)

                        kwh_before = kwh
                        kwh += delta_e
                        if kwh > MAX_ENERGY_KWH:
                            kwh = MAX_ENERGY_KWH

                        actual_delta = kwh - kwh_before
                        total_charged += actual_delta
                        soc_pct_curr = 100.0 * (kwh / BATTERY_KWH)

                        record_point(dist_cum_km, soc_pct_curr,
                                     total_used, total_charged, "charge")

                continue

    profile = pd.DataFrame({
        "dist_km": xs,
        "soc_pct": socs,
        "net_energy_kwh": net_energy,
        "phase": phase,
        "cum_used_kwh": used_hist,
        "cum_charged_kwh": charged_hist,
    })
    return profile


