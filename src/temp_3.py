import numpy as np
import pandas as pd


t_sec = np.array([0, 360, 600, 1800, 2400, 2900, 3800, 4200, 4800, 5200])
p_kw = np.array([250, 250.2, 251.9, 258.7, 270, 230, 140, 120, 100, 80])
soc_pct = np.array([8, 21.9, 26.6, 49.7, 60, 70, 78, 84, 88, 90])


def time_to_sec(tstr):
    h, m, s = map(int, tstr.split(":"))
    return h * 3600 + m * 60 + s


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
    if code is None:
        return False
    if isinstance(code, int):
        return code in matched_codes
    if isinstance(code, float):
        if code != code:  # NaN
            return False
        return int(code) in matched_codes
    if isinstance(code, str):
        s = code.strip()
        if s.isdigit():
            return int(s) in matched_codes
    return False


def _normalize_stop_code(code):
    if code is None:
        return None
    if isinstance(code, int):
        return code
    if isinstance(code, float):
        if code != code:  # NaN
            return None
        return int(code)
    if isinstance(code, str):
        s = code.strip()
        return int(s) if s.isdigit() else None
    return None


def _choose_charge_stop(prev_end_code, next_start_code, matched_codes):
    p = _normalize_stop_code(prev_end_code)
    n = _normalize_stop_code(next_start_code)
    if p is not None and p in matched_codes:
        return p
    if n is not None and n in matched_codes:
        return n
    return None


def _infer_trip_distance_km(trip):
    if "trip_distance_km" in trip and trip["trip_distance_km"] is not None:
        return float(trip["trip_distance_km"])

    if trip.get("start_time") and trip.get("end_time"):
        t0 = time_to_sec(trip["start_time"])
        t1 = time_to_sec(trip["end_time"])
        dur_h = max(0.0, (t1 - t0) / 3600.0)
        avg_speed_kmh = 30.0
        return dur_h * avg_speed_kmh

    return 1.0


BATTERY_KWH = 564.0
MAX_SOC_FRAC = 0.90
MAX_ENERGY_KWH = BATTERY_KWH * MAX_SOC_FRAC
INTERLINE_TYPES = {"interline"}

MIN_CHARGE_SEC = 180
CHARGE_TRIGGER_SOC_PCT = 70.0


def build_block_profile_with_charging(
    block_trips,
    matched_codes,
    mode="medium",
    layover_assume_min=8,
    prep_time_min=3,
    charge_trigger_soc_pct=CHARGE_TRIGGER_SOC_PCT,
):
    """
    Build distance-based energy/SOC profile for a single block, including
    on-route charging metadata required by the dashboard.

    Returns DataFrame with:
      - dist_km
      - soc_pct
      - net_energy_kwh
      - phase
      - cum_used_kwh
      - cum_charged_kwh
      - stop_code
      - charge_kwh
      - charge_duration_sec
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
            ]
        )

    energy_key = "energy_medium_kwh" if mode == "medium" else "energy_heavy_kwh"
    buffer_sec = int(prep_time_min * 60)

    def clamp_kwh(x):
        return min(float(MAX_ENERGY_KWH), float(x))

    def soc_from_kwh(kwh_val):
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

    xs = []
    socs = []
    net_energy = []
    phase = []
    used_hist = []
    charged_hist = []
    stop_code_hist = []
    charge_kwh_hist = []
    charge_dur_s_hist = []

    def record_point(
        dist,
        soc_pct_val,
        total_used_val,
        total_charged_val,
        tag,
        stop_code=None,
        charge_kwh=0.0,
        charge_duration_sec=0.0,
    ):
        net_kwh = MAX_ENERGY_KWH - total_used_val + total_charged_val

        xs.append(float(dist))
        socs.append(float(soc_pct_val))
        net_energy.append(float(net_kwh))
        phase.append(str(tag))
        used_hist.append(float(total_used_val))
        charged_hist.append(float(total_charged_val))

        stop_code_hist.append(str(stop_code).strip() if stop_code is not None else None)
        charge_kwh_hist.append(float(charge_kwh or 0.0))
        charge_dur_s_hist.append(float(charge_duration_sec or 0.0))

    def apply_charge_here(duration_sec, chosen_stop_code):
        nonlocal kwh, total_charged, soc_pct_curr

        if duration_sec < MIN_CHARGE_SEC:
            return False
        if soc_pct_curr >= charge_trigger_soc_pct:
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
            total_used,
            total_charged,
            "charge",
            stop_code=chosen_stop_code,
            charge_kwh=actual_delta,
            charge_duration_sec=duration_sec,
        )
        return True

    record_point(dist_cum_km, soc_pct_curr, total_used, total_charged, "start")

    for i, trip in enumerate(trips):
        ttype = trip.get("type")
        use = float(trip.get(energy_key, 0.0) or 0.0)
        dist_km = _infer_trip_distance_km(trip)

        dist_start = dist_cum_km
        dist_end = dist_cum_km + dist_km

        # -------- interline special handling --------
        if ttype in INTERLINE_TYPES and 0 < i < len(trips) - 1:
            prev_trip = trips[i - 1]
            next_trip = trips[i + 1]

            if prev_trip.get("type") == "in_service" and next_trip.get("type") == "in_service":
                inter_start_code = trip.get("start_stop_code")
                inter_end_code = trip.get("end_stop_code")

                has_start = _code_in_matched(inter_start_code, matched_codes)
                has_end = _code_in_matched(inter_end_code, matched_codes)

                if has_start or has_end:
                    # charge at start only when start is charger and end is not
                    if has_start and not has_end:
                        duration_sec = max(0, layover_assume_min * 60 - buffer_sec)
                        apply_charge_here(duration_sec, inter_start_code)

                    # drive interline
                    total_used += use
                    kwh = clamp_kwh(kwh - use)
                    soc_pct_curr = soc_from_kwh(kwh)

                    record_point(
                        dist_end,
                        soc_pct_curr,
                        total_used,
                        total_charged,
                        "drive",
                    )
                    dist_cum_km = dist_end

                    # charge at end if end exists
                    if has_end:
                        duration_sec = max(0, layover_assume_min * 60 - buffer_sec)
                        apply_charge_here(duration_sec, inter_end_code)

                    continue

        # -------- generic driving --------
        total_used += use
        kwh = clamp_kwh(kwh - use)
        soc_pct_curr = soc_from_kwh(kwh)

        record_point(
            dist_end,
            soc_pct_curr,
            total_used,
            total_charged,
            "drive",
        )
        dist_cum_km = dist_end

        # -------- pull_out -> first in_service --------
        if ttype == "pull_out" and first_in_idx is not None:
            next_trip = trips[first_in_idx]
            prev_end_code = trip.get("end_stop_code")
            next_start_code = next_trip.get("start_stop_code")

            eligible = (
                _code_in_matched(prev_end_code, matched_codes)
                or _code_in_matched(next_start_code, matched_codes)
            )

            if eligible:
                chosen_stop = _choose_charge_stop(prev_end_code, next_start_code, matched_codes)
                duration_sec = max(0, layover_assume_min * 60 - buffer_sec)
                apply_charge_here(duration_sec, chosen_stop)

            continue

        # -------- in_service -> in_service layover --------
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
                    _code_in_matched(prev_end_code, matched_codes)
                    or _code_in_matched(next_start_code, matched_codes)
                )

                if eligible:
                    chosen_stop = _choose_charge_stop(prev_end_code, next_start_code, matched_codes)
                    duration_sec = max(0, layover - buffer_sec)
                    apply_charge_here(duration_sec, chosen_stop)

                continue

    profile = pd.DataFrame(
        {
            "dist_km": xs,
            "soc_pct": socs,
            "net_energy_kwh": net_energy,
            "phase": phase,
            "cum_used_kwh": used_hist,
            "cum_charged_kwh": charged_hist,
            "stop_code": stop_code_hist,
            "charge_kwh": charge_kwh_hist,
            "charge_duration_sec": charge_dur_s_hist,
        }
    )
    return profile