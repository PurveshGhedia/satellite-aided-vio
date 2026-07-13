#!/usr/bin/env python3
"""
estimate_heading_offset.py
----------------------------
Estimates the rotation (heading) offset between VINS-Mono's local ENU
frame and true North/East, by comparing recorded VIO track points
against ground truth at matching timestamps.

This is the 2D equivalent of what `evo`'s trajectory alignment does
for the full SE3 ATE computation -- solving for the best-fit rotation
rather than assuming the two frames are already aligned.

Usage:
    python3 estimate_heading_offset.py \
        --vio_csv vio_track.csv \
        --pos_file bell412_dataset6_frl.pos \
        --origin_lat 45.325253240071 \
        --origin_lon -75.664506212371
"""
import argparse
import csv
import math
import re
from datetime import datetime, timezone, timedelta

import numpy as np

GPS_UTC_LEAP_SECONDS = 18


def gpst_to_unix(date_str, time_str, leap_seconds=GPS_UTC_LEAP_SECONDS):
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S.%f")
    dt = dt.replace(tzinfo=timezone.utc)
    return (dt - timedelta(seconds=leap_seconds)).timestamp()


def load_pos_file(pos_path):
    rows = []
    with open(pos_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 6:
                continue
            try:
                t = gpst_to_unix(parts[0], parts[1])
                lat, lon = float(parts[2]), float(parts[3])
                rows.append((t, lat, lon))
            except (ValueError, IndexError):
                continue
    rows.sort()
    return rows


def nearest(t, rows):
    lo, hi = 0, len(rows) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and abs(rows[lo-1][0] - t) < abs(rows[lo][0] - t):
        lo -= 1
    return rows[lo]


def latlon_to_local_xy(lat, lon, origin_lat, origin_lon):
    """Convert lat/lon to local ENU meters relative to origin (inverse of _odom_to_latlon)."""
    x = (lon - origin_lon) * 111320.0 * math.cos(math.radians(origin_lat))
    y = (lat - origin_lat) * 111320.0
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vio_csv", required=True)
    p.add_argument("--pos_file", required=True)
    p.add_argument("--origin_lat", type=float, required=True)
    p.add_argument("--origin_lon", type=float, required=True)
    p.add_argument("--max_dt", type=float, default=0.15,
                   help="Max time gap (s) to accept a GT match")
    args = p.parse_args()

    gt_rows = load_pos_file(args.pos_file)
    print(f"Loaded {len(gt_rows)} GT rows")

    vio_xy = []
    gt_xy = []
    with open(args.vio_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp"])
            vio_lat, vio_lon = float(row["lat"]), float(row["lon"])

            gt_t, gt_lat, gt_lon = nearest(ts, gt_rows)
            if abs(gt_t - ts) > args.max_dt:
                continue

            # Convert VIO's own lat/lon back to local x,y (undo the naive
            # conversion) using the SAME origin/formula, so we recover the
            # raw VINS x,y that was fed into that formula in the first place.
            vx, vy = latlon_to_local_xy(
                vio_lat, vio_lon, args.origin_lat, args.origin_lon)
            gx, gy = latlon_to_local_xy(
                gt_lat, gt_lon, args.origin_lat, args.origin_lon)

            vio_xy.append((vx, vy))
            gt_xy.append((gx, gy))

    print(f"Matched {len(vio_xy)} VIO/GT point pairs (max_dt={args.max_dt}s)")

    V = np.array(vio_xy)
    G = np.array(gt_xy)

    # 2D Procrustes / Kabsch: solve for rotation R minimizing ||R@V.T - G.T||
    # (translation is already ~0 at t=0 since both are relative to the same origin,
    #  but we still de-mean to be safe/correct in general.)
    V_mean = V.mean(axis=0)
    G_mean = G.mean(axis=0)
    Vc = V - V_mean
    Gc = G - G_mean

    H = Vc.T @ Gc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    heading_offset_rad = math.atan2(R[1, 0], R[0, 0])
    heading_offset_deg = math.degrees(heading_offset_rad)

    # Full correction is: corrected = R @ raw + translation
    # where translation = G_mean - R @ V_mean (recovers the origin/offset
    # mismatch on top of the pure rotation).
    translation = G_mean - R @ V_mean

    # Residual error before and after correction
    before_err = np.linalg.norm(V - G, axis=1)
    V_corrected = (R @ V.T).T + translation
    after_err = np.linalg.norm(V_corrected - G, axis=1)

    print(f"\nEstimated heading offset: {heading_offset_deg:.2f} degrees")
    print(f"Estimated translation offset (x, y) in metres: "
          f"({translation[0]:.2f}, {translation[1]:.2f})")
    print(f"Mean position error BEFORE correction: {before_err.mean():.1f} m")
    print(
        f"Mean position error AFTER  correction (rotation + translation): {after_err.mean():.1f} m")
    print(
        f"Median position error AFTER correction: {np.median(after_err):.1f} m")


if __name__ == "__main__":
    main()
