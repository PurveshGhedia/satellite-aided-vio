"""
ORB Matcher Benchmark on UAV-VisLoc Dataset
--------------------------------------------
Tests the ORB + RANSAC homography matcher against real drone-to-satellite
image pairs with GPS ground truth.

Metrics reported per image:
  - Match success / failure
  - Estimated pixel location on satellite map
  - Ground truth pixel location (from GPS + rasterio georeferencing)
  - Pixel error (Euclidean distance in satellite map pixels)
  - Geographic error (metres, approx)
  - Number of good matches found
  - Inlier ratio after RANSAC

Usage:
  python benchmark_orb.py --dataset /path/to/UAV_VisLoc_dataset --scene 03
  python benchmark_orb.py --dataset /path/to/UAV_VisLoc_dataset --scene all
"""

import os
import sys
import csv
import argparse
import time
import numpy as np
import cv2
import rasterio
from rasterio.warp import transform as rio_transform
from pathlib import Path
from rectify import rectify_drone_image


# ---------------------------------------------------------------------------
# Matching core (same logic as app_v1.py, isolated for benchmarking)
# ---------------------------------------------------------------------------

def orb_match(img_satellite, img_drone, n_features=2000, top_pct=0.15, ransac_thresh=5.0):
    """
    Match drone image against satellite patch using ORB + BF + RANSAC.

    Returns:
        dict with keys: success, est_px, M, n_matches, inlier_ratio, elapsed_ms
        or None if matching failed.
    """
    t0 = time.time()

    orb = cv2.ORB_create(nfeatures=n_features)
    kp_sat, des_sat = orb.detectAndCompute(img_satellite, None)
    kp_drone, des_drone = orb.detectAndCompute(img_drone, None)

    if des_sat is None or des_drone is None or len(kp_sat) < 4 or len(kp_drone) < 4:
        return {"success": False, "reason": "insufficient keypoints",
                "n_matches": 0, "inlier_ratio": 0.0,
                "elapsed_ms": (time.time() - t0) * 1000}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_drone, des_sat)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:max(4, int(len(matches) * top_pct))]

    if len(good) < 4:
        return {"success": False, "reason": "too few matches",
                "n_matches": len(good), "inlier_ratio": 0.0,
                "elapsed_ms": (time.time() - t0) * 1000}

    src_pts = np.float32(
        [kp_drone[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_sat[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    elapsed_ms = (time.time() - t0) * 1000

    if M is None:
        return {"success": False, "reason": "homography failed",
                "n_matches": len(good), "inlier_ratio": 0.0,
                "elapsed_ms": elapsed_ms}

    inlier_ratio = mask.ravel().sum() / len(mask) if mask is not None else 0.0

    h, w = img_drone.shape[:2]
    drone_center = np.array([[[w / 2, h / 2]]], dtype=np.float32)
    est_px = cv2.perspectiveTransform(drone_center, M)[0][0]

    return {
        "success": True,
        "est_px": est_px,           # (x, y) in satellite patch pixels
        "M": M,
        "n_matches": len(good),
        "inlier_ratio": float(inlier_ratio),
        "elapsed_ms": elapsed_ms
    }


# ---------------------------------------------------------------------------
# Geo utilities
# ---------------------------------------------------------------------------

def latlon_to_pixel(lat, lon, rasterio_dataset):
    """Convert WGS84 lat/lon to pixel (col, row) in the satellite TIF."""
    ds = rasterio_dataset
    if ds.crs and not ds.crs.is_geographic:
        xs, ys = rio_transform("EPSG:4326", ds.crs, [lon], [lat])
        col, row = ds.index(xs[0], ys[0])  # returns (row, col)
        return float(col), float(row)
    else:
        row, col = ds.index(lon, lat)
        return float(col), float(row)


def pixel_error(est_px, gt_px):
    """Euclidean pixel distance."""
    return float(np.linalg.norm(np.array(est_px) - np.array(gt_px)))


def pixel_to_metres(px_error, rasterio_dataset):
    """Approximate pixel error to metres using dataset resolution."""
    ds = rasterio_dataset
    res_x = abs(ds.transform[0])
    if ds.crs and ds.crs.is_geographic:
        metres_per_pixel = res_x * 111320.0
    else:
        metres_per_pixel = res_x
    return px_error * metres_per_pixel


# ---------------------------------------------------------------------------
# Satellite crop around estimated drone position
# ---------------------------------------------------------------------------

def crop_search_region(sat_dataset, gt_col, gt_row, search_radius_px=600):
    """
    Crop a square search region from the satellite TIF centred on the
    ground truth pixel. In a real system this would be centred on the
    VINS-Mono estimated position. Using GT here gives the matcher
    the best possible conditions -- an upper bound on performance.
    """
    half = search_radius_px
    c_off = max(0, int(gt_col - half))
    r_off = max(0, int(gt_row - half))
    width = min(search_radius_px * 2, sat_dataset.width - c_off)
    height = min(search_radius_px * 2, sat_dataset.height - r_off)

    window = rasterio.windows.Window(c_off, r_off, width, height)
    data = sat_dataset.read(window=window)

    if data.shape[0] >= 3:
        img = np.transpose(data[:3], (1, 2, 0))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = data[0].astype(np.uint8)

    # Ground truth position within the cropped patch
    gt_in_patch = (gt_col - c_off, gt_row - r_off)
    return img_gray, gt_in_patch, (c_off, r_off)


# ---------------------------------------------------------------------------
# Single scene benchmark
# ---------------------------------------------------------------------------

def benchmark_scene(scene_dir, max_images=None, search_radius_px=600, verbose=True, rectify=False, hfov_deg=84.0):
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name

    # Find satellite TIF
    tif_files = list(scene_dir.glob("*.tif"))
    if not tif_files:
        print(f"[{scene_id}] No .tif found, skipping.")
        return []
    sat_path = tif_files[0]

    # Find CSV
    csv_files = list(scene_dir.glob("*.csv"))
    if not csv_files:
        print(f"[{scene_id}] No CSV found, skipping.")
        return []
    csv_path = csv_files[0]

    # Load ground truth (now also includes pitch/roll for rectification)
    gt_lookup = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_lookup[row["filename"]] = {
                "lat":    float(row["lat"]),
                "lon":    float(row["lon"]),
                "height": float(row["height"]),
                "pitch":  float(row["Omega"]),
                "roll":   float(row["Kappa"]),
            }

    drone_dir = scene_dir / "drone"
    drone_images = sorted(drone_dir.glob("*.JPG"))
    if max_images:
        drone_images = drone_images[:max_images]

    results = []

    with rasterio.open(sat_path) as sat_ds:
        for img_path in drone_images:
            fname = img_path.name
            if fname not in gt_lookup:
                continue

            gt = gt_lookup[fname]

            # Ground truth pixel on satellite
            try:
                gt_col, gt_row = latlon_to_pixel(gt["lat"], gt["lon"], sat_ds)
            except Exception as e:
                if verbose:
                    print(f"  [{fname}] GT pixel conversion failed: {e}")
                continue

            # Crop search region centred on GT (simulates VINS estimate = GT for now)
            sat_patch, gt_in_patch, patch_origin = crop_search_region(
                sat_ds, gt_col, gt_row, search_radius_px)

            # Load drone image
            drone_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if drone_img is None:
                continue

            # Resize drone image to a reasonable size (satellite patches are large)
            drone_img = cv2.resize(drone_img, (512, 512))

            # Optionally rectify to nadir view using pitch/roll
            if rectify:
                pitch = gt_lookup[fname]["pitch"]
                roll = gt_lookup[fname]["roll"]
                drone_img, _ = rectify_drone_image(
                    drone_img, pitch, roll, hfov_deg)

            # Run matcher
            match_result = orb_match(sat_patch, drone_img)

            row_result = {
                "scene":        scene_id,
                "filename":     fname,
                "gt_lat":       gt["lat"],
                "gt_lon":       gt["lon"],
                "height_m":     gt["height"],
                "gt_col":       gt_col,
                "gt_row":       gt_row,
                "success":      match_result["success"],
                "n_matches":    match_result["n_matches"],
                "inlier_ratio": match_result["inlier_ratio"],
                "elapsed_ms":   match_result["elapsed_ms"],
                "px_error":     None,
                "geo_error_m":  None,
                "reason":       match_result.get("reason", ""),
            }

            if match_result["success"]:
                # est_px is in patch coordinates; convert back to global satellite coords
                est_in_patch = match_result["est_px"]
                est_col = est_in_patch[0] + patch_origin[0]
                est_row = est_in_patch[1] + patch_origin[1]

                px_err = pixel_error((est_col, est_row), (gt_col, gt_row))
                geo_err = pixel_to_metres(px_err, sat_ds)

                row_result["px_error"] = px_err
                row_result["geo_error_m"] = geo_err

                if verbose:
                    print(f"  [{fname}] OK | px_err={px_err:.1f}px | "
                          f"geo_err={geo_err:.1f}m | "
                          f"matches={match_result['n_matches']} | "
                          f"inliers={match_result['inlier_ratio']:.2f} | "
                          f"{match_result['elapsed_ms']:.0f}ms")
            else:
                if verbose:
                    print(f"  [{fname}] FAIL | reason={match_result.get('reason')} | "
                          f"matches={match_result['n_matches']} | "
                          f"{match_result['elapsed_ms']:.0f}ms")

            results.append(row_result)

    return results


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_summary(results):
    if not results:
        print("No results.")
        return

    total = len(results)
    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    rate = len(success) / total * 100

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total images tested : {total}")
    print(f"Successful matches  : {len(success)} ({rate:.1f}%)")
    print(f"Failed matches      : {len(failed)} ({100-rate:.1f}%)")

    if success:
        errors = [r["geo_error_m"] for r in success]
        px_err = [r["px_error"] for r in success]
        timing = [r["elapsed_ms"] for r in results]

        print(f"\n--- Localisation Error (successful matches only) ---")
        print(f"  Mean  : {np.mean(errors):.1f} m  ({np.mean(px_err):.1f} px)")
        print(
            f"  Median: {np.median(errors):.1f} m  ({np.median(px_err):.1f} px)")
        print(f"  Std   : {np.std(errors):.1f} m")
        print(f"  Min   : {np.min(errors):.1f} m")
        print(f"  Max   : {np.max(errors):.1f} m")

        print(f"\n--- Match Quality ---")
        inliers = [r["inlier_ratio"] for r in success]
        print(f"  Mean inlier ratio : {np.mean(inliers):.3f}")
        print(
            f"  Mean match count  : {np.mean([r['n_matches'] for r in success]):.0f}")

        print(f"\n--- Timing ---")
        print(f"  Mean per image : {np.mean(timing):.0f} ms")
        print(f"  Max per image  : {np.max(timing):.0f} ms")

    if failed:
        reasons = {}
        for r in failed:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print(f"\n--- Failure Reasons ---")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print("="*60)


# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------

def save_results(results, output_path):
    if not results:
        return
    keys = results[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ORB matcher on UAV-VisLoc")
    parser.add_argument("--dataset",      required=True,
                        help="Path to UAV_VisLoc_dataset/")
    parser.add_argument("--scene",        default="03",
                        help="Scene ID (e.g. 03) or 'all'")
    parser.add_argument("--max_images",   type=int,
                        default=None, help="Limit images per scene")
    parser.add_argument("--search_radius", type=int,
                        default=600,  help="Satellite crop radius in px")
    parser.add_argument("--output",       default="orb_benchmark_results.csv")
    parser.add_argument("--quiet",        action="store_true")
    parser.add_argument("--rectify",      action="store_true",
                        help="Apply pitch/roll rectification before matching")
    parser.add_argument("--hfov",         type=float,
                        default=84.0, help="Camera horizontal FOV in degrees")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    all_results = []

    if args.scene == "all":
        scenes = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    else:
        scenes = [dataset_root / args.scene]

    for scene_dir in scenes:
        print(f"\n--- Scene: {scene_dir.name} ---")
        results = benchmark_scene(
            scene_dir,
            max_images=args.max_images,
            search_radius_px=args.search_radius,
            verbose=not args.quiet,
            rectify=args.rectify,
            hfov_deg=args.hfov,
        )
        all_results.extend(results)

    print_summary(all_results)
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
