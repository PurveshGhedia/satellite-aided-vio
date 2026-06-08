"""
LightGlue + SuperPoint Benchmark on UAV-VisLoc Dataset
-------------------------------------------------------
Tests SuperPoint (learned keypoints) + LightGlue (transformer matcher)
against real drone-to-satellite image pairs with GPS ground truth.

Key differences from SIFT:
  - SuperPoint detects keypoints using a CNN trained on homographic pairs
    -- finds more repeatable keypoints across viewpoint/lighting changes
  - LightGlue matches using a transformer that reasons about all keypoints
    globally rather than matching each descriptor independently
  - No ratio test needed -- LightGlue outputs a confidence score per match
    and filters internally

Usage:
  python benchmark_lightglue.py --dataset /path/to/UAV_VisLoc_dataset --scene 03
  python benchmark_lightglue.py --dataset /path/to/UAV_VisLoc_dataset --scene all
"""

import os
import sys
import csv
import argparse
import time
import numpy as np
import cv2
import torch
import rasterio
from rasterio.warp import transform as rio_transform
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from rectify import rectify_drone_image

# ---------------------------------------------------------------------------
# Load models once at module level (avoid reloading per image)
# ---------------------------------------------------------------------------

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[LightGlue] Using device: {DEVICE}")

extractor = SuperPoint(max_num_keypoints=2048).eval().to(DEVICE)
matcher = LightGlue(features="superpoint").eval().to(DEVICE)


# ---------------------------------------------------------------------------
# Matching core — SuperPoint + LightGlue + RANSAC
# ---------------------------------------------------------------------------

def to_tensor(img_gray):
    """Convert grayscale numpy image to normalised float tensor [1,1,H,W]."""
    t = torch.from_numpy(img_gray).float() / 255.0
    return t.unsqueeze(0).unsqueeze(0).to(DEVICE)


def lightglue_match(img_satellite, img_drone, ransac_thresh=5.0):
    """
    Match drone image against satellite patch using SuperPoint + LightGlue.

    SuperPoint extracts keypoints and descriptors using a CNN.
    LightGlue matches them using a transformer -- it reasons about the
    global context of all keypoints simultaneously and outputs only
    high-confidence matches, so no ratio test is needed.

    RANSAC is still applied afterward to estimate the homography robustly
    and to get an inlier ratio for comparison with classical methods.

    Args:
        img_satellite:  grayscale satellite patch (numpy uint8)
        img_drone:      grayscale drone image (numpy uint8)
        ransac_thresh:  RANSAC reprojection threshold in pixels

    Returns:
        dict with keys: success, est_px, M, n_matches, inlier_ratio, elapsed_ms
    """
    t0 = time.time()

    with torch.no_grad():
        feats_sat = extractor.extract(to_tensor(img_satellite))
        feats_drone = extractor.extract(to_tensor(img_drone))
        matches_out = matcher({"image0": feats_sat, "image1": feats_drone})

    # rbd removes the batch dimension
    feats_sat,   feats_drone, matches_out = rbd(
        feats_sat), rbd(feats_drone), rbd(matches_out)

    # (N, 2) matched keypoint indices
    match_indices = matches_out["matches"]
    n_matches = match_indices.shape[0]
    elapsed_ms = (time.time() - t0) * 1000

    if n_matches < 4:
        return {"success": False, "reason": "too few matches from LightGlue",
                "n_matches": n_matches, "inlier_ratio": 0.0,
                "elapsed_ms": elapsed_ms}

    kp_sat = feats_sat["keypoints"][match_indices[:, 0]].cpu().numpy()
    kp_drone = feats_drone["keypoints"][match_indices[:, 1]].cpu().numpy()

    src_pts = kp_drone.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = kp_sat.reshape(-1, 1, 2).astype(np.float32)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    elapsed_ms = (time.time() - t0) * 1000

    if M is None:
        return {"success": False, "reason": "homography failed",
                "n_matches": n_matches, "inlier_ratio": 0.0,
                "elapsed_ms": elapsed_ms}

    inlier_ratio = mask.ravel().sum() / len(mask) if mask is not None else 0.0

    h, w = img_drone.shape[:2]
    drone_center = np.array([[[w / 2, h / 2]]], dtype=np.float32)
    est_px = cv2.perspectiveTransform(drone_center, M)[0][0]

    return {
        "success":      True,
        "est_px":       est_px,
        "M":            M,
        "n_matches":    n_matches,
        "inlier_ratio": float(inlier_ratio),
        "elapsed_ms":   elapsed_ms,
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

def benchmark_scene(scene_dir, max_images=None, search_radius_px=600, verbose=True,
                    rectify=False, hfov_deg=84.0,
                    min_inliers=15, min_inlier_ratio=0.20):
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
            match_result = lightglue_match(sat_patch, drone_img)

            # Confidence gate: reject matches that don't meet minimum quality thresholds.
            # A rejected correction is always safer than a wrong correction in a nav system.
            inlier_count = int(match_result["n_matches"] * match_result["inlier_ratio"]) \
                if match_result["success"] else 0
            gate_passed = (
                match_result["success"] and
                inlier_count >= min_inliers and
                match_result["inlier_ratio"] >= min_inlier_ratio
            )

            row_result = {
                "scene":        scene_id,
                "filename":     fname,
                "gt_lat":       gt["lat"],
                "gt_lon":       gt["lon"],
                "height_m":     gt["height"],
                "gt_col":       gt_col,
                "gt_row":       gt_row,
                "success":      match_result["success"],
                "gate_passed":  gate_passed,
                "inlier_count": inlier_count,
                "n_matches":    match_result["n_matches"],
                "inlier_ratio": match_result["inlier_ratio"],
                "elapsed_ms":   match_result["elapsed_ms"],
                "px_error":     None,
                "geo_error_m":  None,
                "reason":       match_result.get("reason", ""),
            }

            if match_result["success"]:
                est_in_patch = match_result["est_px"]
                est_col = est_in_patch[0] + patch_origin[0]
                est_row = est_in_patch[1] + patch_origin[1]

                px_err = pixel_error((est_col, est_row), (gt_col, gt_row))
                geo_err = pixel_to_metres(px_err, sat_ds)

                row_result["px_error"] = px_err
                row_result["geo_error_m"] = geo_err

                gate_str = "ACCEPTED" if gate_passed else f"GATED OUT (inliers={inlier_count}, ratio={match_result['inlier_ratio']:.2f})"
                if verbose:
                    print(f"  [{fname}] OK | px_err={px_err:.1f}px | "
                          f"geo_err={geo_err:.1f}m | "
                          f"matches={match_result['n_matches']} | "
                          f"inliers={inlier_count} | "
                          f"ratio={match_result['inlier_ratio']:.2f} | "
                          f"{gate_str} | "
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

def print_summary(results, min_inliers=15, min_inlier_ratio=0.20):
    if not results:
        print("No results.")
        return

    total = len(results)
    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    gated = [r for r in results if r.get(
        "gate_passed") and r["geo_error_m"] is not None]
    rejected = [r for r in success if not r.get("gate_passed")]
    rate = len(success) / total * 100

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total images tested : {total}")
    print(f"Matcher succeeded   : {len(success)} ({rate:.1f}%)")
    print(f"Matcher failed      : {len(failed)} ({100-rate:.1f}%)")
    print(f"Gate accepted       : {len(gated)} ({len(gated)/total*100:.1f}%)")
    print(
        f"Gate rejected       : {len(rejected)} ({len(rejected)/total*100:.1f}%)")
    print(
        f"[Gate thresholds: min_inliers={min_inliers}, min_inlier_ratio={min_inlier_ratio}]")

    if success:
        errors_all = [r["geo_error_m"] for r in success]
        print(f"\n--- Localisation Error (all successful matches) ---")
        print(f"  Mean  : {np.mean(errors_all):.1f} m")
        print(f"  Median: {np.median(errors_all):.1f} m")
        print(f"  Std   : {np.std(errors_all):.1f} m")
        print(f"  Min   : {np.min(errors_all):.1f} m")
        print(f"  Max   : {np.max(errors_all):.1f} m")

    if gated:
        errors_gated = [r["geo_error_m"] for r in gated]
        print(
            f"\n--- Localisation Error (gate-accepted only, {len(gated)} images) ---")
        print(f"  Mean  : {np.mean(errors_gated):.1f} m")
        print(f"  Median: {np.median(errors_gated):.1f} m")
        print(f"  Std   : {np.std(errors_gated):.1f} m")
        print(f"  Min   : {np.min(errors_gated):.1f} m")
        print(f"  Max   : {np.max(errors_gated):.1f} m")

        print(f"\n--- Rejected matches (would have been bad corrections) ---")
        for r in rejected:
            print(f"  {r['filename']} | geo_err={r['geo_error_m']:.1f}m | "
                  f"inliers={r['inlier_count']} | ratio={r['inlier_ratio']:.2f}")

    if success:
        print(f"\n--- Match Quality ---")
        print(
            f"  Mean inlier ratio : {np.mean([r['inlier_ratio'] for r in success]):.3f}")
        print(
            f"  Mean inlier count : {np.mean([r['inlier_count'] for r in success]):.0f}")
        print(
            f"  Mean match count  : {np.mean([r['n_matches'] for r in success]):.0f}")

        timing = [r["elapsed_ms"] for r in results]
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
        description="Benchmark LightGlue matcher on UAV-VisLoc")
    parser.add_argument("--dataset",           required=True)
    parser.add_argument("--scene",             default="03")
    parser.add_argument("--max_images",        type=int, default=None)
    parser.add_argument("--search_radius",     type=int, default=600)
    parser.add_argument(
        "--output",            default="lightglue_benchmark_results.csv")
    parser.add_argument("--quiet",             action="store_true")
    parser.add_argument("--rectify",           action="store_true")
    parser.add_argument("--hfov",              type=float, default=84.0)
    parser.add_argument("--min_inliers",       type=int,   default=15,
                        help="Minimum inlier count to accept a correction (default: 15)")
    parser.add_argument("--min_inlier_ratio",  type=float, default=0.20,
                        help="Minimum inlier ratio to accept a correction (default: 0.20)")
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
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
        )
        all_results.extend(results)

    print_summary(all_results, args.min_inliers, args.min_inlier_ratio)
    save_results(all_results, args.output)
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
