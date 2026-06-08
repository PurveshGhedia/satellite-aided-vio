"""
Visualize ORB matches between drone and satellite patch.
--------------------------------------------------------
Loads results from the benchmark CSV, picks a few images,
and shows the keypoint matches side by side in a popup window.

Usage:
    python visualize_matches.py --dataset /path/to/UAV_VisLoc_dataset \
                                --results orb_benchmark_results.csv \
                                --scene 03 \
                                --n 5
"""

import argparse
import csv
import cv2
import numpy as np
import rasterio
from pathlib import Path


# ---------------------------------------------------------------------------
# Reuse the same crop logic as the benchmark
# ---------------------------------------------------------------------------

def crop_search_region(sat_dataset, gt_col, gt_row, search_radius_px=600):
    half = search_radius_px
    c_off = max(0, int(gt_col - half))
    r_off = max(0, int(gt_row - half))
    width = min(search_radius_px * 2, sat_dataset.width - c_off)
    height = min(search_radius_px * 2, sat_dataset.height - r_off)

    window = rasterio.windows.Window(c_off, r_off, width, height)
    data = sat_dataset.read(window=window)

    if data.shape[0] >= 3:
        img = np.transpose(data[:3], (1, 2, 0))
        img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = data[0].astype(np.uint8)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    gt_in_patch = (gt_col - c_off, gt_row - r_off)
    return img_gray, img_bgr, gt_in_patch, (c_off, r_off)


def latlon_to_pixel(lat, lon, ds):
    if ds.crs and not ds.crs.is_geographic:
        from rasterio.warp import transform as rio_transform
        xs, ys = rio_transform("EPSG:4326", ds.crs, [lon], [lat])
        col, row = ds.index(xs[0], ys[0])
        return float(col), float(row)
    else:
        row, col = ds.index(lon, lat)
        return float(col), float(row)


# ---------------------------------------------------------------------------
# Draw matches with estimated and GT position marked
# ---------------------------------------------------------------------------

def visualize_one(sat_bgr, sat_gray, drone_bgr, drone_path,
                  gt_in_patch, geo_error_m, px_error,
                  n_features=2000, top_pct=0.15):

    drone_gray = cv2.cvtColor(drone_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp_sat,   des_sat = orb.detectAndCompute(sat_gray,   None)
    kp_drone, des_drone = orb.detectAndCompute(drone_gray, None)

    if des_sat is None or des_drone is None:
        print(f"  No descriptors found for {drone_path.name}")
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_drone, des_sat)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:max(4, int(len(matches) * top_pct))]

    if len(good) < 4:
        print(f"  Too few matches for {drone_path.name}")
        return

    src_pts = np.float32(
        [kp_drone[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_sat[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Colour matches: green = inlier, red = outlier
    matchesMask = mask.ravel().tolist() if mask is not None else [
        0] * len(good)
    draw_params = dict(
        matchColor=(0, 255, 0),   # inliers: green
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=cv2.DrawMatchesFlags_DEFAULT
    )

    # Draw inlier matches using colour images
    vis = cv2.drawMatches(drone_bgr, kp_drone, sat_bgr, kp_sat,
                          good, None, **draw_params)

    # Also draw outliers in red
    outlier_mask = [1 - m for m in matchesMask]
    draw_outliers = dict(matchColor=(0, 0, 255),
                         singlePointColor=None,
                         matchesMask=outlier_mask,
                         flags=cv2.DrawMatchesFlags_DEFAULT)
    vis = cv2.drawMatches(drone_bgr, kp_drone, sat_bgr, kp_sat,
                          good, vis, **draw_outliers)

    # Mark estimated position on satellite patch (right side of vis)
    drone_w = drone_bgr.shape[1]
    if M is not None:
        h, w = drone_bgr.shape[:2]
        center = np.array([[[w / 2, h / 2]]], dtype=np.float32)
        est_px = cv2.perspectiveTransform(center, M)[0][0]
        est_on_vis = (int(est_px[0]) + drone_w, int(est_px[1]))
        cv2.drawMarker(vis, est_on_vis, (0, 255, 255),
                       cv2.MARKER_CROSS, 30, 3)
        cv2.putText(vis, "EST", (est_on_vis[0] + 8, est_on_vis[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Mark ground truth position on satellite patch
    gt_on_vis = (int(gt_in_patch[0]) + drone_w, int(gt_in_patch[1]))
    cv2.drawMarker(vis, gt_on_vis, (0, 255, 0),
                   cv2.MARKER_STAR, 30, 3)
    cv2.putText(vis, "GT", (gt_on_vis[0] + 8, gt_on_vis[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Info overlay
    inlier_count = sum(matchesMask)
    inlier_ratio = inlier_count / len(good) if good else 0
    info = [
        f"Image : {drone_path.name}",
        f"Matches: {len(good)}  Inliers: {inlier_count}  Ratio: {inlier_ratio:.2f}",
        f"Pixel error : {px_error:.1f} px",
        f"Geo error   : {geo_error_m:.1f} m",
        f"Green lines = inliers | Red lines = outliers",
        f"Yellow cross = estimated | Green star = ground truth",
        f"Press any key for next image, Q to quit",
    ]
    y = 20
    for line in info:
        cv2.putText(vis, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        y += 22

    # Resize to fit screen if too large
    screen_w = 1800
    h_vis, w_vis = vis.shape[:2]
    if w_vis > screen_w:
        scale = screen_w / w_vis
        vis = cv2.resize(vis, (int(w_vis * scale), int(h_vis * scale)))

    cv2.imshow("ORB Match Visualizer", vis)
    key = cv2.waitKey(0)
    return key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize ORB matches on UAV-VisLoc")
    parser.add_argument("--dataset",       required=True,
                        help="Path to UAV_VisLoc_dataset/")
    parser.add_argument("--results",       required=True,
                        help="Path to orb_benchmark_results.csv")
    parser.add_argument("--scene",         default="03",
                        help="Scene ID to visualize")
    parser.add_argument("--n",             type=int,
                        default=5, help="Number of images to show")
    parser.add_argument("--search_radius", type=int, default=600)
    parser.add_argument("--mode",          default="random",
                        choices=["random", "best", "worst"],
                        help="Which images to pick: random, best (lowest error), worst (highest error)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    scene_dir = dataset_root / args.scene

    # Load results CSV and filter to this scene
    rows = []
    with open(args.results, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["scene"] == args.scene and row["success"] == "True" and row["geo_error_m"]:
                row["geo_error_m"] = float(row["geo_error_m"])
                row["px_error"] = float(row["px_error"])
                row["gt_lat"] = float(row["gt_lat"])
                row["gt_lon"] = float(row["gt_lon"])
                rows.append(row)

    if not rows:
        print(
            f"No successful results found for scene {args.scene} in {args.results}")
        return

    # Pick images based on mode
    if args.mode == "best":
        rows = sorted(rows, key=lambda r: r["geo_error_m"])[:args.n]
    elif args.mode == "worst":
        rows = sorted(rows, key=lambda r: r["geo_error_m"], reverse=True)[
            :args.n]
    else:
        import random
        rows = random.sample(rows, min(args.n, len(rows)))

    print(f"Showing {len(rows)} images (mode={args.mode})")
    print("Press any key to advance, Q to quit.\n")

    tif_path = next(scene_dir.glob("*.tif"))

    with rasterio.open(tif_path) as sat_ds:
        for row in rows:
            drone_path = scene_dir / "drone" / row["filename"]
            if not drone_path.exists():
                print(f"  Image not found: {drone_path}")
                continue

            gt_col, gt_row = latlon_to_pixel(
                row["gt_lat"], row["gt_lon"], sat_ds)
            sat_gray, sat_bgr, gt_in_patch, patch_origin = crop_search_region(
                sat_ds, gt_col, gt_row, args.search_radius)

            drone_bgr = cv2.imread(str(drone_path), cv2.IMREAD_COLOR)
            if drone_bgr is None:
                print(f"  Could not load {drone_path}")
                continue
            drone_bgr = cv2.resize(drone_bgr, (512, 512))

            print(
                f"  {row['filename']} | geo_error={row['geo_error_m']:.1f}m | px_error={row['px_error']:.1f}px")
            key = visualize_one(sat_bgr, sat_gray, drone_bgr, drone_path,
                                gt_in_patch, row["geo_error_m"], row["px_error"])

            if key == ord("q") or key == ord("Q"):
                print("Quit.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
