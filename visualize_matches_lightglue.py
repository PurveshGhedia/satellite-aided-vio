"""
Visualize LightGlue matches between drone and satellite patch.
--------------------------------------------------------------
Loads results from the LightGlue benchmark CSV, picks a few images,
and shows the keypoint matches side by side in a popup window.

Usage:
    python visualize_matches_lightglue.py --dataset /path/to/UAV_VisLoc_dataset \
                                          --results lightglue_benchmark_results.csv \
                                          --scene 03 \
                                          --n 5 \
                                          --mode worst
"""

import argparse
import csv
import cv2
import numpy as np
import torch
import rasterio
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd


# ---------------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------------

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(DEVICE)
matcher = LightGlue(features="superpoint").eval().to(DEVICE)
print(f"[LightGlue] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Geo / crop helpers (same as benchmark)
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
    else:
        img_bgr = cv2.cvtColor(data[0].astype(np.uint8), cv2.COLOR_GRAY2BGR)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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


def to_tensor(img_gray):
    t = torch.from_numpy(img_gray).float() / 255.0
    return t.unsqueeze(0).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Visualize one image pair
# ---------------------------------------------------------------------------

def visualize_one(sat_bgr, sat_gray, drone_bgr, drone_path,
                  gt_in_patch, geo_error_m, px_error, patch_origin):

    drone_gray = cv2.cvtColor(drone_bgr, cv2.COLOR_BGR2GRAY)

    with torch.no_grad():
        feats_sat = extractor.extract(to_tensor(sat_gray))
        feats_drone = extractor.extract(to_tensor(drone_gray))
        matches_out = matcher({"image0": feats_sat, "image1": feats_drone})

    feats_sat = rbd(feats_sat)
    feats_drone = rbd(feats_drone)
    matches_out = rbd(matches_out)

    match_indices = matches_out["matches"]
    n_matches = match_indices.shape[0]

    if n_matches < 4:
        print(f"  Too few matches ({n_matches}) for {drone_path.name}")
        return None

    kp_sat_np = feats_sat["keypoints"][match_indices[:, 0]].cpu().numpy()
    kp_drone_np = feats_drone["keypoints"][match_indices[:, 1]].cpu().numpy()

    src_pts = kp_drone_np.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = kp_sat_np.reshape(-1, 1, 2).astype(np.float32)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist() if mask is not None else [
        0] * n_matches
    inlier_count = sum(matchesMask)
    inlier_ratio = inlier_count / n_matches if n_matches > 0 else 0

    # Convert keypoints to cv2.KeyPoint for drawMatches
    kp_sat_cv = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kp_sat_np]
    kp_drone_cv = [cv2.KeyPoint(float(p[0]), float(p[1]), 1)
                   for p in kp_drone_np]
    dm_matches = [cv2.DMatch(i, i, 0) for i in range(n_matches)]

    # Draw inliers (green)
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    vis = cv2.drawMatches(drone_bgr, kp_drone_cv, sat_bgr, kp_sat_cv,
                          dm_matches, None, **draw_params)

    # Draw outliers (red)
    outlier_mask = [1 - m for m in matchesMask]
    draw_outliers = dict(matchColor=(0, 0, 255), singlePointColor=None,
                         matchesMask=outlier_mask,
                         flags=cv2.DrawMatchesFlags_DEFAULT)
    vis = cv2.drawMatches(drone_bgr, kp_drone_cv, sat_bgr, kp_sat_cv,
                          dm_matches, vis, **draw_outliers)

    drone_w = drone_bgr.shape[1]

    # Mark estimated position
    if M is not None:
        h, w = drone_gray.shape[:2]
        center = np.array([[[w / 2, h / 2]]], dtype=np.float32)
        est_px = cv2.perspectiveTransform(center, M)[0][0]
        est_on_vis = (int(est_px[0]) + drone_w, int(est_px[1]))
        cv2.drawMarker(vis, est_on_vis, (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(vis, "EST", (est_on_vis[0] + 8, est_on_vis[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Mark ground truth position
    gt_on_vis = (int(gt_in_patch[0]) + drone_w, int(gt_in_patch[1]))
    cv2.drawMarker(vis, gt_on_vis, (0, 255, 0), cv2.MARKER_STAR, 30, 3)
    cv2.putText(vis, "GT", (gt_on_vis[0] + 8, gt_on_vis[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Info overlay
    info = [
        f"Image : {drone_path.name}",
        f"Matches: {n_matches}  Inliers: {inlier_count}  Ratio: {inlier_ratio:.2f}",
        f"Pixel error : {px_error:.1f} px",
        f"Geo error   : {geo_error_m:.1f} m",
        f"Green lines = inliers | Red lines = outliers",
        f"Yellow cross = estimated | Green star = ground truth",
        f"Press any key for next image, Q to quit",
    ]
    y = 20
    for line in info:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        cv2.putText(vis, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        y += 22

    # Resize to fit screen
    screen_w = 1800
    h_vis, w_vis = vis.shape[:2]
    if w_vis > screen_w:
        scale = screen_w / w_vis
        vis = cv2.resize(vis, (int(w_vis * scale), int(h_vis * scale)))

    cv2.imshow("LightGlue Match Visualizer", vis)
    return cv2.waitKey(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize LightGlue matches on UAV-VisLoc")
    parser.add_argument("--dataset",       required=True)
    parser.add_argument("--results",       required=True)
    parser.add_argument("--scene",         default="03")
    parser.add_argument("--n",             type=int, default=5)
    parser.add_argument("--search_radius", type=int, default=600)
    parser.add_argument("--mode",          default="random",
                        choices=["random", "best", "worst"])
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    scene_dir = dataset_root / args.scene

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
        print(f"No successful results found for scene {args.scene}")
        return

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
                continue

            gt_col, gt_row = latlon_to_pixel(
                row["gt_lat"], row["gt_lon"], sat_ds)
            sat_gray, sat_bgr, gt_in_patch, patch_origin = crop_search_region(
                sat_ds, gt_col, gt_row, args.search_radius)

            drone_bgr = cv2.imread(str(drone_path), cv2.IMREAD_COLOR)
            if drone_bgr is None:
                continue
            drone_bgr = cv2.resize(drone_bgr, (512, 512))

            print(
                f"  {row['filename']} | geo_error={row['geo_error_m']:.1f}m | px_error={row['px_error']:.1f}px")
            key = visualize_one(sat_bgr, sat_gray, drone_bgr, drone_path,
                                gt_in_patch, row["geo_error_m"], row["px_error"],
                                patch_origin)
            if key == ord("q") or key == ord("Q"):
                print("Quit.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
