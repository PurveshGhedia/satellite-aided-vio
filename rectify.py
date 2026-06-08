"""
Drone Image Pre-Rectification
------------------------------
Warps a tilted drone image into a simulated top-down (nadir) view
using pitch and roll angles from the dataset CSV (or VINS-Mono pose
in production).

The rectification builds a homography from:
  - Camera intrinsic matrix K (estimated from FOV if not known)
  - Rotation matrix R from pitch (Omega) and roll (Kappa)

Yaw (Phi) is intentionally excluded -- yaw rotates the view around
the vertical axis which changes image orientation but NOT perspective
distortion. We handle orientation separately during matching.

Usage as a module:
    from rectify import rectify_drone_image
    rectified = rectify_drone_image(img, pitch_deg, roll_deg)

Usage as a script (visual check):
    python rectify.py --image /path/to/drone.JPG --pitch 5.2 --roll 2.1
    python rectify.py --dataset /path/to/UAV_VisLoc_dataset --scene 03 --n 5
"""

import cv2
import numpy as np
import argparse
import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Camera intrinsics estimation
# ---------------------------------------------------------------------------

def estimate_K(img_width, img_height, hfov_deg=84.0):
    """
    Build a camera intrinsic matrix K from image dimensions and
    horizontal field of view.

    For a typical drone survey camera (DJI FC300X class):
      hfov ~ 84 degrees → fx ~ 0.78 * width

    In production this should be replaced with your actual calibrated
    K matrix from camera-IMU calibration (Kalibr output).

    Args:
        img_width:  image width in pixels
        img_height: image height in pixels
        hfov_deg:   horizontal field of view in degrees

    Returns:
        K: 3x3 intrinsic matrix
    """
    fx = (img_width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    fy = fx  # assume square pixels
    cx = img_width / 2.0
    cy = img_height / 2.0

    K = np.array([
        [fx,  0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    return K


# ---------------------------------------------------------------------------
# Rotation matrix from pitch and roll
# ---------------------------------------------------------------------------

def rotation_matrix(pitch_deg, roll_deg):
    """
    Build a 3x3 rotation matrix from pitch and roll angles.

    Convention (matching UAV-VisLoc dataset):
      pitch (Omega): rotation around X axis
      roll  (Kappa): rotation around Y axis

    We invert the angles because we want to UNDO the tilt,
    not apply it.

    Args:
        pitch_deg: pitch angle in degrees (Omega from CSV)
        roll_deg:  roll angle in degrees  (Kappa from CSV)

    Returns:
        R: 3x3 rotation matrix
    """
    p = np.radians(-pitch_deg)   # negate to undo tilt
    r = np.radians(-roll_deg)

    # Rotation around X (pitch)
    Rx = np.array([
        [1,          0,           0],
        [0,  np.cos(p),  -np.sin(p)],
        [0,  np.sin(p),   np.cos(p)]
    ], dtype=np.float64)

    # Rotation around Y (roll)
    Ry = np.array([
        [np.cos(r), 0, np.sin(r)],
        [0, 1,         0],
        [-np.sin(r), 0, np.cos(r)]
    ], dtype=np.float64)

    return Ry @ Rx


# ---------------------------------------------------------------------------
# Core rectification
# ---------------------------------------------------------------------------

def rectify_drone_image(img, pitch_deg, roll_deg, hfov_deg=84.0, output_size=None):
    """
    Warp a drone image to simulate a top-down nadir view by undoing
    the pitch and roll tilt.

    Args:
        img:         input drone image (BGR or grayscale, any size)
        pitch_deg:   pitch angle in degrees (Omega from CSV / VINS pose)
        roll_deg:    roll angle in degrees  (Kappa from CSV / VINS pose)
        hfov_deg:    horizontal FOV in degrees (tune if rectification looks wrong)
        output_size: (width, height) of output image. Defaults to input size.

    Returns:
        rectified:   warped image, same type as input
        H:           3x3 homography matrix used for warping
    """
    h, w = img.shape[:2]
    if output_size is None:
        output_size = (w, h)

    K = estimate_K(w, h, hfov_deg)
    R = rotation_matrix(pitch_deg, roll_deg)

    # Homography for planar rectification:
    # H = K * R * K_inv
    # This maps tilted image coordinates to rectified (nadir) coordinates
    K_inv = np.linalg.inv(K)
    H = K @ R @ K_inv

    rectified = cv2.warpPerspective(img, H, output_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
    return rectified, H


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def visualize_rectification(original, rectified, pitch_deg, roll_deg, filename=""):
    """
    Show original and rectified images side by side with info overlay.
    Returns the key pressed (ord('q') to quit).
    """
    # Resize both to same height for display
    display_h = 600
    scale_o = display_h / original.shape[0]
    scale_r = display_h / rectified.shape[0]

    orig_disp = cv2.resize(
        original,  (int(original.shape[1] * scale_o), display_h))
    rect_disp = cv2.resize(
        rectified, (int(rectified.shape[1] * scale_r), display_h))

    # Add labels
    def put_label(img, text):
        cv2.putText(img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    put_label(orig_disp, "Original (tilted)")
    put_label(rect_disp, "Rectified (nadir)")

    info = f"{filename}  |  pitch={pitch_deg:.2f} deg  roll={roll_deg:.2f} deg  |  press any key for next, Q to quit"
    combined = np.hstack([orig_disp, rect_disp])

    # Info bar at bottom
    bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    combined = np.vstack([combined, bar])

    cv2.imshow("Rectification Preview", combined)
    return cv2.waitKey(0)


# ---------------------------------------------------------------------------
# Script mode: visual check on dataset images
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Drone image rectification preview")
    parser.add_argument("--image",   help="Single image path for quick test")
    parser.add_argument("--pitch",   type=float,
                        default=5.0, help="Pitch in degrees")
    parser.add_argument("--roll",    type=float,
                        default=2.0, help="Roll in degrees")
    parser.add_argument("--dataset", help="Path to UAV_VisLoc_dataset/")
    parser.add_argument("--scene",   default="03", help="Scene ID")
    parser.add_argument("--n",       type=int, default=5,
                        help="Number of images to preview")
    parser.add_argument("--hfov",    type=float, default=84.0,
                        help="Camera horizontal FOV in degrees")
    parser.add_argument("--mode",    default="random",
                        choices=["random", "high_tilt", "low_tilt"],
                        help="Which images to pick")
    args = parser.parse_args()

    # --- Single image mode ---
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not load {args.image}")
            return
        rectified, H = rectify_drone_image(
            img, args.pitch, args.roll, args.hfov)
        print(f"Homography H:\n{H}")
        visualize_rectification(img, rectified, args.pitch, args.roll,
                                filename=Path(args.image).name)
        cv2.destroyAllWindows()
        return

    # --- Dataset mode ---
    if not args.dataset:
        print("Provide either --image or --dataset")
        return

    scene_dir = Path(args.dataset) / args.scene
    csv_path = next(scene_dir.glob("*.csv"))

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "filename": row["filename"],
                "pitch":    float(row["Omega"]),
                "roll":     float(row["Kappa"]),
            })

    # Pick images based on mode
    if args.mode == "high_tilt":
        # Images with largest combined tilt
        rows = sorted(rows, key=lambda r: abs(
            r["pitch"]) + abs(r["roll"]), reverse=True)[:args.n]
    elif args.mode == "low_tilt":
        rows = sorted(rows, key=lambda r: abs(
            r["pitch"]) + abs(r["roll"]))[:args.n]
    else:
        import random
        rows = random.sample(rows, min(args.n, len(rows)))

    print(f"Previewing {len(rows)} images (mode={args.mode})")
    print("Press any key to advance, Q to quit.\n")

    drone_dir = scene_dir / "drone"

    for row in rows:
        img_path = drone_dir / row["filename"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not load {img_path}")
            continue

        pitch, roll = row["pitch"], row["roll"]
        print(f"  {row['filename']}  pitch={pitch:.2f}  roll={roll:.2f}")

        rectified, H = rectify_drone_image(img, pitch, roll, args.hfov)
        key = visualize_rectification(img, rectified, pitch, roll,
                                      filename=row["filename"])
        if key == ord("q") or key == ord("Q"):
            print("Quit.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
