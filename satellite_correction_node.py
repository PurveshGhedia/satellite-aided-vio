#!/usr/bin/env python3
"""
satellite_correction_node.py
-----------------------------
ROS1 node that subscribes to drone camera images and a VIO odometry estimate,
runs LightGlue+SuperPoint matching against a satellite GeoTIFF, and publishes
corrected position estimates back into the navigation pipeline.

Architecture
------------
Subscriptions:
    /drone/image                sensor_msgs/Image
        Raw drone camera frames (debayered, BGR or grayscale).

    /vins_estimator/odometry    nav_msgs/Odometry
        Current VIO position estimate. Used to centre the satellite crop
        around the estimated drone position rather than ground truth.
        Falls back to last known position if odometry is stale.

Publications:
    /correction/pose            geometry_msgs/PoseWithCovarianceStamped
        Corrected position in the satellite map frame (ENU, origin at
        TIF top-left). Covariance[0,0] and [1,1] encode matching quality:
        low inlier ratio -> high covariance -> VINS treats this as a
        weak correction. Published ONLY when confidence gate passes.

    /correction/diagnostics     std_msgs/String  (JSON)
        Per-frame diagnostic blob. Published on every image regardless
        of gate outcome. Useful for Foxglove monitoring.
        Fields: filename, n_matches, inlier_count, inlier_ratio,
                gate_passed, geo_error_m (if GT available), elapsed_ms,
                reason (on failure).

Parameters (ROS params, set in launch file or rosparam)
----------
    ~sat_tif        str     Path to satellite GeoTIFF for the current scene.
    ~search_radius  int     Crop half-width in pixels (default: 600).
                            Should cover the worst-case VIO drift between
                            correction intervals.
    ~trigger_every  int     Run matching every N images received (default: 10).
                            LightGlue at ~500ms/frame on M1 CPU means you
                            don't want to run every frame.
    ~min_inliers    int     Confidence gate: minimum RANSAC inliers (default: 15).
    ~min_inlier_ratio float Confidence gate: minimum inlier/total ratio (default: 0.20).
    ~drone_resize   int     Resize drone image to this square before matching (default: 512).
    ~frame_id       str     Frame ID for published poses (default: "map").

Usage
-----
    # In your launch file:
    <node pkg="satellite_aided_vio" type="satellite_correction_node.py"
          name="satellite_correction" output="screen">
        <param name="sat_tif"       value="/data/UAV_VisLoc_dataset/03/scene_03.tif"/>
        <param name="search_radius" value="600"/>
        <param name="trigger_every" value="10"/>
    </node>

    # Or run standalone for testing (no Docker needed):
    python satellite_correction_node.py \
        --sat_tif ~/data/UAV_VisLoc_dataset/03/scene_03.tif \
        --mode standalone \
        --image_dir ~/data/UAV_VisLoc_dataset/03/drone \
        --csv ~/data/UAV_VisLoc_dataset/03/scene_03.csv
"""

import yaml
from lightglue.utils import rbd
from lightglue import LightGlue, SuperPoint
import sys
import json
import time
import argparse
import threading
import numpy as np
import cv2
import torch
import rasterio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple
torch.set_num_threads(2)

# LightGlue

# ---------------------------------------------------------------------------
# Conditional ROS import — allows the matcher core to run without ROS
# for standalone testing and unit tests.
# ---------------------------------------------------------------------------

try:
    import rospy
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


def load_kannala_brandt_calib(yaml_path: str):
    """
    Load KANNALA_BRANDT intrinsics from a VINS-Mono style camera yaml.
    Strips the '%YAML:1.0' header line that OpenCV writes but PyYAML
    chokes on, and registers a constructor for OpenCV's custom
    '!!opencv-matrix' tag (used for extrinsicRotation/Translation etc.)
    so the rest of the file parses even though we don't need those fields.
    """
    def _opencv_matrix_constructor(loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        return mapping  # we don't use these fields, just need it to parse

    loader_cls = yaml.SafeLoader
    loader_cls.add_constructor(
        "tag:yaml.org,2002:opencv-matrix", _opencv_matrix_constructor
    )

    with open(yaml_path, "r") as f:
        lines = f.readlines()
    lines = [l for l in lines if not l.strip().startswith("%YAML")]
    cfg = yaml.load("".join(lines), Loader=loader_cls)

    proj = cfg["projection_parameters"]
    K = np.array([
        [proj["mu"], 0.0,         proj["u0"]],
        [0.0,        proj["mv"],  proj["v0"]],
        [0.0,        0.0,         1.0],
    ], dtype=np.float64)
    D = np.array([proj["k2"], proj["k3"], proj["k4"],
                 proj["k5"]], dtype=np.float64)
    image_size = (int(cfg["image_width"]), int(cfg["image_height"]))
    return K, D, image_size


def central_gradient(img, centre_fraction: float = 0.5) -> float:
    """
    Mean Sobel gradient magnitude in the central region of the image.
    Accepts either a grayscale (2D) or BGR (3D) image.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    h, w = gray.shape
    margin_h = int(h * (1 - centre_fraction) / 2)
    margin_w = int(w * (1 - centre_fraction) / 2)
    central = gray[margin_h:h-margin_h, margin_w:w-margin_w]
    gx = cv2.Sobel(central, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(central, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.mean(mag))


# ===========================================================================
# Matcher core — pure Python, no ROS dependency
# ===========================================================================


@dataclass
class MatchResult:
    """Structured output from one matching attempt."""
    success: bool
    gate_passed: bool = False
    # Pixel position of drone centre in the FULL satellite image (col, row)
    est_col: Optional[float] = None
    est_row: Optional[float] = None
    # Geo position (WGS84) derived from est_col/row
    est_lat: Optional[float] = None
    est_lon: Optional[float] = None
    n_matches: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    elapsed_ms: float = 0.0
    reason: str = ""


@dataclass
class CropResult:
    """Output from satellite crop operation."""
    img_gray: np.ndarray
    patch_origin: Tuple[int, int]   # (col_offset, row_offset) in full TIF


class SatelliteMatcher:
    """
    Encapsulates the LightGlue+SuperPoint matching pipeline.

    Designed to be instantiated once and called repeatedly. All model
    weights are loaded at construction time — matching itself is stateless.

    This class has zero ROS dependency and can be tested standalone.
    """

    def __init__(
        self,
        sat_tif_path: str,
        search_radius_px: int = 600,
        min_inliers: int = 15,
        min_inlier_ratio: float = 0.20,
        drone_resize: int = 512,
        ransac_thresh: float = 5.0,
        max_keypoints: int = 2048,
        cam_K: Optional[np.ndarray] = None,      # NEW
        cam_D: Optional[np.ndarray] = None,      # NEW
        cam_image_size: Optional[Tuple[int, int]] = None,  # NEW
        undistort_balance: float = 0.5,          # NEW
        use_similarity_transform: bool = False,   # NEW
    ):
        self.sat_tif_path = sat_tif_path
        self.search_radius_px = search_radius_px
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.drone_resize = drone_resize
        self.ransac_thresh = ransac_thresh
        self.use_similarity_transform = use_similarity_transform

        # --- Fisheye undistortion setup (NEW) ---
        self._undistort_maps = None
        if cam_K is not None and cam_D is not None and cam_image_size is not None:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                cam_K, cam_D, cam_image_size, np.eye(3),
                balance=undistort_balance,
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                cam_K, cam_D, np.eye(3), new_K, cam_image_size, cv2.CV_16SC2,
            )
            self._undistort_maps = (map1, map2)
            print(f"[SatelliteMatcher] Fisheye undistortion enabled "
                  f"(image_size={cam_image_size}, balance={undistort_balance})")
        else:
            print("[SatelliteMatcher] WARNING: no camera calibration provided — "
                  "running on raw (distorted) frames.")
        # ------------------------------------------

        # Device selection: MPS on Apple Silicon, CUDA if available, else CPU.
        # In Docker on M1 you'll get CPU — that's fine at 1 Hz trigger rate.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"[SatelliteMatcher] Device: {self.device}")
        print(f"[SatelliteMatcher] Loading SuperPoint + LightGlue...")

        self.extractor = SuperPoint(
            max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

        print(f"[SatelliteMatcher] Models loaded.")
        print(f"[SatelliteMatcher] TIF: {sat_tif_path}")

        # Open the rasterio dataset once and keep it open.
        # rasterio datasets are NOT thread-safe — acquire self._rasterio_lock
        # before any read operation.
        self._sat_ds = rasterio.open(sat_tif_path)
        self._rasterio_lock = threading.Lock()

        self._log_tif_info()

    def _log_tif_info(self):
        ds = self._sat_ds
        print(f"[SatelliteMatcher] TIF size  : {ds.width} x {ds.height} px")
        print(f"[SatelliteMatcher] TIF CRS   : {ds.crs}")
        res = abs(ds.transform[0])
        if ds.crs and ds.crs.is_geographic:
            metres_per_px = res * 111320.0
        else:
            metres_per_px = res
        print(f"[SatelliteMatcher] Resolution: {metres_per_px:.2f} m/px")

    def _undistort(self, img_bgr: np.ndarray) -> np.ndarray:
        if self._undistort_maps is None:
            return img_bgr
        map1, map2 = self._undistort_maps
        return cv2.remap(img_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # Geo utilities
    # ------------------------------------------------------------------

    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        """Convert WGS84 lat/lon to (col, row) in the satellite TIF."""
        ds = self._sat_ds
        if ds.crs and not ds.crs.is_geographic:
            from rasterio.warp import transform as rio_transform
            xs, ys = rio_transform("EPSG:4326", ds.crs, [lon], [lat])
            row, col = ds.index(xs[0], ys[0])
            return float(col), float(row)
        else:
            row, col = ds.index(lon, lat)
            return float(col), float(row)

    def pixel_to_latlon(self, col: float, row: float) -> Tuple[float, float]:
        """Convert (col, row) in the satellite TIF to WGS84 lat/lon."""
        ds = self._sat_ds
        # rasterio.transform.xy returns (x, y) in the dataset CRS
        x, y = rasterio.transform.xy(ds.transform, row, col)
        if ds.crs and not ds.crs.is_geographic:
            from rasterio.warp import transform as rio_transform
            lons, lats = rio_transform(ds.crs, "EPSG:4326", [x], [y])
            return float(lats[0]), float(lons[0])
        else:
            # Geographic CRS: x=lon, y=lat
            return float(y), float(x)

    def metres_per_pixel(self) -> float:
        ds = self._sat_ds
        res = abs(ds.transform[0])
        if ds.crs and ds.crs.is_geographic:
            return res * 111320.0
        return res

    # ------------------------------------------------------------------
    # Satellite crop
    # ------------------------------------------------------------------

    def crop_search_region(
        self,
        centre_col: float,
        centre_row: float,
    ) -> CropResult:
        """
        Crop a square patch from the satellite TIF centred on (centre_col, centre_row).

        In the benchmark this was centred on GT — here it will be centred on the
        VINS-Mono odometry estimate. The search_radius_px must comfortably cover
        the worst-case VIO drift between correction intervals.

        Thread-safe via self._rasterio_lock.
        """
        half = self.search_radius_px
        c_off = max(0, int(centre_col - half))
        r_off = max(0, int(centre_row - half))

        with self._rasterio_lock:
            ds = self._sat_ds
            width = min(half * 2, ds.width - c_off)
            height = min(half * 2, ds.height - r_off)
            window = rasterio.windows.Window(c_off, r_off, width, height)
            data = ds.read(window=window)

        if data.shape[0] >= 3:
            rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = data[0].astype(np.uint8)

        return CropResult(img_gray=img_gray, patch_origin=(c_off, r_off))

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _to_tensor(self, img_gray: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img_gray).float() / 255.0
        return t.unsqueeze(0).unsqueeze(0).to(self.device)

    def match(
        self,
        drone_img_bgr: np.ndarray,
        vins_lat: Optional[float] = None,
        vins_lon: Optional[float] = None,
        centre_col: Optional[float] = None,
        centre_row: Optional[float] = None,
    ) -> MatchResult:
        """
        Run one matching attempt.

        Crop centre can be provided as either:
          - (vins_lat, vins_lon): converted to pixel internally, or
          - (centre_col, centre_row): already in TIF pixel space.

        If neither is provided, returns a failed MatchResult immediately.

        Args:
            drone_img_bgr:  BGR drone image (any size — will be resized internally).
            vins_lat/lon:   VINS-Mono position estimate in WGS84.
            centre_col/row: VINS-Mono position estimate in TIF pixel space.

        Returns:
            MatchResult with all fields populated.
        """
        t0 = time.time()

        # Resolve crop centre
        if centre_col is None or centre_row is None:
            if vins_lat is None or vins_lon is None:
                return MatchResult(success=False, reason="no position estimate provided")
            try:
                centre_col, centre_row = self.latlon_to_pixel(
                    vins_lat, vins_lon)
            except Exception as e:
                return MatchResult(success=False, reason=f"latlon_to_pixel failed: {e}")

        # Crop satellite patch
        try:
            crop = self.crop_search_region(centre_col, centre_row)
        except Exception as e:
            return MatchResult(success=False, reason=f"crop failed: {e}")

        sat_gray = crop.img_gray

        # Prepare drone image
        drone_img_bgr = self._undistort(drone_img_bgr)
        drone_gray = cv2.cvtColor(drone_img_bgr, cv2.COLOR_BGR2GRAY) \
            if drone_img_bgr.ndim == 3 else drone_img_bgr.copy()

        scale = self.drone_resize / max(drone_gray.shape[:2])
        new_w = int(round(drone_gray.shape[1] * scale))
        new_h = int(round(drone_gray.shape[0] * scale))
        drone_gray = cv2.resize(drone_gray, (new_w, new_h))

        # SuperPoint + LightGlue
        with torch.no_grad():
            feats_sat = self.extractor.extract(self._to_tensor(sat_gray))
            feats_drone = self.extractor.extract(self._to_tensor(drone_gray))
            matches_out = self.matcher(
                {"image0": feats_sat, "image1": feats_drone})

        feats_sat = rbd(feats_sat)
        feats_drone = rbd(feats_drone)
        matches_out = rbd(matches_out)

        match_indices = matches_out["matches"]
        n_matches = match_indices.shape[0]

        if n_matches < 4:
            elapsed = (time.time() - t0) * 1000
            return MatchResult(
                success=False,
                reason="too few matches from LightGlue",
                n_matches=n_matches,
                elapsed_ms=elapsed,
            )

        kp_sat = feats_sat["keypoints"][match_indices[:, 0]].cpu().numpy()
        kp_drone = feats_drone["keypoints"][match_indices[:, 1]].cpu().numpy()

        src_pts = kp_drone.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = kp_sat.reshape(-1, 1, 2).astype(np.float32)

        if self.use_similarity_transform:
            M, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh)
            # estimateAffinePartial2D returns a 2x3 matrix, but
            # perspectiveTransform later needs a 3x3 matrix, so pad it.
            if M is not None:
                M = np.vstack([M, [0, 0, 1]])
        else:
            M, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)

        elapsed = (time.time() - t0) * 1000

        if M is None:
            return MatchResult(
                success=False,
                reason="RANSAC transform failed",
                n_matches=n_matches,
                elapsed_ms=elapsed,
            )

        inlier_count = int(mask.ravel().sum())
        inlier_ratio = inlier_count / n_matches if n_matches > 0 else 0.0

        # Project drone image centre through homography to get position in sat patch
        h, w = drone_gray.shape[:2]
        drone_centre = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
        est_in_patch = cv2.perspectiveTransform(drone_centre, M)[0][0]

        # Back to full TIF coordinates
        est_col = float(est_in_patch[0]) + crop.patch_origin[0]
        est_row = float(est_in_patch[1]) + crop.patch_origin[1]

        # Confidence gate
        gate_passed = (
            inlier_count >= self.min_inliers
            and inlier_ratio >= self.min_inlier_ratio
        )

        # Convert estimated pixel to lat/lon
        try:
            est_lat, est_lon = self.pixel_to_latlon(est_col, est_row)
        except Exception as e:
            est_lat, est_lon = None, None

        return MatchResult(
            success=True,
            gate_passed=gate_passed,
            est_col=est_col,
            est_row=est_row,
            est_lat=est_lat,
            est_lon=est_lon,
            n_matches=n_matches,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            elapsed_ms=elapsed,
        )

    def shutdown(self):
        self._sat_ds.close()

    def visualize_match(
        self,
        drone_img_bgr: np.ndarray,
        save_path: str,
        vins_lat: Optional[float] = None,
        vins_lon: Optional[float] = None,
    ) -> "MatchResult":
        """
        Run one match and save a side-by-side image showing only the
        INLIER matches as connecting lines. Useful for a manual sanity
        check that matches are real correspondences, not coincidences
        that happened to fit a transform.
        """
        # Re-run the same steps as match(), but keep the intermediate
        # images and keypoints around so we can draw them.
        centre_col, centre_row = self.latlon_to_pixel(vins_lat, vins_lon)
        crop = self.crop_search_region(centre_col, centre_row)
        sat_gray = crop.img_gray

        drone_img_bgr = self._undistort(drone_img_bgr)
        drone_gray = cv2.cvtColor(drone_img_bgr, cv2.COLOR_BGR2GRAY) \
            if drone_img_bgr.ndim == 3 else drone_img_bgr.copy()

        scale = self.drone_resize / max(drone_gray.shape[:2])
        new_w = int(round(drone_gray.shape[1] * scale))
        new_h = int(round(drone_gray.shape[0] * scale))
        drone_gray = cv2.resize(drone_gray, (new_w, new_h))

        with torch.no_grad():
            feats_sat = self.extractor.extract(self._to_tensor(sat_gray))
            feats_drone = self.extractor.extract(self._to_tensor(drone_gray))
            matches_out = self.matcher(
                {"image0": feats_sat, "image1": feats_drone})

        feats_sat = rbd(feats_sat)
        feats_drone = rbd(feats_drone)
        matches_out = rbd(matches_out)
        match_indices = matches_out["matches"]

        kp_sat = feats_sat["keypoints"][match_indices[:, 0]].cpu().numpy()
        kp_drone = feats_drone["keypoints"][match_indices[:, 1]].cpu().numpy()

        src_pts = kp_drone.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = kp_sat.reshape(-1, 1, 2).astype(np.float32)

        if self.use_similarity_transform:
            M, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh)
        else:
            M, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)

        inlier_mask = mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())
        print(
            f"[visualize_match] {n_inliers} inliers out of {len(match_indices)} matches")

        # Build cv2.KeyPoint lists and DMatch list for drawMatches,
        # but only include the INLIER matches so the picture isn't cluttered.
        kp_sat_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp_sat]
        kp_drone_cv = [cv2.KeyPoint(float(x), float(y), 1)
                       for x, y in kp_drone]

        good_dmatches = [
            cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0)
            for i in range(len(kp_sat)) if inlier_mask[i]
        ]

        sat_color = cv2.cvtColor(sat_gray, cv2.COLOR_GRAY2BGR)
        drone_color = cv2.cvtColor(drone_gray, cv2.COLOR_GRAY2BGR)

        vis = cv2.drawMatches(
            drone_color, kp_drone_cv,
            sat_color, kp_sat_cv,
            good_dmatches, None,
            matchColor=(0, 255, 0),      # green lines = inliers
            singlePointColor=(0, 0, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(save_path, vis)
        print(f"[visualize_match] Saved to {save_path}")


# ===========================================================================
# ROS node wrapper
# ===========================================================================

class SatelliteCorrectionNode:
    """
    ROS1 node wrapping SatelliteMatcher.

    Designed for ROS1 (Noetic / Melodic). For ROS2 the structure is similar
    but uses rclpy and Node subclassing — adapt when you migrate.
    """

    def __init__(self):
        rospy.init_node("satellite_correction_node", anonymous=False)

        # Parameters
        sat_tif = rospy.get_param("~sat_tif")
        search_radius = rospy.get_param("~search_radius",      300)
        self.trigger_n = rospy.get_param("~trigger_every",       10)
        min_inliers = rospy.get_param("~min_inliers",         10)
        min_inlier_ratio = rospy.get_param("~min_inlier_ratio", 0.20)
        drone_resize = rospy.get_param("~drone_resize",        512)
        self.frame_id = rospy.get_param("~frame_id",          "map")
        min_gradient = rospy.get_param("~min_gradient", 32.0)
        self.min_gradient = min_gradient
        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self._frame_available = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._matching_worker, daemon=True)
        self._worker_thread.start()

        # NEW: camera calibration + similarity transform
        cam_config = rospy.get_param("~cam_config", None)
        undistort_balance = rospy.get_param("~undistort_balance", 0.5)
        use_similarity_transform = rospy.get_param(
            "~use_similarity_transform", True)

        cam_K = cam_D = cam_image_size = None
        if cam_config:
            cam_K, cam_D, cam_image_size = load_kannala_brandt_calib(
                cam_config)

        # Internal state
        self._bridge = CvBridge()
        self._image_count = 0
        self._latest_odom = None   # nav_msgs/Odometry from VINS
        self._odom_lock = threading.Lock()

        # Matcher (loads models — takes a few seconds)
        self.matcher = SatelliteMatcher(
            sat_tif_path=sat_tif,
            search_radius_px=search_radius,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            drone_resize=drone_resize,
            cam_K=cam_K,
            cam_D=cam_D,
            cam_image_size=cam_image_size,
            undistort_balance=undistort_balance,
            use_similarity_transform=use_similarity_transform,
        )

        # Publishers
        self._pub_pose = rospy.Publisher(
            "/correction/pose",
            PoseWithCovarianceStamped,
            queue_size=5,
        )
        self._pub_diag = rospy.Publisher(
            "/correction/diagnostics",
            String,
            queue_size=20,
        )

        # Subscribers — image drives processing, odometry is just state
        rospy.Subscriber("/drone/image",            Image,
                         self._image_cb,   queue_size=2)
        rospy.Subscriber("/vins_estimator/odometry", Odometry,
                         self._odom_cb,   queue_size=5)

        rospy.loginfo(
            "[SatCorr] Node ready. Triggering every %d images.", self.trigger_n)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: "Odometry"):
        """Cache the latest VIO odometry. Non-blocking."""
        with self._odom_lock:
            self._latest_odom = msg

    def _image_cb(self, msg: "Image"):
        """
        Lightweight callback — just stores the latest frame and returns
        immediately. The actual matching work happens in a separate
        thread (_matching_worker), so this callback never blocks and
        ROS can keep reading new images in real time.
        """
        self._image_count += 1
        if self._image_count % self.trigger_n != 0:
            return

        try:
            drone_bgr = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn("[SatCorr] imgmsg_to_cv2 failed: %s", e)
            return

        arrival_time = time.time()
        with self._latest_frame_lock:
            self._latest_frame = (msg.header, drone_bgr, arrival_time)
        self._frame_available.set()

    def _matching_worker(self):
        """
        Runs in a separate thread. Always processes the MOST RECENT
        frame available — if a new frame arrives while a match is still
        running, the old one is simply discarded rather than queued.
        This prevents the backlog/staleness problem entirely.
        """
        while not rospy.is_shutdown():
            got_frame = self._frame_available.wait(timeout=1.0)
            if not got_frame:
                continue
            self._frame_available.clear()

            with self._latest_frame_lock:
                if self._latest_frame is None:
                    continue
                header, drone_bgr, arrival_time = self._latest_frame
                self._latest_frame = None  # consumed — next one won't be stale

            # measure how old this frame actually is when we start processing it
            queue_delay = time.time() - arrival_time
            rospy.loginfo(
                "[SatCorr] Frame sat in queue for %.2fs before processing", queue_delay)

            grad = central_gradient(drone_bgr)
            if grad < self.min_gradient:
                rospy.loginfo_throttle(
                    5, "[SatCorr] Skipping low-texture frame (gradient=%.1f < %.1f)",
                    grad, self.min_gradient)
                continue

            with self._odom_lock:
                odom = self._latest_odom
            vins_lat, vins_lon = self._odom_to_latlon(odom)
            if vins_lat is None:
                rospy.logwarn_throttle(
                    10, "[SatCorr] No VIO odometry yet — skipping correction.")
                continue

            result = self.matcher.match(
                drone_img_bgr=drone_bgr,
                vins_lat=vins_lat,
                vins_lon=vins_lon,
            )

            self._publish_diagnostics(header, result)

            if result.gate_passed:
                self._publish_pose(header, result)
                rospy.loginfo(
                    "[SatCorr] ACCEPTED | lat=%.6f lon=%.6f | "
                    "inliers=%d ratio=%.2f | %.0fms",
                    result.est_lat, result.est_lon,
                    result.inlier_count, result.inlier_ratio, result.elapsed_ms,
                )
            else:
                reason = result.reason if not result.success else (
                    f"gate failed: inliers={result.inlier_count}, ratio={result.inlier_ratio:.2f}"
                )
                rospy.loginfo("[SatCorr] REJECTED | %s | %.0fms",
                              reason, result.elapsed_ms)
    # ------------------------------------------------------------------
    # VINS odometry -> lat/lon
    # ------------------------------------------------------------------

    def _odom_to_latlon(
        self, odom: Optional["Odometry"]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract a geographic position from VINS-Mono odometry.

        VINS-Mono publishes in its local ENU frame, not WGS84. To centre
        the satellite crop you need a geographic reference. Three options
        in increasing sophistication:

        Option A (current — temporary):
            Hardcode the scene origin lat/lon as a ROS param. Works for
            UAV-VisLoc where you know the TIF coverage area.

        Option B (next step):
            Publish a NavSatFix from the dataset_publisher_node and
            subscribe to it here. The dataset_publisher already has GT
            lat/lon per frame — use that as the "VIO estimate" during
            testing to replicate the benchmark's GT-centred crop.

        Option C (production):
            Subscribe to /mavros/global_position/global (NavSatFix) for
            real hardware. VINS provides relative drift, GPS provides the
            absolute reference for the crop centre.

        For now this node reads a scene_origin_lat/lon param (Option A),
        which is good enough to get the pipeline running.
        """
        if odom is None:
            return None, None

        # Read scene origin from params (set in launch file for the current scene)
        origin_lat = rospy.get_param("~scene_origin_lat", None)
        origin_lon = rospy.get_param("~scene_origin_lon", None)

        if origin_lat is None or origin_lon is None:
            rospy.logwarn_throttle(
                30, "[SatCorr] scene_origin_lat/lon not set. "
                "Cannot convert VIO position to geographic coordinates.")
            return None, None

        # VINS local ENU position (in metres from origin)
        # x = East, y = North in standard ENU
        x_m = odom.pose.pose.position.x
        y_m = odom.pose.pose.position.y

        # Approximate: 1 degree lat ~ 111320m, 1 degree lon ~ 111320 * cos(lat)
        origin_lat = float(origin_lat)
        origin_lon = float(origin_lon)
        import math
        lat = origin_lat + (y_m / 111320.0)
        lon = origin_lon + \
            (x_m / (111320.0 * math.cos(math.radians(origin_lat))))

        return lat, lon

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish_pose(self, header, result: MatchResult):
        """
        Publish the corrected position as PoseWithCovarianceStamped.

        Position: est_col, est_row in the satellite TIF pixel frame.
        (x = col, y = row, z = 0). The consuming node (or VINS loop
        closure interface) needs to know this is in pixel space and
        convert appropriately. For a real system, publish in ENU metres
        relative to the scene origin instead.

        Covariance: diagonal, scaled by (1 - inlier_ratio). High inlier
        ratio -> low covariance -> strong correction. This lets VINS or
        an EKF weight the correction appropriately if you wire it in.
        """
        msg = PoseWithCovarianceStamped()
        msg.header = header
        msg.header.frame_id = self.frame_id

        msg.pose.pose.position.x = result.est_col
        msg.pose.pose.position.y = result.est_row
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.w = 1.0  # identity, no orientation estimate

        # Covariance matrix (6x6 row-major). Position variance in x and y.
        # Units: pixels^2. Scale with matching quality.
        # Perfect match (ratio=1.0) -> sigma ~ 5px. Weak match -> larger.
        sigma_px = 5.0 + (1.0 - result.inlier_ratio) * 50.0
        variance = sigma_px ** 2
        cov = [0.0] * 36
        cov[0] = variance   # x variance
        cov[7] = variance   # y variance
        cov[14] = 1e6        # z (unknown)
        cov[21] = 1e6        # roll (unknown)
        cov[28] = 1e6        # pitch (unknown)
        cov[35] = 1e6        # yaw (unknown)
        msg.pose.covariance = cov

        self._pub_pose.publish(msg)

    def _publish_diagnostics(self, header, result: MatchResult):
        """Publish a JSON diagnostic blob for every processed frame."""
        diag = {
            "stamp":         header.stamp.to_sec() if hasattr(header.stamp, "to_sec") else 0,
            "success":       result.success,
            "gate_passed":   result.gate_passed,
            "n_matches":     result.n_matches,
            "inlier_count":  result.inlier_count,
            "inlier_ratio":  round(result.inlier_ratio, 3),
            "elapsed_ms":    round(result.elapsed_ms, 1),
            "est_lat":       result.est_lat,
            "est_lon":       result.est_lon,
            "reason":        result.reason,
        }
        self._pub_diag.publish(String(data=json.dumps(diag)))

    def spin(self):
        rospy.spin()

    def shutdown(self):
        self.matcher.shutdown()
        rospy.loginfo("[SatCorr] Shutdown complete.")


# ===========================================================================
# Standalone mode — runs the matcher directly against dataset images
# without any ROS infrastructure. Uses the same SatelliteMatcher class
# so you're testing the exact code that will run in the ROS node.
# ===========================================================================

def run_standalone(args):
    """
    Replay UAV-VisLoc dataset images through the matcher and print results.
    No ROS required. Useful during development and for quick sanity checks
    after any code change.

    Usage:
        python satellite_correction_node.py \
            --mode standalone \
            --sat_tif /data/UAV_VisLoc/03/scene_03.tif \
            --image_dir /data/UAV_VisLoc/03/drone \
            --csv /data/UAV_VisLoc/03/scene_03.csv \
            --n 20
    """
    import csv as csv_module

    vio_track = []
    if args.vio_csv:
        with open(args.vio_csv, newline="") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                vio_track.append(
                    (float(row["timestamp"]), float(row["lat"]), float(row["lon"])))
        vio_track.sort()

    def nearest_vio(ts):
        if not vio_track:
            return None, None
        import bisect
        idx = bisect.bisect_left(vio_track, (ts,))
        idx = min(max(idx, 0), len(vio_track) - 1)
        return vio_track[idx][1], vio_track[idx][2]

    cam_K = cam_D = cam_image_size = None
    if args.cam_config:
        cam_K, cam_D, cam_image_size = load_kannala_brandt_calib(
            args.cam_config)

    matcher = SatelliteMatcher(
        sat_tif_path=args.sat_tif,
        search_radius_px=args.search_radius,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        cam_K=cam_K,
        cam_D=cam_D,
        cam_image_size=cam_image_size,
        undistort_balance=args.undistort_balance,
        use_similarity_transform=args.use_similarity_transform,   # NEW
    )

    # Load GT from CSV (used to centre the crop, same as benchmark)
    gt_lookup = {}
    if args.csv:
        with open(args.csv, newline="") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                gt_lookup[row["filename"]] = {
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                }

    image_dir = Path(args.image_dir)
    images = sorted(image_dir.glob("*.jpg"))
    if args.n:
        images = images[:args.n]

    print(f"\n[Standalone] Running on {len(images)} images...\n")

    if args.visualize_frame:
        img_path = image_dir / args.visualize_frame
        drone_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt = gt_lookup.get(args.visualize_frame)
        if gt is None:
            print(f"No GT found for {args.visualize_frame}, cannot visualize.")
        else:
            matcher.visualize_match(
                drone_img_bgr=drone_bgr,
                save_path=args.visualize_out,
                vins_lat=gt["lat"],
                vins_lon=gt["lon"],
            )
        matcher.shutdown()
        return

    results = []
    for img_path in images:
        fname = img_path.name

        drone_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if drone_bgr is None:
            print(f"  [SKIP] Could not load {fname}")
            continue

        # Use VIO track if provided (simulates live behavior); else fall back to GT
        if vio_track:
            frame_ts = float(fname.replace("frame_", "").replace(".jpg", ""))
            vins_lat, vins_lon = nearest_vio(frame_ts)
        else:
            # Use GT as crop centre (replicates benchmark behaviour)
            vins_lat = vins_lon = None
            if fname in gt_lookup:
                vins_lat = gt_lookup[fname]["lat"]
                vins_lon = gt_lookup[fname]["lon"]

        result = matcher.match(
            drone_img_bgr=drone_bgr,
            vins_lat=vins_lat,
            vins_lon=vins_lon,
        )

        # Compute geo error if we have GT
        geo_err_str = "N/A"
        if result.success and fname in gt_lookup:
            gt_col, gt_row = matcher.latlon_to_pixel(
                gt_lookup[fname]["lat"], gt_lookup[fname]["lon"]
            )
            px_err = float(np.linalg.norm(
                np.array([result.est_col, result.est_row]) -
                np.array([gt_col, gt_row])
            ))
            geo_err_m = px_err * matcher.metres_per_pixel()
            geo_err_str = f"{geo_err_m:.1f}m"

        gate_str = "ACCEPTED" if result.gate_passed else "REJECTED"
        status = "OK  " if result.success else "FAIL"

        print(
            f"  {fname} | {status} | {gate_str} | "
            f"matches={result.n_matches:3d} inliers={result.inlier_count:3d} "
            f"ratio={result.inlier_ratio:.2f} | "
            f"geo_err={geo_err_str} | {result.elapsed_ms:.0f}ms"
            + (f" | {result.reason}" if result.reason else "")
        )
        results.append(result)

    # Summary
    total = len(results)
    accepted = sum(1 for r in results if r.gate_passed)
    failed = sum(1 for r in results if not r.success)
    print(f"\n{'='*60}")
    print(f"Total: {total}  |  Accepted: {accepted}  |  Failed: {failed}")
    print(f"{'='*60}\n")

    matcher.shutdown()


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Satellite correction node — ROS or standalone mode")
    parser.add_argument(
        "--mode", choices=["ros", "standalone"], default="ros",
        help="'ros' starts the ROS node; 'standalone' runs directly on dataset images.")

    # Standalone args
    parser.add_argument("--sat_tif",          help="Path to satellite GeoTIFF")
    parser.add_argument(
        "--image_dir",        help="[standalone] Path to drone image folder")
    parser.add_argument(
        "--csv",              help="[standalone] Path to GT CSV")
    parser.add_argument("--n",    type=int,
                        help="[standalone] Max images to process")
    parser.add_argument("--search_radius",    type=int,   default=1200)
    parser.add_argument("--min_inliers",      type=int,   default=15)
    parser.add_argument("--min_inlier_ratio", type=float, default=0.20)
    parser.add_argument(
        "--cam_config", help="[standalone] Path to VINS camera yaml (KANNALA_BRANDT)")
    parser.add_argument("--undistort_balance", type=float, default=0.5)
    parser.add_argument("--use_similarity_transform", action="store_true",
                        help="Use a similarity transform (rotate+scale+shift only) instead of a full homography")
    parser.add_argument(
        "--visualize_frame", help="Filename of a single frame to visualize matches for (saves a PNG)")
    parser.add_argument("--visualize_out", default="/tmp/match_visualization.png",
                        help="Where to save the visualization")
    parser.add_argument(
        "--vio_csv", help="Path to recorded VIO track CSV (timestamp,lat,lon) — centers crop on VIO instead of GT")

    if ROS_AVAILABLE:
        clean_argv = rospy.myargv(argv=sys.argv)[1:]
    else:
        clean_argv = sys.argv[1:]

    args = parser.parse_args(clean_argv)

    if args.mode == "standalone":
        if not args.sat_tif or not args.image_dir:
            parser.error(
                "--sat_tif and --image_dir are required in standalone mode")
        run_standalone(args)

    else:
        if not ROS_AVAILABLE:
            print("ERROR: ROS not available. Run with --mode standalone for testing.")
            sys.exit(1)
        node = SatelliteCorrectionNode()
        try:
            node.spin()
        except rospy.ROSInterruptException:
            pass
        finally:
            node.shutdown()


if __name__ == "__main__":
    main()
