#!/usr/bin/env python3
"""
dataset_publisher_node.py
--------------------------
Replays a UAV-VisLoc scene as a live ROS feed, turning the static dataset
into a simulated drone flying the scene in order.

This is the missing piece between your benchmarked satellite correction module
and a testable ROS pipeline. Once this node is running alongside
satellite_correction_node, you have a complete closed-loop test without
needing a real drone or a rosbag.

Publications
------------
    /drone/image            sensor_msgs/Image
        Drone camera frames in BGR8 encoding, published at ~publish_hz.

    /drone/gps_gt           sensor_msgs/NavSatFix
        Ground-truth GPS position from dataset CSV. Published in sync with
        each image. The satellite_correction_node can subscribe to this
        instead of /vins_estimator/odometry during testing (Option B from
        the correction node comments) — this eliminates the ENU->WGS84
        conversion issue entirely while you don't yet have real VIO running.

    /drone/camera_info      sensor_msgs/CameraInfo
        Basic camera intrinsics estimated from FOV. Needed by VINS-Mono.

    /drone/imu              sensor_msgs/Imu
        Synthetic IMU at imu_hz. Populated with zeros — enough to keep
        VINS-Mono's subscriber alive. Replace with real IMU data when you
        have a rosbag.

    /dataset/progress       std_msgs/String  (JSON)
        Per-frame metadata: filename, lat, lon, pitch, roll, frame index,
        total frames, elapsed time. Useful for Foxglove monitoring.

Parameters (ROS params)
-----------------------
    ~scene_dir      str     Path to scene folder (e.g. .../UAV_VisLoc/03)
    ~publish_hz     float   Image publish rate in Hz (default: 1.0)
                            1 Hz is realistic for a satellite correction
                            trigger interval. Use 0.0 to publish as fast
                            as possible (smoke test mode).
    ~imu_hz         float   Synthetic IMU rate in Hz (default: 200.0)
    ~loop           bool    Loop the dataset after last frame (default: False)
    ~start_index    int     Skip to this frame index on startup (default: 0)
    ~resize         int     Resize drone images to square before publishing
                            (default: 0 = no resize, publish original size)
    ~image_encoding str     ROS image encoding: bgr8 or mono8 (default: bgr8)

Standalone mode (no ROS)
------------------------
    python dataset_publisher_node.py \
        --mode standalone \
        --scene_dir ~/data/UAV_VisLoc_dataset/03 \
        --n 10

    Prints frame metadata to stdout. Useful for checking CSV parsing and
    image loading before starting Docker.

Usage in launch file
--------------------
    <node pkg="satellite_aided_vio" type="dataset_publisher_node.py"
          name="dataset_publisher" output="screen">
        <param name="scene_dir"   value="/data/UAV_VisLoc_dataset/03"/>
        <param name="publish_hz"  value="1.0"/>
        <param name="loop"        value="false"/>
    </node>
"""

import sys
import json
import time
import argparse
import csv
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

try:
    import rospy
    from sensor_msgs.msg import Image, NavSatFix, NavSatStatus, CameraInfo, Imu
    from std_msgs.msg import String, Header
    from cv_bridge import CvBridge
    import geometry_msgs.msg
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


# ===========================================================================
# Dataset loader — pure Python, no ROS
# ===========================================================================

@dataclass
class SceneFrame:
    """One row of the UAV-VisLoc CSV, with image path resolved."""
    index:      int
    filename:   str
    image_path: Path
    lat:        float
    lon:        float
    altitude:   float      # metres, from CSV if available else 0
    pitch_deg:  float      # Omega
    roll_deg:   float      # Kappa
    yaw_deg:    float      # Phi


class SceneLoader:
    """
    Loads a UAV-VisLoc scene directory and provides ordered frames.

    UAV-VisLoc CSV columns (Scene 03 confirmed):
        filename, lat, lon, altitude, Omega (pitch), Kappa (roll), Phi (yaw)

    Column names vary slightly between UAV-VisLoc versions. The loader
    tries known variants and falls back gracefully.
    """

    # Column name aliases — (preferred, fallback...)
    _LAT_COLS = ["lat", "latitude", "Lat"]
    _LON_COLS = ["lon", "longitude", "Lon"]
    _ALT_COLS = ["altitude", "alt", "Alt", "height"]
    _PITCH_COLS = ["Omega", "omega", "pitch"]
    _ROLL_COLS = ["Kappa", "kappa", "roll"]
    _YAW_COLS = ["Phi", "phi", "yaw"]

    def __init__(self, scene_dir: str):
        self.scene_dir = Path(scene_dir)
        self.drone_dir = self.scene_dir / "drone"
        self.frames: List[SceneFrame] = []
        self._load()

    def _pick(self, row: dict, candidates: list, default: float = 0.0) -> float:
        for col in candidates:
            if col in row and row[col].strip():
                return float(row[col])
        return default

    def _load(self):
        csv_files = list(self.scene_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV found in {self.scene_dir}")
        csv_path = csv_files[0]

        frames = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                fname = row["filename"].strip()
                img_path = self.drone_dir / fname
                if not img_path.exists():
                    # Try case-insensitive JPG/jpg
                    alt = self.drone_dir / fname.replace(".JPG", ".jpg")
                    if alt.exists():
                        img_path = alt
                    else:
                        continue  # skip missing images silently

                frames.append(SceneFrame(
                    index=i,
                    filename=fname,
                    image_path=img_path,
                    lat=self._pick(row, self._LAT_COLS),
                    lon=self._pick(row, self._LON_COLS),
                    altitude=self._pick(row, self._ALT_COLS, default=0.0),
                    pitch_deg=self._pick(row, self._PITCH_COLS),
                    roll_deg=self._pick(row, self._ROLL_COLS),
                    yaw_deg=self._pick(row, self._YAW_COLS),
                ))

        self.frames = frames
        print(
            f"[SceneLoader] Loaded {len(frames)} frames from {csv_path.name}")
        if frames:
            print(f"[SceneLoader] First: {frames[0].filename}  "
                  f"lat={frames[0].lat:.6f}  lon={frames[0].lon:.6f}")
            print(f"[SceneLoader] Last : {frames[-1].filename}  "
                  f"lat={frames[-1].lat:.6f}  lon={frames[-1].lon:.6f}")

    def load_image(self, frame: SceneFrame, resize: int = 0) -> Optional[np.ndarray]:
        img = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if resize > 0:
            img = cv2.resize(img, (resize, resize))
        return img


# ===========================================================================
# ROS node
# ===========================================================================

class DatasetPublisherNode:
    """
    Publishes UAV-VisLoc frames as ROS topics at a configurable rate.

    The image callback drives everything — a rospy.Timer fires at
    publish_hz and publishes the next frame. A separate faster Timer
    handles synthetic IMU at imu_hz.

    All publishers share the same timestamp (rospy.Time.now()) per frame,
    so image and NavSatFix are time-aligned for the correction node.
    """

    def __init__(self):
        rospy.init_node("dataset_publisher_node", anonymous=False)

        scene_dir = rospy.get_param("~scene_dir")
        publish_hz = rospy.get_param("~publish_hz",     1.0)
        self.imu_hz = rospy.get_param("~imu_hz",       200.0)
        self.loop = rospy.get_param("~loop",          False)
        start_index = rospy.get_param("~start_index",      0)
        self.resize = rospy.get_param("~resize",            0)
        self.encoding = rospy.get_param("~image_encoding", "bgr8")

        self.loader = SceneLoader(scene_dir)
        self.frames = self.loader.frames[start_index:]
        self._idx = 0
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._done = False
        self._t0 = None

        if not self.frames:
            rospy.logerr("[DataPub] No frames loaded. Check scene_dir.")
            return

        # Publishers
        self._pub_image = rospy.Publisher(
            "/drone/image",       Image,       queue_size=2)
        self._pub_gps = rospy.Publisher(
            "/drone/gps_gt",      NavSatFix,   queue_size=5)
        self._pub_info = rospy.Publisher(
            "/drone/camera_info", CameraInfo,  queue_size=5)
        self._pub_imu = rospy.Publisher(
            "/drone/imu",         Imu,         queue_size=10)
        self._pub_prog = rospy.Publisher(
            "/dataset/progress",  String,      queue_size=10)

        # Timers
        if publish_hz > 0:
            period = 1.0 / publish_hz
        else:
            period = 0.001  # as fast as possible
        rospy.Timer(rospy.Duration(period),       self._image_timer_cb)
        rospy.Timer(rospy.Duration(1.0/self.imu_hz), self._imu_timer_cb)

        rospy.loginfo(
            "[DataPub] Ready. %d frames at %.1f Hz. Loop=%s",
            len(self.frames), publish_hz, self.loop
        )

    # ------------------------------------------------------------------
    # Image + GPS timer
    # ------------------------------------------------------------------

    def _image_timer_cb(self, event):
        with self._lock:
            if self._done:
                return
            idx = self._idx

        if idx >= len(self.frames):
            if self.loop:
                with self._lock:
                    self._idx = 0
                    idx = 0
            else:
                if not self._done:
                    rospy.loginfo(
                        "[DataPub] All frames published. Shutting down.")
                    with self._lock:
                        self._done = True
                    rospy.signal_shutdown("Dataset exhausted")
                return

        frame = self.frames[idx]
        now = rospy.Time.now()

        if self._t0 is None:
            self._t0 = time.time()

        img = self.loader.load_image(frame, resize=self.resize)
        if img is None:
            rospy.logwarn(
                "[DataPub] Could not load %s, skipping.", frame.filename)
            with self._lock:
                self._idx += 1
            return

        # Publish image
        if self.encoding == "mono8":
            img_pub = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_pub = img

        try:
            img_msg = self._bridge.cv2_to_imgmsg(
                img_pub, encoding=self.encoding)
        except Exception as e:
            rospy.logwarn("[DataPub] cv2_to_imgmsg failed: %s", e)
            with self._lock:
                self._idx += 1
            return

        img_msg.header.stamp = now
        img_msg.header.frame_id = "drone_camera"
        self._pub_image.publish(img_msg)

        # Publish camera info
        self._pub_info.publish(self._make_camera_info(now, img.shape))

        # Publish NavSatFix (ground truth GPS)
        gps_msg = NavSatFix()
        gps_msg.header.stamp = now
        gps_msg.header.frame_id = "drone_gps"
        gps_msg.status.status = NavSatStatus.STATUS_FIX
        gps_msg.status.service = NavSatStatus.SERVICE_GPS
        gps_msg.latitude = frame.lat
        gps_msg.longitude = frame.lon
        gps_msg.altitude = frame.altitude
        # Covariance: ~5m GPS accuracy (diagonal, metres^2)
        gps_msg.position_covariance = [25.0, 0, 0, 0, 25.0, 0, 0, 0, 25.0]
        gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self._pub_gps.publish(gps_msg)

        # Publish progress
        elapsed = time.time() - self._t0
        prog = {
            "frame_index":  idx,
            "total_frames": len(self.frames),
            "filename":     frame.filename,
            "lat":          frame.lat,
            "lon":          frame.lon,
            "altitude_m":   frame.altitude,
            "pitch_deg":    frame.pitch_deg,
            "roll_deg":     frame.roll_deg,
            "yaw_deg":      frame.yaw_deg,
            "elapsed_s":    round(elapsed, 1),
        }
        self._pub_prog.publish(String(data=json.dumps(prog)))

        rospy.loginfo(
            "[DataPub] Frame %d/%d | %s | lat=%.6f lon=%.6f",
            idx + 1, len(self.frames), frame.filename, frame.lat, frame.lon
        )

        with self._lock:
            self._idx += 1

    # ------------------------------------------------------------------
    # Synthetic IMU timer
    # ------------------------------------------------------------------

    def _imu_timer_cb(self, event):
        """
        Publish a zero-filled IMU message to keep VINS-Mono's IMU subscriber
        alive. VINS-Mono requires IMU data before it will initialize.

        For real testing you'd replace this with actual IMU data from a rosbag
        or from a hardware interface. For pipeline smoke-testing (does the
        graph connect? do topics flow?) this is sufficient.
        """
        now = rospy.Time.now()
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "imu"

        # Identity orientation (unknown)
        imu.orientation.w = 1.0
        imu.orientation_covariance[0] = -1  # signals "unknown"

        # Zero angular velocity and linear acceleration
        # (In production: read from dataset or rosbag)
        imu.angular_velocity_covariance[0] = -1
        imu.linear_acceleration_covariance[0] = -1

        self._pub_imu.publish(imu)

    # ------------------------------------------------------------------
    # Camera info
    # ------------------------------------------------------------------

    def _make_camera_info(self, stamp, img_shape) -> "CameraInfo":
        """
        Publish basic camera intrinsics estimated from FOV.
        For VINS-Mono you will eventually need calibrated values from Kalibr,
        but this is sufficient for pipeline connectivity testing.
        """
        h, w = img_shape[:2]
        hfov_deg = 84.0
        import math
        fx = (w / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        fy = fx
        cx = w / 2.0
        cy = h / 2.0

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = "drone_camera"
        info.width = w
        info.height = h
        info.distortion_model = "plumb_bob"
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.K = [fx,  0,  cx,
                  0, fy,  cy,
                  0,  0,   1]
        info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        info.P = [fx,  0, cx, 0,
                  0, fy, cy, 0,
                  0,  0,  1, 0]
        return info

    def spin(self):
        rospy.spin()


# ===========================================================================
# Standalone mode
# ===========================================================================

def run_standalone(args):
    """
    Print frame metadata to stdout without ROS.
    Use this to verify CSV parsing and image loading before starting Docker.
    """
    loader = SceneLoader(args.scene_dir)
    frames = loader.frames
    if args.n:
        frames = frames[:args.n]

    print(f"\n{'='*70}")
    print(f"{'idx':>4}  {'filename':<30}  {'lat':>10}  {'lon':>11}  "
          f"{'pitch':>7}  {'roll':>7}  {'img_ok'}")
    print(f"{'='*70}")

    for frame in frames:
        img = loader.load_image(frame)
        img_ok = img is not None
        shape = f"{img.shape[1]}x{img.shape[0]}" if img_ok else "MISSING"
        print(
            f"{frame.index:>4}  {frame.filename:<30}  "
            f"{frame.lat:>10.6f}  {frame.lon:>11.6f}  "
            f"{frame.pitch_deg:>7.2f}  {frame.roll_deg:>7.2f}  "
            f"{shape}"
        )

    print(f"{'='*70}")
    print(f"Total frames in CSV: {len(loader.frames)}")
    if args.n and args.n < len(loader.frames):
        print(f"(showing first {args.n})")
    print()


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="UAV-VisLoc dataset publisher — ROS or standalone")
    parser.add_argument(
        "--mode", choices=["ros", "standalone"], default="ros")

    parser.add_argument(
        "--scene_dir", help="Path to scene directory (e.g. .../03)")
    parser.add_argument(
        "--n", type=int, help="[standalone] Max frames to print")

    args = parser.parse_args()

    if args.mode == "standalone":
        if not args.scene_dir:
            parser.error("--scene_dir is required in standalone mode")
        run_standalone(args)

    else:
        if not ROS_AVAILABLE:
            print("ERROR: ROS not available. Run with --mode standalone.")
            sys.exit(1)
        node = DatasetPublisherNode()
        try:
            node.spin()
        except rospy.ROSInterruptException:
            pass


if __name__ == "__main__":
    main()
