# satellite-aided-vio

Satellite-image-based drift correction for GPS-denied drone navigation. Corrects VIO drift by matching drone camera frames against preloaded satellite imagery, with a confidence gate to reject unreliable corrections before they touch the pose estimate.

---

## Overview

GPS-denied UAV navigation relies on Visual-Inertial Odometry (VIO) to estimate position using only a monocular camera and IMU. VIO gives accurate high-frequency state estimation, but it accumulates drift over time — small errors compound into large position errors over long flights. This matters most in exactly the situations where GPS isn't available to correct for it: jamming, indoor flight, urban canyons, disaster response.

This project implements a **"local estimation, global correction"** architecture:

- **VINS-Mono** handles continuous high-frequency pose estimation via visual-inertial odometry
- A **satellite-image-based correction module** periodically matches the drone's camera feed against preloaded satellite map tiles to recover a globally consistent position and correct accumulated drift

The correction module operates without GPS, using only preloaded satellite imagery stored onboard the drone's compute platform.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Drone Onboard                      │
│                                                      │
│  Monocular Camera ──┐                                │
│                     ├──► VINS-Mono ──► Pose Estimate │
│  IMU ───────────────┘         │                      │
│                               │ (periodic trigger)   │
│                               ▼                      │
│              Satellite Correction Module             │
│         ┌─────────────────────────────────┐          │
│         │ 1. Crop satellite patch around  │          │
│         │    VINS odometry estimate       │          │
│         │    (never GPS — avoids leakage) │          │
│         │ 2. Match drone image against    │          │
│         │    satellite patch (LightGlue)  │          │
│         │ 3. Confidence gate              │          │
│         │ 4. Publish corrected position   │          │
│         └─────────────────────────────────┘          │
│                               │                      │
│                               ▼                      │
│                    Corrected Global Position         │
└─────────────────────────────────────────────────────┘
```

The search patch is always centered on the VINS odometry estimate, never on ground-truth GPS — using GT as the crop anchor would leak the answer into the search and invalidate the evaluation.

---

## Status

- [x] ORB, SIFT, LightGlue benchmarks on UAV-VisLoc (Scenes 03 & 04)
- [x] Confidence gate for reliable correction filtering
- [x] Visualization tooling for match analysis
- [x] Real-world validation on helicopter flight data (Bell412 / MUN-FRL)
- [x] Baseline VIO drift quantified (no correction) via `evo` ATE
- [x] Standalone satellite matcher tuned and validated on real flight frames
- [x] ROS package scaffolding + node built and running inside VINS-Mono container
- [ ] Gradient-based frame filtering to skip featureless terrain (in progress)
- [ ] Resolve LightGlue/PyTorch threading conflict with VINS-Mono's failure detector
- [ ] End-to-end closed-loop test: corrected ATE vs. baseline on Bell412
- [ ] Real camera feed testing with live satellite correction on Jetson

---

## UAV-VisLoc Benchmark (synthetic-style drone dataset)

Three feature matching approaches were benchmarked on [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc):

| Matcher | Mean Error | Median Error | Success Rate | Notes |
|---|---|---|---|---|
| ORB + BF | 109.3 m | 108.6 m | 100% | Low inlier ratio (0.096); matches mostly wrong |
| SIFT + FLANN (ratio=0.85) | 81.9 m | 69.8 m | 100% | Better descriptors; 25% improvement over ORB |
| **LightGlue + SuperPoint** | **96.6 m** | **29.4 m** | 95% | Bimodal — excellent or catastrophic |

> Mean and median diverge sharply for LightGlue because of occasional catastrophic failures on low-texture scenes (roads, rivers, featureless fields). This motivated the confidence gate below.

### Confidence Gate

Without filtering, LightGlue occasionally fits a homography to entirely wrong correspondences, producing position errors in the thousands of metres. Injecting a wrong correction into VINS-Mono is far more damaging than skipping a correction entirely, so the gate is a hard prerequisite, not a tunable extra:

- **Minimum inlier count:** 15
- **Minimum inlier ratio:** 0.20

**Gated results, Scene 03 (768 images):**

| Metric | All Matches | Gate-Accepted (19.4%) |
|---|---|---|
| Mean error | 179.8 m | **19.0 m** |
| Median error | 97.2 m | **18.1 m** |
| Std | 625.0 m | **9.4 m** |
| Max error | 11819.8 m | **57.0 m** |

**Gated results, Scene 04 (738 images):**

| Metric | All Matches | Gate-Accepted (44.2%) |
|---|---|---|
| Mean error | 248.1 m | **36.9 m** |
| Median error | 57.4 m | **32.4 m** |
| Std | 2147.4 m | **32.5 m** |
| Max error | 53933.1 m | **381.8 m** |

The gate accepts roughly 1 in 5 correction attempts on Scene 03 and nearly 1 in 2 on Scene 04, delivering consistent sub-40m corrections while rejecting catastrophic failures entirely.

---

## Real Helicopter Flight Validation (Bell412 / MUN-FRL)

Moved from the UAV-VisLoc benchmark to real flight data: `bell412_dataset6.bag`, a ~520s Bell412 helicopter flight over an Ottawa airfield from the MUN-FRL dataset. (An earlier DJI M600/Quarry1 sequence was rejected — too short, low altitude, and limited texture variety.)

**VINS-Mono configuration.** Fisheye camera model (Kannala-Brandt) calibrated from the flight's intrinsics, with the airframe's down-facing camera extrinsic and tuned IMU noise parameters.

**Ground truth.** The bag's `/fix` GPS topic is corrupted (known issue, documented in the MUN-FRL paper — it shows jagged spikes). Ground truth instead comes from RTKLib PPK post-processing of the raw GPS log, fully converged to ~2–4cm uncertainty across the flight, with GPS-time-to-UTC leap-second correction applied and verified.

**Baseline drift (no correction).** Running VINS-Mono standalone on the full flight and comparing against PPK ground truth with `evo`:

| Metric | SE3 ATE | Sim3 ATE |
|---|---|---|
| RMSE | 135.7 m | — |
| Mean | 120.7 m | 119.3 m |
| Max | 341.1 m | — |

SE3 and Sim3 error are nearly identical, which confirms the drift is genuine trajectory shape/direction error rather than a scale problem — corrections need to fix heading and position, not rescale the trajectory. VINS's estimated path length (4216m) covers about 86% of the ground-truth path length (4901m), and drift worsens noticeably toward the end of the flight, especially through the final turn before landing.

**Satellite tile.** A 6000×6000px GeoTIFF (~0.44m/px) covering a padded bounding box around the flight path, sized to 1.5× the maximum measured baseline drift, was prepared as the correction reference.

**Standalone matcher results.** Running the matcher standalone (outside ROS) against extracted flight frames surfaced a new failure mode not present in the UAV-VisLoc benchmark: much of the Ottawa airfield is low-texture farmland and runway, so LightGlue frequently returns only 0–5 matches and the inlier count sits at the RANSAC minimum — an initial pass on 50 unfiltered color frames yielded 0/50 accepted corrections.

After tuning the matcher for this environment:

- Reduced satellite search radius from 1200px → 300px (the single biggest fix — went from 0 to 17 accepted frames)
- Switched to aspect-ratio-preserving resize instead of a forced square resize
- Switched from 8-DOF homography to a similarity transform (`cv2.estimateAffinePartial2D`, min 10 inliers) — more appropriate given the near-planar, near-nadir geometry

Result on 533 candidate frames: **28 accepted** (mean geo-error 113.4m across accepted frames, 111 frames under 50m error). The strongest cluster, around t≈807–808s over airfield structures, hit 134–241 inliers with 35–40m geo-error, visually confirmed correct.

**Current blocker: featureless terrain.** The dominant failure mode is low-texture ground cover, not fisheye distortion. The fix in progress is gradient-based pre-filtering of candidate frames — a gradient-score calibration pass across the flight to pick a texture threshold, then dense frame extraction filtered to that threshold, before re-running the matcher.

---

## ROS Integration

A `satellite_aided_vio` catkin package is built and running inside the VINS-Mono Docker container. The correction node subscribes to the mono camera topic and computes gradient-based texture scores on incoming frames (handles both 2D and 3D inputs). The scene's local origin is derived from the first VINS odometry timestamp, converted to UTC with the leap-second offset, and matched against the PPK ground-truth file.

Resolved along the way: `argparse` conflicting with ROS's own remapping arguments (fixed via `rospy.myargv()`), `use_sim_time` needing to be set before `roslaunch` rather than after, and topic remap syntax for feeding in a renamed camera topic.

**Open issue:** PyTorch's multi-threaded CPU inference during LightGlue calls monopolizes CPU cores, which triggers VINS-Mono's own failure-detection heuristics. Needs thread-limiting or throttling before the two can run concurrently.

---

## Datasets

- [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) — large-scale UAV visual localization dataset with drone images, georeferenced satellite TIF maps, and GPS ground truth. Used for the initial matcher benchmark (Scenes 03 & 04).
- **Bell412 / MUN-FRL** — real helicopter flight data over an Ottawa airfield, used for real-world VIO drift measurement and matcher validation.

---

## Repository Structure

```
satellite-aided-vio/
├── benchmark_orb.py               # ORB + BF matcher benchmark (UAV-VisLoc)
├── benchmark_sift.py              # SIFT + FLANN matcher benchmark (UAV-VisLoc)
├── benchmark_lightglue.py         # LightGlue + SuperPoint benchmark with confidence gate
├── visualize_matches.py           # ORB match visualizer
├── visualize_matches_lightglue.py # LightGlue match visualizer (best/worst/random modes)
├── rectify.py                     # Drone image pre-rectification using pitch/roll angles
├── app_v1.py                      # Early GCS demo with satellite map overlay
├── satellite_correction_node.py   # SatelliteMatcher class + ROS node + standalone mode
├── dataset_publisher_node.py      # Publishes UAV-VisLoc as a live ROS feed
├── gt_path_publisher_pos.py       # Publishes PPK .pos ground truth as nav_msgs/Path
├── extract_paths_to_tum.py        # Extracts recorded bag paths to TUM format for evo
├── compute_ate.py                 # SE3 + Sim3 ATE computation via evo
├── compute_flight_bbox.py         # Computes padded GPS bounding box from .pos file
├── extract_bag_frames.py          # Extracts frames from a bag with GT + gradient filtering
├── bell412_config.yaml            # VINS-Mono config for Bell412 sequences
└── assets/
    └── demo.png                   # Sample LightGlue match visualization
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/PurveshGhedia/satellite-aided-vio.git
cd satellite-aided-vio

# Install dependencies
pip install opencv-python rasterio numpy lightglue evo pymap3d
pip install git+https://github.com/cvg/LightGlue.git
```

---

## Usage

**Run LightGlue benchmark on a UAV-VisLoc scene:**
```bash
python benchmark_lightglue.py \
  --dataset /path/to/UAV_VisLoc_dataset \
  --scene 03 \
  --min_inliers 15 \
  --min_inlier_ratio 0.20 \
  --output results.csv
```

**Visualize best/worst matches:**
```bash
python visualize_matches_lightglue.py \
  --dataset /path/to/UAV_VisLoc_dataset \
  --results results.csv \
  --scene 03 \
  --mode best \
  --n 5
```

**Compute baseline ATE against PPK ground truth:**
```bash
python compute_ate.py \
  --est vins_path.tum \
  --gt bell412_dataset6_frl.pos \
  --t_max_diff 0.1
```

---

## Hardware

Developed and tested on:
- MacBook Pro M1 (development, benchmarking) — VINS-Mono in Docker (ROS1 Noetic, AMD64 emulation via Rosetta)
- NVIDIA Jetson Orin Nano Super (target edge deployment / flight compute)
- Raspberry Pi 5 + Raspberry Pi Camera
- SmartElex 9DoF IMU Breakout (ISM330DHCX + MMC5983MA)

---

## References

- [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) — Robust and Versatile Monocular Visual-Inertial State Estimator
- [LightGlue](https://github.com/cvg/LightGlue) — Local Feature Matching at Light Speed
- [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) — A Large-scale Dataset for UAV Visual Localization
- [FoundLoc](https://arxiv.org/abs/2310.16299) — Foundation Model-based Indoor Localization (CMU)