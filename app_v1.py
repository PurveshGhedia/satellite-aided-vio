import sys
import os
import math
import json
import random
import numpy as np
import cv2
import rasterio
from rasterio.warp import transform
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QGraphicsView, QGraphicsScene, QFileDialog,
    QMessageBox, QLabel, QDoubleSpinBox, QProgressBar,
    QGraphicsRectItem, QGraphicsEllipseItem, QFrame, QTextEdit, QGroupBox,
    QSlider
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor,
    QBrush, QPainterPath, QTransform
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal

# --- CONFIGURATION ---
CAMERA_FOV_DEG = 95.0
OVERLAP_PERCENT = 0.30
DATA_DIR = "data"
MAP_FILE = os.path.join(os.path.dirname(__file__), "mahajan_map.tif")

# --- STYLESHEET ---
DARK_STYLESHEET = """
QMainWindow { background-color: #2b2b2b; }
QWidget { color: #e0e0e0; font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 14px; }
QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; color: #8ab4f8; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QPushButton { background-color: #3c4043; border: 1px solid #5f6368; border-radius: 4px; padding: 8px; min-height: 25px; }
QPushButton:hover { background-color: #484c50; border-color: #8ab4f8; }
QPushButton:pressed { background-color: #202124; }
QPushButton#PrimaryBtn { background-color: #1a73e8; border: none; color: white; font-weight: bold; }
QPushButton#PrimaryBtn:hover { background-color: #1557b0; }
QPushButton#SuccessBtn { background-color: #1e8e3e; border: none; color: white; font-weight: bold; }
QPushButton#SuccessBtn:hover { background-color: #137333; }
QPushButton#WarningBtn { background-color: #e37400; border: none; color: white; font-weight: bold; }
QPushButton#WarningBtn:hover { background-color: #c26400; }
QDoubleSpinBox { background-color: #202124; border: 1px solid #5f6368; border-radius: 4px; padding: 5px; color: white; }
QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; background-color: #202124; }
QProgressBar::chunk { background-color: #1a73e8; width: 10px; }
QTextEdit { background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px; color: #aaaaaa; font-family: 'Consolas', monospace; font-size: 12px; }
QGraphicsView { border: none; background-color: #181818; }
QLabel#Header { font-size: 18px; font-weight: bold; color: white; margin-bottom: 10px; }

/* SLIDER STYLING */
QSlider::groove:vertical {
    border: 1px solid #333;
    width: 8px; /* Narrow groove */
    background: #202124;
    margin: 0px 10px;
    border-radius: 4px;
}
QSlider::handle:vertical {
    background: #8ab4f8;
    border: 1px solid #5f6368;
    height: 16px;
    width: 24px;
    margin: 0 -9px; /* expand outside groove */
    border-radius: 8px;
}
QSlider::handle:vertical:hover {
    background: #aecbfa;
}
"""


class ClickableDot(QGraphicsEllipseItem):
    def __init__(self, x, y, r, parent_window):
        super().__init__(x - r, y - r, r * 2, r * 2)
        self.setBrush(QBrush(QColor(0, 255, 0)))
        self.setPen(QPen(Qt.black))
        self.setAcceptHoverEvents(True)
        self.center_x = x
        self.center_y = y
        self.parent_window = parent_window

    def mousePressEvent(self, event):
        self.parent_window.generate_homography_popup(
            self.center_x, self.center_y)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor(100, 255, 100)))

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(QColor(0, 255, 0)))


def solve_location_static(img_map, img_drone):
    orb = cv2.ORB_create(nfeatures=2000)
    kp_map, des_map = orb.detectAndCompute(img_map, None)
    kp_drone, des_drone = orb.detectAndCompute(img_drone, None)

    if des_map is None or des_drone is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_drone, des_map)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.15)]

    if len(good_matches) < 4:
        return None

    src_pts = np.float32(
        [kp_drone[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        return None

    h, w = img_drone.shape
    drone_center = np.array([[[w / 2, h / 2]]], dtype=np.float32)
    try:
        mapped_center = cv2.perspectiveTransform(drone_center, M)
        est_x, est_y = mapped_center[0][0]
        return ((est_x, est_y), M, img_map, img_drone, good_matches, kp_map, kp_drone)
    except:
        return None

# --- MAIN UI ---


class TacticalMapView(QGraphicsView):
    # Signal to report zoom level back to slider (0.0 to 2.0 range usually)
    zoomChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(24, 24, 24)))
        self._current_scale = 1.0

    def wheelEvent(self, event):
        # Standard Zoom Logic
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(zoom_factor, zoom_factor)

        # Track current total scale to sync with slider
        self._current_scale *= zoom_factor
        self.zoomChanged.emit(self._current_scale)

    def set_absolute_scale(self, scale_value):
        # Reset transform to identity then apply absolute scale
        # This prevents drift and allows the slider to be absolute
        self.resetTransform()
        self.scale(scale_value, scale_value)
        self._current_scale = scale_value


class VinsGroundControl(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VINS-Fusion Ground Control Station")
        self.resize(1280, 800)
        self.setStyleSheet(DARK_STYLESHEET)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet(
            "background-color: #252526; border-right: 1px solid #3e3e3e;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)
        sidebar_layout.setSpacing(15)

        header = QLabel("NCNC DEMO GCS")
        header.setObjectName("Header")
        header.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(header)

        # Setup
        grp_setup = QGroupBox("SYSTEM SETUP")
        lay_setup = QVBoxLayout()
        self.btn_load_mission = QPushButton("Load Mission File")
        self.btn_load_mission.setObjectName("PrimaryBtn")
        self.btn_load_mission.clicked.connect(self.load_mission_file)
        lay_setup.addWidget(self.btn_load_mission)
        grp_setup.setLayout(lay_setup)
        sidebar_layout.addWidget(grp_setup)

        # Config
        grp_params = QGroupBox("SENSOR CONFIG")
        lay_params = QVBoxLayout()
        lbl_alt = QLabel("Flight Altitude (AGL):")
        self.alt_input = QDoubleSpinBox()
        self.alt_input.setRange(5.0, 500.0)
        self.alt_input.setValue(50.0)
        self.alt_input.setSuffix(" m")
        self.alt_input.valueChanged.connect(self.refresh_path)
        lay_params.addWidget(lbl_alt)
        lay_params.addWidget(self.alt_input)
        grp_params.setLayout(lay_params)
        sidebar_layout.addWidget(grp_params)

        # Simulation
        grp_sim = QGroupBox("SIMULATION & DEMO")
        lay_sim = QVBoxLayout()
        self.demo_btn = QPushButton("Run Live Demo Overlay")
        self.demo_btn.setObjectName("WarningBtn")
        self.demo_btn.clicked.connect(self.run_live_demo_simulation)
        lay_sim.addWidget(self.demo_btn)
        grp_sim.setLayout(lay_sim)
        sidebar_layout.addWidget(grp_sim)

        # Log
        grp_log = QGroupBox("SYSTEM LOG")
        lay_log = QVBoxLayout()
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setPlaceholderText("System Ready.")
        lay_log.addWidget(self.log_console)
        grp_log.setLayout(lay_log)
        sidebar_layout.addWidget(grp_log)

        main_layout.addWidget(sidebar)

        # --- MAP & ZOOM ---
        map_container = QWidget()
        map_layout = QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)

        self.map_view = TacticalMapView()
        # Connect Mouse Wheel -> Slider
        self.map_view.zoomChanged.connect(self.sync_slider_from_mouse)
        map_layout.addWidget(self.map_view)

        # --- SLIDER OVERLAY ---
        self.zoom_overlay = QWidget(self.map_view)
        self.zoom_overlay.setAttribute(Qt.WA_TranslucentBackground)
        self.zoom_overlay.setGeometry(20, 20, 40, 200)  # Tall and narrow

        zoom_lay = QVBoxLayout(self.zoom_overlay)
        zoom_lay.setContentsMargins(0, 0, 0, 0)

        # Vertical Slider
        self.zoom_slider = QSlider(Qt.Vertical)
        self.zoom_slider.setMinimum(10)   # 10% Zoom
        self.zoom_slider.setMaximum(500)  # 500% Zoom
        self.zoom_slider.setValue(100)    # 100% Default
        self.zoom_slider.setTickPosition(QSlider.TicksRight)
        self.zoom_slider.setTickInterval(50)

        # When Slider Moves -> Update Map
        self.zoom_slider.valueChanged.connect(self.sync_map_from_slider)

        zoom_lay.addWidget(self.zoom_slider)
        main_layout.addWidget(map_container)

        # Vars
        self.transform_matrix = None
        self.waypoints = []
        self.current_map_path = None
        self.map_pixmap_item = None
        self.load_offline_map(MAP_FILE)

    def log(self, message):
        self.log_console.append(f">> {message}")
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    # --- ZOOM SYNC LOGIC ---
    def sync_map_from_slider(self, value):
        # Value is 10-500, we need scale 0.1 - 5.0
        scale_factor = value / 100.0
        self.map_view.set_absolute_scale(scale_factor)

    def sync_slider_from_mouse(self, current_scale):
        # Current scale is float (e.g. 1.5), slider needs int (150)
        slider_val = int(current_scale * 100)

        # Block signals so we don't create an infinite loop
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(slider_val)
        self.zoom_slider.blockSignals(False)

    # --- POPUP LOGIC ---

    def generate_homography_popup(self, cx, cy):
        if not self.current_map_path:
            return
        self.log(f"Generating Analysis for Point ({int(cx)}, {int(cy)})...")
        try:
            sat_size = 800
            half_size = sat_size // 2
            with rasterio.open(self.current_map_path) as src:
                c_off = max(0, int(cx - half_size))
                r_off = max(0, int(cy - half_size))
                width = min(sat_size, src.width - c_off)
                height = min(sat_size, src.height - r_off)
                window = rasterio.windows.Window(c_off, r_off, width, height)
                data = src.read(window=window)
                if data.shape[0] >= 3:
                    img_sat_color = np.transpose(data[:3, :, :], (1, 2, 0))
                    img_sat_color = cv2.cvtColor(
                        img_sat_color, cv2.COLOR_RGB2BGR)
                    img_sat = cv2.cvtColor(img_sat_color, cv2.COLOR_BGR2GRAY)
                else:
                    img_sat = data[0]

            h_s, w_s = img_sat.shape
            drone_size = 400
            dx1 = int((w_s - drone_size) / 2)
            dy1 = int((h_s - drone_size) / 2)
            img_drone_raw = img_sat[dy1:dy1+drone_size, dx1:dx1+drone_size]

            angle = random.randint(-45, 45)
            M_rot = cv2.getRotationMatrix2D(
                (drone_size//2, drone_size//2), angle, 1.0)
            img_drone_rot = cv2.warpAffine(
                img_drone_raw, M_rot, (drone_size, drone_size))
            img_drone_final = cv2.convertScaleAbs(
                img_drone_rot, alpha=0.8, beta=-30)

            result = solve_location_static(img_sat, img_drone_final)
            if result:
                (est_x, est_y), M, _, _, matches, kp_map, kp_drone = result
                img_sat_rgb = cv2.cvtColor(img_sat, cv2.COLOR_GRAY2RGB)
                img_drone_rgb = cv2.cvtColor(
                    img_drone_final, cv2.COLOR_GRAY2RGB)
                h, w = img_drone_final.shape
                pts = np.float32(
                    [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                img_sat_box = cv2.polylines(
                    img_sat_rgb, [np.int32(dst)], True, (0, 255, 0), 5, cv2.LINE_AA)
                draw_params = dict(matchColor=(0, 255, 0),
                                   singlePointColor=None, flags=2)
                img_matches = cv2.drawMatches(
                    img_drone_rgb, kp_drone, img_sat_box, kp_map, matches, None, **draw_params)
                plt.figure(figsize=(10, 6))
                plt.imshow(img_matches)
                plt.title(
                    f"LIVE MATCH ANALYSIS\nRotation: {angle} deg | Matches: {len(matches)}")
                plt.axis('off')
                plt.show()
            else:
                self.log("Match Failed (Synthetic generation error).")
        except Exception as e:
            self.log(f"Popup Error: {e}")

    # --- SIMULATION ---
    def run_live_demo_simulation(self):
        if not self.current_map_path:
            QMessageBox.warning(self, "Error", "Map not initialized.")
            return
        path_pixels = self.generate_noisy_vins_path()
        if not path_pixels:
            return
        self.log("Running Visual Localization Demo...")
        indices_to_match = [
            int(len(path_pixels)*0.2), int(len(path_pixels)*0.5), int(len(path_pixels)*0.8)]
        for idx in indices_to_match:
            center_pt = path_pixels[idx]
            cx, cy = center_pt.x(), center_pt.y()
            dot = ClickableDot(cx, cy, 10, self)
            self.map_view.scene.addItem(dot)
            box_size = 200
            rect_item = QGraphicsRectItem(
                cx - box_size/2, cy - box_size/2, box_size, box_size)
            rect_item.setPen(QPen(Qt.green, 3))
            rect_item.setTransformOriginPoint(cx, cy)
            rect_item.setRotation(random.uniform(-20, 20))
            self.map_view.scene.addItem(rect_item)
            text = self.map_view.scene.addText(
                f"MATCH: {random.randint(92, 99)}%")
            text.setDefaultTextColor(Qt.white)
            text.setPos(cx + box_size/2, cy - box_size/2)
        self.log("Click on Green Dots to see Homography Analysis.")

    def generate_noisy_vins_path(self):
        if not self.waypoints or not self.transform_matrix:
            return []
        with rasterio.open(self.current_map_path) as ds:
            is_geographic = ds.crs.is_geographic if ds.crs else True
            path_pixels = []
            for lon, lat in self.waypoints:
                try:
                    if ds.crs and ds.crs != "EPSG:4326":
                        try:
                            xs, ys = transform(
                                "EPSG:4326", ds.crs, [lon], [lat])
                            row, col = ds.index(xs[0], ys[0])
                        except:
                            row, col = ds.index(lon, lat)
                    else:
                        row, col = ds.index(lon, lat)
                    path_pixels.append(QPointF(col, row))
                except:
                    continue

        if len(path_pixels) < 2:
            return []
        vins_path = QPainterPath()
        vins_path.moveTo(path_pixels[0])
        for i in range(len(path_pixels) - 1):
            p1 = path_pixels[i]
            p2 = path_pixels[i+1]
            steps = 8
            for j in range(1, steps + 1):
                f = j / steps
                lx = p1.x() + (p2.x() - p1.x()) * f
                ly = p1.y() + (p2.y() - p1.y()) * f
                vins_path.lineTo(
                    QPointF(lx + random.uniform(-10, 10), ly + random.uniform(-10, 10)))
        vins_pen = QPen(QColor(0, 191, 255))
        vins_pen.setWidth(2)
        self.map_view.scene.addPath(vins_path, vins_pen)
        return path_pixels

    def load_offline_map(self, tif_path):
        try:
            self.current_map_path = tif_path
            pixmap = QPixmap(tif_path)
            if pixmap.isNull():
                return
            self.map_view.scene.clear()
            self.map_pixmap_item = self.map_view.scene.addPixmap(pixmap)
            self.map_view.scene.setSceneRect(
                0, 0, pixmap.width(), pixmap.height())
            try:
                with rasterio.open(tif_path) as dataset:
                    self.transform_matrix = dataset.transform
            except:
                self.transform_matrix = rasterio.Affine(
                    1.0, 0.0, 0.0, 0.0, -1.0, pixmap.height())
        except Exception as e:
            QMessageBox.critical(self, "Map Error", f"{e}")

    def load_mission_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Mission", "", "Files (*.mission *.plan *.json)")
        if fname:
            self.waypoints = self.parse_json(fname)
            self.refresh_path()

    def parse_json(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            items = []
            if "mission" in data and "items" in data["mission"]:
                items = data["mission"]["items"]
            elif "items" in data:
                items = data["items"]
            points = []
            for item in items:
                if item.get("command") in [16, 21, 22] and len(item.get("params", [])) >= 7:
                    lat = float(item["params"][4])
                    lon = float(item["params"][5])
                    if abs(lat) > 0.1 and abs(lon) > 0.1:
                        points.append((lon, lat))
            return points
        except:
            return []

    def refresh_path(self):
        if not self.waypoints or not self.current_map_path:
            return

        # Clear previous overlays but keep map
        for item in self.map_view.scene.items():
            if item != self.map_pixmap_item:
                self.map_view.scene.removeItem(item)

        path_pixels = []

        try:
            with rasterio.open(self.current_map_path) as ds:

                # ---- Convert GPS → Pixel ----
                for lon, lat in self.waypoints:
                    try:
                        if ds.crs and ds.crs != "EPSG:4326":
                            xs, ys = transform(
                                "EPSG:4326", ds.crs, [lon], [lat])
                            row, col = ds.index(xs[0], ys[0])
                        else:
                            row, col = ds.index(lon, lat)

                        path_pixels.append(QPointF(col, row))
                    except:
                        continue

                if len(path_pixels) < 2:
                    return

                # ---- Compute Swath Size in Pixels ----
                alt = self.alt_input.value()
                fov = math.radians(CAMERA_FOV_DEG)
                swath_meters = 2 * alt * math.tan(fov / 2)

                res_x = abs(ds.transform[0])
                is_geographic = ds.crs.is_geographic if ds.crs else True

                if is_geographic:
                    meters_per_pixel = res_x * 111320.0
                else:
                    meters_per_pixel = res_x

                if meters_per_pixel < 0.001:
                    meters_per_pixel = 1.0

                swath_px = swath_meters / meters_per_pixel
                step_px = swath_px * (1.0 - OVERLAP_PERCENT)

        except Exception as e:
            self.log(f"Map Read Error: {e}")
            return

        # ---- Draw Flight Path ----
        path = QPainterPath()
        path.moveTo(path_pixels[0])
        for pt in path_pixels[1:]:
            path.lineTo(pt)

        pen = QPen(QColor(255, 215, 0))
        pen.setWidth(3)
        self.map_view.scene.addPath(path, pen)

        # ---- Improved Rotated Tile Placement ----
        box_pen = QPen(QColor(255, 50, 50, 150))
        box_pen.setWidth(2)

        for i in range(len(path_pixels) - 1):
            p1 = path_pixels[i]
            p2 = path_pixels[i + 1]

            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            segment_length = math.hypot(dx, dy)

            if segment_length == 0:
                continue

            heading = math.degrees(math.atan2(dy, dx))
            steps = max(1, int(segment_length / step_px))

            for j in range(steps + 1):
                t = j / steps
                cx = p1.x() + dx * t
                cy = p1.y() + dy * t

                rect = QGraphicsRectItem(
                    cx - swath_px / 2,
                    cy - swath_px / 2,
                    swath_px,
                    swath_px
                )

                rect.setPen(box_pen)
                rect.setBrush(QBrush(QColor(255, 0, 0, 30)))
                rect.setTransformOriginPoint(cx, cy)
                rect.setRotation(heading)

                self.map_view.scene.addItem(rect)

        # ---- Center View ----
        self.map_view.centerOn(path_pixels[0])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VinsGroundControl()
    window.show()
    sys.exit(app.exec_())
