import sys
import os
import json
import cv2
import numpy as np
import pytesseract
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PySide6 import QtGui
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QLineEdit, 
    QTextEdit, QFileDialog, QTabWidget, QCheckBox, 
    QSpinBox, QGroupBox, QFormLayout
)

# --- Constant Management ---
MAX_ROIS = 5
CONFIG_FILE_NAME = "VideoEventTracker.json"


# --- Configuration Management Class ---
class ConfigManager:
    """Class to manage saving and loading of configurations."""
    def __init__(self):
        self.config_file = CONFIG_FILE_NAME
        self.data = {
            "parent_folder": "", 
            "log_path": "",
            "rois": [
                {"enabled": True, "x": 100, "y": 100, "w": 200, "h": 200, "threshold": 500, "cooldown_ms": 3000},
                {"enabled": False, "x": 400, "y": 100, "w": 200, "h": 200, "threshold": 500, "cooldown_ms": 3000},
                {"enabled": False, "x": 100, "y": 400, "w": 200, "h": 200, "threshold": 500, "cooldown_ms": 3000},
                {"enabled": False, "x": 400, "y": 400, "w": 200, "h": 200, "threshold": 500, "cooldown_ms": 3000},
                {"enabled": False, "x": 300, "y": 300, "w": 200, "h": 200, "threshold": 500, "cooldown_ms": 3000},
            ],
            "ocr_settings": {
                "x": 50, "y": 50, "w": 150, "h": 50,
                "padding": 10,
                "grayscale": True,
                "threshold": 127,
                "tesseract_path": F"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            },
            "detection_settings": {
                "threshold": 500,
                "cooldown_ms": 3000
            }
        }

    def load(self) -> None:
        """Load configuration from a file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    self.data.update(loaded_data)
            except Exception as e:
                print(f"Config load error: {e}")

    def save(self, path: str) -> None:
        """Save configuration to the specified path."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            print(f"Config save error: {e}")


# --- Video Analysis Worker (Multi-threading) ---
class VideoAnalyzer(QThread):
    """Worker class to sequentially read video files, perform ROI detection and OCR."""
    frame_ready = Signal(np.ndarray, str)      # Image, log message
    progress_update = Signal(str)               # Progress status
    ocr_image_ready = Signal(np.ndarray)       # Pre-processed image for OCR
    config_updated = Signal(dict) 

    def __init__(self, folder_path: str, config: dict, log_path: str):
        super().__init__()
        self.folder_path = folder_path
        self.config = config
        self.log_path = log_path
        self.is_running = True
        self.last_detection_time: Dict[int, float] = {}
        self.frame_index = 0

    @Slot(dict)
    def update_config(self, new_config: dict):
        """Overwrite internal config with new settings sent from the UI."""
        self.config = new_config

    def run(self) -> None:
        video_files = list(Path(self.folder_path).rglob("*.mp4"))
        if not video_files:
            self.progress_update.emit("No .mp4 files found in the target directory.")
            return

        for vid_path in video_files:
            if not self.is_running:
                break
            self.progress_update.emit(f"Processing: {vid_path.name}")
            
            cap = cv2.VideoCapture(str(vid_path))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or not self.is_running:
                    break

                self.frame_index += 1  # Count frame number

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_time = datetime.now().timestamp() * 1000

                for i, roi in enumerate(self.config["rois"]):
                    if not roi["enabled"]:
                        continue
                    
                    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                    roi_img = gray[max(0, y):min(frame.shape[0], y + h), 
                                    max(0, x):min(frame.shape[1], x + w)]
                    
                    if roi_img.size == 0:
                        continue

                    avg_brightness = np.mean(roi_img)
                    threshold = roi["threshold"]
                    cooldown = roi["cooldown_ms"]

                    if avg_brightness > threshold:
                        last_time = self.last_detection_time.get(i, 0)
                        
                        if (current_time - last_time) > cooldown:
                            self.last_detection_time[i] = current_time
                            ocr_res, ocr_img = self.perform_ocr(frame, roi, i)
                            if ocr_res:
                                self.frame_ready.emit(frame, f"ROI {i+1} OCR Result: {ocr_res}")
                                self.write_log(ocr_res, self.frame_index, str(vid_path))
                            self.ocr_image_ready.emit(ocr_img)

                # Always send the latest frame to preview (without log)
                self.frame_ready.emit(frame, "")
            cap.release()
        
        self.progress_update.emit("All processing completed.")

    def write_log(self, text: str, frame_index: int, video_path: str):
        if not self.log_path:
            return

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{text},{frame_index},{video_path}\n")
        except Exception as e:
            print(f"Log write error: {e}")

    def perform_ocr(self, frame: np.ndarray, roi: dict, index: int) -> Tuple[Optional[str], np.ndarray]:
        """Crop the specified area based on OCR settings and add padding background."""
        tess_path = self.config["ocr_settings"].get("tesseract_path", "")
        if tess_path:
            pytesseract.pytesseract.tesseract_cmd = tess_path
            
        # Get settings
        ocr = self.config["ocr_settings"]
        x, y, w, h = ocr["x"], ocr["y"], ocr["w"], ocr["h"]
        padding = ocr["padding"]
        bg_color = ocr.get("bg_color", [0, 0, 0]) # Get background color

        # --- Accurately crop the specified area ---
        # Limit coordinates to stay within frame boundaries (clipping)
        y1 = max(0, y)
        y2 = min(frame.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(frame.shape[1], x + w)

        # Extract specified range from original image
        source_crop = frame[y1:y2, x1:x2]
        actual_h = y2 - y1
        actual_w = x2 - x1

        # --- Create canvas with padding and paste crop ---
        # Canvas size = specified width/height + padding * 2
        canvas_h = h + (padding * 2)
        canvas_w = w + (padding * 2)

        # Create empty image filled with background color (canvas)
        canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

        # Paste the cropped image into the center of the canvas
        # Adjust placement so it stays correct even if clipped at edges
        canvas[padding : padding + actual_h, padding : padding + actual_w] = source_crop

        # Subsequent processing (grayscale, etc.) uses the canvas
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        if ocr["grayscale"]:
            _, thresh = cv2.threshold(gray, ocr["threshold"], 255, cv2.THRESH_BINARY)
        else:
            thresh = gray

        try:
            text = pytesseract.image_to_string(thresh, lang='jpn+eng').strip()
            if text:
                return text, thresh
        except Exception as e:
            print(f"OCR Error: {e}")

        return None, thresh


# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Surveillance & OCR Tool")
        self.resize(1400, 950) 

        # Organize initialization sequence
        self.config_manager = ConfigManager()
        self.config_manager.load()
        self.config = self.config_manager.data
        
        # Build UI
        self.init_ui()

        # Reflect settings from file
        default_path = str(Path(__file__).parent / CONFIG_FILE_NAME)
        if Path(default_path).exists():
            self.load_settings(default_path)

        # Variables for state management
        self.drag_target: Optional[Tuple[str, int]] = None # ("roi", index) or ("ocr", None)
        self.drag_start_pos = None
        self.video_w = 0
        self.video_h = 0

    def init_ui(self):
        """Build the main UI layout."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Top: Preview and Tabs ---
        top_layout = QHBoxLayout()
        
        # Video preview
        self.preview_label = QLabel("Video Preview")
        self.preview_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.preview_label.setFixedSize(800, 600)

        # Enable mouse tracking and focus
        self.preview_label.setMouseTracking(True)
        self.preview_label.setFocusPolicy(Qt.StrongFocus)

        # Connect mouse events
        self.preview_label.mousePressEvent = self.handle_mouse_press
        self.preview_label.mouseMoveEvent = self.handle_mouse_move
        self.preview_label.mouseReleaseEvent = self.handle_mouse_release

        # Tab control
        tabs = QTabWidget()
        tab1 = QWidget()
        t1_layout = QVBoxLayout(tab1)
        self.status_label = QLabel("Waiting...")
        t1_layout.addWidget(self.status_label)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        t1_layout.addWidget(self.log_area)
        
        self.ocr_preview_label = QLabel("OCR Preview")
        self.ocr_preview_label.setStyleSheet("background-color: black; border: 2px solid blue;")
        self.ocr_preview_label.setFixedSize(300, 200)
        t1_layout.addWidget(self.ocr_preview_label)

        tab2 = QWidget()
        t2_layout = QVBoxLayout(tab2)
        # Split settings area into methods for construction
        roi_group = self._create_roi_settings_group()
        ocr_group = self._create_ocr_settings_group()
        
        t2_layout.addWidget(roi_group)
        t2_layout.addWidget(ocr_group)

        tabs.addTab(tab1, "Status & Logs")
        tabs.addTab(tab2, "Settings")
        
        top_layout.addWidget(self.preview_label)
        top_layout.addWidget(tabs)

        # --- Bottom: Path Selection and Action Buttons ---
        bottom_layout = QHBoxLayout()
        
        path_layout = QVBoxLayout()
        self.path_input = QLineEdit()
        btn_browse = QPushButton("Select Folder")
        btn_browse.clicked.connect(self.browse_folder)

        self.log_path_input = QLineEdit()
        btn_log_browse = QPushButton("Select Log File")
        btn_log_browse.clicked.connect(self.browse_log_file)

        path_layout.addWidget(QLabel("Parent Folder Path:"))
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(btn_browse)
        path_layout.addWidget(QLabel("Log Output File Path:"))
        path_layout.addWidget(self.log_path_input)
        path_layout.addWidget(btn_log_browse)

        ctrl_layout = QVBoxLayout()
        self.btn_start = QPushButton("Start Process")
        self.btn_stop = QPushButton("Stop Process")
        self.btn_save = QPushButton("Save Setting")
        self.btn_load = QPushButton("Load Setting")
        
        self.btn_start.clicked.connect(self.start_process)
        self.btn_stop.clicked.connect(self.stop_process)
        self.btn_save.clicked.connect(self.save_settings)
        self.btn_load.clicked.connect(self.handle_load_setting_button)

        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_save)
        ctrl_layout.addWidget(self.btn_load)

        bottom_layout.addLayout(path_layout, 2)
        bottom_layout.addLayout(ctrl_layout, 1)
        
        layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)

    def browse_log_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select Log File", "ocr_log.txt")
        if path:
            self.log_path_input.setText(path)


    # --- Helper methods for configuration updates ---
    def update_config_roi(self, idx: int, key: str, value: int):
        self.config["rois"][idx][key] = value

        # Update preview immediately
        if hasattr(self, "last_frame"):
            self.update_preview(self.last_frame, "")

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.update_config.emit(self.config)


    def update_config_ocr(self, key: str, value: int):
        self.config["ocr_settings"][key] = value

        # Update preview immediately
        if hasattr(self, "last_frame"):
            self.update_preview(self.last_frame, "")

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.update_config.emit(self.config)


    def _create_roi_settings_group(self) -> QGroupBox:
        """Helper method to build UI for ROI settings (with real-time reflection)."""
        group = QGroupBox("ROI Settings (Max 5)")
        grid_layout = QVBoxLayout()
        self.roi_spinboxes = [] 

        for i in range(MAX_ROIS):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            
            cb = QCheckBox(f"Region {i+1}")
            cb.setChecked(self.config["rois"][i]["enabled"])
            # Reflect checkbox changes immediately
            cb.clicked.connect(lambda ch, idx=i: self.toggle_roi(idx))
            
            x_sb = QSpinBox()
            y_sb = QSpinBox()
            w_sb = QSpinBox()
            h_sb = QSpinBox()
            for sb in [x_sb, y_sb, w_sb, h_sb]: 
                sb.setRange(0, 5000)
            
            r = self.config["rois"][i]
            x_sb.setValue(r["x"])
            y_sb.setValue(r["y"])
            w_sb.setValue(r["w"])
            h_sb.setValue(r["h"])

            thresh_sb = QSpinBox()
            cooldown_sb = QSpinBox()
            for sb in [thresh_sb, cooldown_sb]: 
                sb.setRange(0, 10000)
            thresh_sb.setValue(r["threshold"])
            cooldown_sb.setValue(r["cooldown_ms"])

            # --- Connect signals to reflect SpinBox changes immediately in config ---
            x_sb.valueChanged.connect(lambda val, idx=i: self.update_config_roi(idx, "x", val))
            y_sb.valueChanged.connect(lambda val, idx=i: self.update_config_roi(idx, "y", val))
            w_sb.valueChanged.connect(lambda val, idx=i: self.update_config_roi(idx, "w", val))
            h_sb.valueChanged.connect(lambda val, idx=i: self.update_config_roi(idx, "h", val))
            thresh_sb.valueChanged.connect(lambda val, idx=i: self.update_config_roi(idx, "threshold", val))
            cooldown_sb.valueChanged.connect(lambda val, idx=i: self.update_config_roi(idx, "cooldown_ms", val))

            row_layout.addWidget(cb)
            for label_text, sb in [("X:", x_sb), ("Y:", y_sb), ("W:", w_sb), ("H:", h_sb)]:
                row_layout.addWidget(QLabel(label_text))
                row_layout.addWidget(sb)
            
            row_layout.addSpacing(20)
            row_layout.addWidget(QLabel("Thr:"))
            row_layout.addWidget(thresh_sb)
            row_layout.addWidget(QLabel("CD(ms):"))
            row_layout.addWidget(cooldown_sb)

            self.roi_spinboxes.append({
                "cb": cb, "x": x_sb, "y": y_sb, "w": w_sb, "h": h_sb, 
                "threshold": thresh_sb, "cooldown": cooldown_sb
            })
            grid_layout.addWidget(row_widget)
        
        group.setLayout(grid_layout)
        return group


    def _create_ocr_settings_group(self) -> QGroupBox:
        """Helper method to build UI for OCR settings (with real-time reflection)."""
        group = QGroupBox("OCR Settings")
        form = QFormLayout()
        
        self.ocr_x = QSpinBox()
        self.ocr_y = QSpinBox()
        self.ocr_w = QSpinBox()
        self.ocr_h = QSpinBox()
        self.ocr_pad = QSpinBox()
        for s in [self.ocr_x, self.ocr_y, self.ocr_w, self.ocr_h, self.ocr_pad]: 
            s.setRange(0, 5000)
            
        o_cfg = self.config["ocr_settings"]
        self.ocr_x.setValue(o_cfg["x"])
        self.ocr_y.setValue(o_cfg["y"])
        self.ocr_w.setValue(o_cfg["w"])
        self.ocr_h.setValue(o_cfg["h"])
        self.ocr_pad.setValue(o_cfg["padding"])

        # --- Tesseract Path setting field ---
        self.tess_path_input = QLineEdit()
        self.tess_path_input.setText(o_cfg.get("tesseract_path", ""))

        btn_tess_browse = QPushButton("Browse Tesseract")
        btn_tess_browse.clicked.connect(self.browse_tesseract)

        form.addRow("X:", self.ocr_x)
        form.addRow("Y:", self.ocr_y)
        form.addRow("W:", self.ocr_w)
        form.addRow("H:", self.ocr_h)
        form.addRow("Padding:", self.ocr_pad)

        form.addRow("Tesseract Path:", self.tess_path_input)
        form.addRow("", btn_tess_browse)


        # --- Connect signals to reflect SpinBox changes immediately in config ---
        self.ocr_x.valueChanged.connect(lambda val: self.update_config_ocr("x", val))
        self.ocr_y.valueChanged.connect(lambda val: self.update_config_ocr("y", val))
        self.ocr_w.valueChanged.connect(lambda val: self.update_config_ocr("w", val))
        self.ocr_h.valueChanged.connect(lambda val: self.update_config_ocr("h", val))
        self.ocr_pad.valueChanged.connect(lambda val: self.update_config_ocr("padding", val))

        # Reflect path changes immediately
        self.tess_path_input.textChanged.connect(
            lambda val: self.update_config_ocr("tesseract_path", val)
        )

        group.setLayout(form)
        return group

    def browse_tesseract(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Tesseract Executable",
            "",
            "Executable (*.exe)"
        )
        if path:
            self.tess_path_input.setText(path)

    def browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path: 
            self.path_input.setText(path)

    def toggle_roi(self, idx: int):
        cb = self.roi_spinboxes[idx]["cb"]
        self.config["rois"][idx]["enabled"] = cb.isChecked()

        # Update preview immediately
        if hasattr(self, "last_frame"):
            self.update_preview(self.last_frame, "")


    @Slot(np.ndarray, str)
    def update_preview(self, frame: np.ndarray, log_msg: str):
        self.last_frame = frame.copy()
        if log_msg:
            self.log_area.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_msg}")

        self.video_w, self.video_h = frame.shape[1], frame.shape[0]
        disp_w, disp_h = self.preview_label.width(), self.preview_label.height()

        display_frame = frame.copy()
        for i, roi in enumerate(self.config["rois"]):
            if roi["enabled"]:
                x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        ocr = self.config["ocr_settings"]
        cv2.rectangle(display_frame, (ocr["x"], ocr["y"]),
                       (ocr["x"] + ocr["w"], ocr["y"] + ocr["h"]), (255, 0, 0), 3)

        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_image.data, self.video_w, self.video_h,
                        self.video_w * 3, QImage.Format_RGB888)

        # Scale calculation considering aspect ratio
        scale = min(disp_w / self.video_w, disp_h / self.video_h)
        img_w = int(self.video_w * scale)
        img_h = int(self.video_h * scale)

        # Letterbox offset calculation
        self.offset_x = (disp_w - img_w) // 2
        self.offset_y = (disp_h - img_h) // 2
        self.scale = scale

        pixmap = QPixmap.fromImage(q_img).scaled(img_w, img_h, Qt.KeepAspectRatio)
        final_pixmap = QPixmap(disp_w, disp_h)
        final_pixmap.fill(Qt.black)

        painter = QtGui.QPainter(final_pixmap)
        painter.drawPixmap(self.offset_x, self.offset_y, pixmap)
        painter.end()

        self.preview_label.setPixmap(final_pixmap)


    @Slot(np.ndarray)
    def update_ocr_preview(self, ocr_img: np.ndarray):
        rgb_image = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.ocr_preview_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.ocr_preview_label.size(), Qt.KeepAspectRatio))


    def start_process(self):
        path = self.path_input.text()
        if not os.path.exists(path):
            self.log_area.append("Error: Path does not exist.")
            return
        
        # Sync UI state to config (including individual ROI settings)
        for i in range(MAX_ROIS):
            s = self.roi_spinboxes[i]
            self.config["rois"][i]["enabled"] = s["cb"].isChecked()
            self.config["rois"][i]["x"], self.config["rois"][i]["y"] = \
                s["x"].value(), s["y"].value()
            self.config["rois"][i]["w"], self.config["rois"][i]["h"] = \
                s["w"].value(), s["h"].value()
            self.config["rois"][i]["threshold"] = s["threshold"].value()
            self.config["rois"][i]["cooldown_ms"] = s["cooldown"].value()
        
        self.config["ocr_settings"]["x"], self.config["ocr_settings"]["y"] = \
            self.ocr_x.value(), self.ocr_y.value()
        self.config["ocr_settings"]["w"], self.config["ocr_settings"]["h"] = \
            self.ocr_w.value(), self.ocr_h.value()
        self.config["ocr_settings"]["padding"] = self.ocr_pad.value()

        # Create and connect worker
        self.worker = VideoAnalyzer(path, self.config, self.log_path_input.text())
        self.worker.frame_ready.connect(self.update_preview)
        self.worker.progress_update.connect(lambda m: self.status_label.setText(m))
        self.worker.ocr_image_ready.connect(self.update_ocr_preview)
        # --- Establish configuration update connection ---
        self.worker.config_updated.connect(self.worker.update_config) 

        self.worker.start()

    def stop_process(self):
        if hasattr(self, 'worker'): 
            self.worker.is_running = False

    def save_settings(self):
        # Added: include log_path in saved settings
        self.config["parent_folder"] = self.path_input.text()
        self.config["log_path"] = self.log_path_input.text() 
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Settings", CONFIG_FILE_NAME)
        if path:
            self.config_manager.save(path)
            self.log_area.append("Settings saved.")

    def handle_load_setting_button(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Settings", CONFIG_FILE_NAME)
        if path:
            self.load_settings(path)

    def load_settings(self, path: str):
        """Load configuration from a file and update the UI."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                self.config_manager.data.update(loaded)
                self.config = self.config_manager.data
            
            if "parent_folder" in self.config:
                self.path_input.setText(self.config["parent_folder"])

            # Reflect log_path on UI
            if "log_path" in self.config:
                self.log_path_input.setText(self.config["log_path"])

            for i in range(MAX_ROIS):
                s = self.roi_spinboxes[i]
                r = self.config["rois"][i]
                s["cb"].setChecked(r["enabled"])
                s["x"].setValue(r["x"])
                s["y"].setValue(r["y"])
                s["w"].setValue(r["w"])
                s["h"].setValue(r["h"])
                s["threshold"].setValue(r.get("threshold", 500))
                s["cooldown"].setValue(r.get("cooldown_ms", 3000))
            
            o = self.config["ocr_settings"]
            self.ocr_x.setValue(o["x"])
            self.ocr_y.setValue(o["y"])
            self.ocr_w.setValue(o["w"])
            self.ocr_h.setValue(o["h"])
            self.ocr_pad.setValue(o["padding"])
            
            self.log_area.append("Settings loaded.")
        except Exception as e:
            self.log_area.append(f"Error: Failed to load settings ({e})")

    # --- Mouse Operation Logic ---
    def handle_mouse_press(self, event):
        if self.video_w == 0 or not hasattr(self, "scale"):
            return

        x = event.position().x()
        y = event.position().y()

        img_x = x - self.offset_x
        img_y = y - self.offset_y

        if img_x < 0 or img_y < 0:
            return
        if img_x > self.video_w * self.scale or img_y > self.video_h * self.scale:
            return

        # Convert to video coordinates (integer)
        vid_x = int(img_x / self.scale)
        vid_y = int(img_y / self.scale)

        # ROI Detection
        for i, roi in enumerate(self.config["rois"]):
            if roi["enabled"]:
                if (roi["x"] < vid_x < roi["x"] + roi["w"] and 
                    roi["y"] < vid_y < roi["y"] + roi["h"]):
                    self.drag_target = ("roi", i)
                    self.drag_start_pos = (vid_x, vid_y)  # Keep in video coordinates
                    return

        # OCR Detection
        ocr = self.config["ocr_settings"]
        if ocr["x"] < vid_x < ocr["x"] + ocr["w"] and \
           ocr["y"] < vid_y < ocr["y"] + ocr["h"]:
            self.drag_target = ("ocr", None)
            self.drag_start_pos = (vid_x, vid_y)  # Keep in video coordinates

    def handle_mouse_move(self, event):
        if self.drag_target is None or self.drag_start_pos is None:
            return

        x = event.position().x()
        y = event.position().y()

        img_x = x - self.offset_x
        img_y = y - self.offset_y

        # Convert to video coordinates (integer)
        vid_x = int(img_x / self.scale)
        vid_y = int(img_y / self.scale)

        # Calculate difference in video coordinates (zero error)
        dx = vid_x - self.drag_start_pos[0]
        dy = vid_y - self.drag_start_pos[1]

        if self.drag_target[0] == "roi":
            idx = self.drag_target[1]
            new_x = self.config["rois"][idx]["x"] + dx
            new_y = self.config["rois"][idx]["y"] + dy

            # Limit to stay within screen boundaries
            new_x = max(0, min(self.video_w - self.config["rois"][idx]["w"], new_x))
            new_y = max(0, min(self.video_h - self.config["rois"][idx]["h"], new_y))

            self.config["rois"][idx]["x"] = new_x
            self.config["rois"][idx]["y"] = new_y
            self.roi_spinboxes[idx]["x"].setValue(new_x)
            self.roi_spinboxes[idx]["y"].setValue(new_y)

        elif self.drag_target[0] == "ocr":
            new_x = self.config["ocr_settings"]["x"] + dx
            new_y = self.config["ocr_settings"]["y"] + dy

            new_x = max(0, min(self.video_w - self.config["ocr_settings"]["w"], new_x))
            new_y = max(0, min(self.video_h - self.config["ocr_settings"]["h"], new_y))

            self.config["ocr_settings"]["x"] = new_x
            self.config["ocr_settings"]["y"] = new_y
            self.ocr_x.setValue(new_x)
            self.ocr_y.setValue(new_y)

        # Update for next difference calculation (video coordinates)
        self.drag_start_pos = (vid_x, vid_y)


    def handle_mouse_release(self, event):
        self.drag_target = None
        self.drag_start_pos = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
