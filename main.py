import sys
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, 
                            QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, 
                            QTextEdit, QMessageBox, QSplitter, QStatusBar, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent
from PyQt5.QtCore import Qt, QTimer, QMutex, pyqtSignal

import socket
import threading

class TurretClient:
    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.receive_thread = None
        self.lock = threading.Lock()
        self.response_callback = None
        self.error_callback = None
        
    def connect(self):
        """Establish connection to the turret controller"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Start receive thread
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from the turret controller"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def _receive_loop(self):
        """Thread function to receive responses from the server"""
        buffer = ""
        
        while self.connected:
            try:
                data = self.socket.recv(1024).decode('ascii')
                if not data:
                    # Connection closed
                    self.connected = False
                    if self.error_callback:
                        self.error_callback("Connection closed by server")
                    break
                
                buffer += data
                
                # Process complete messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if self.response_callback:
                        self.response_callback(line)
            except Exception as e:
                if self.connected and self.error_callback:
                    self.error_callback(f"Receive error: {str(e)}")
                self.connected = False
                break
    
    def send_command(self, command):
        """Send a command to the turret controller"""
        if not self.connected:
            if self.error_callback:
                self.error_callback("Not connected")
            return False
        
        try:
            with self.lock:
                self.socket.sendall(f"{command}\n".encode('ascii'))
            return True
        except Exception as e:
            self.connected = False
            if self.error_callback:
                self.error_callback(f"Send error: {str(e)}")
            return False
    
    def laser_on(self):
        """Turn on the laser"""
        return self.send_command("LASER:ON")
    
    def laser_off(self):
        """Turn off the laser"""
        return self.send_command("LASER:OFF")
    
    def rotate(self, yaw, pitch):
        """Rotate the turret"""
        return self.send_command(f"ROTATE:{yaw},{pitch}")
    
    def ping(self):
        """Ping the server to check if it's alive"""
        return self.send_command("PING")


class CameraStreamClient:
    """Camera client that fetches frames on demand"""
    def __init__(self):
        self.stream_url = None
        self.video_capture = None
        self.frame = None
        self.last_frame_time = 0
        self.fps = 0
        self.debug = True
        self.mutex = QMutex()
        self.frame_count = 0
        self.connected = False
    
    def connect(self, host='127.0.0.1', port=8080):
        """Connect to the camera stream"""
        try:
            self.stream_url = f"http://{host}:{port}/camera"
            
            if self.debug:
                print(f"Connecting to camera at {self.stream_url}")
            
            # Only initialize the capture, don't start streaming
            self.video_capture = cv2.VideoCapture(self.stream_url)
            
            # Force buffer size to be small
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.video_capture.isOpened():
                if self.debug:
                    print(f"Failed to open video stream at {self.stream_url}")
                return False
            
            if self.debug:
                print("Video capture opened successfully")
            
            self.connected = True
            return True
        except Exception as e:
            if self.debug:
                print(f"Camera connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from the camera stream"""
        if self.debug:
            print("Disconnecting camera")
        
        self.connected = False
        
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass
            self.video_capture = None
    
    def get_frame(self):
        """Get a single frame on demand"""
        if not self.connected or self.video_capture is None:
            return None
        
        try:
            # Lock to prevent concurrent access
            self.mutex.lock()
            
            # Grab and retrieve a new frame
            ret, frame = self.video_capture.read()
            
            if not ret or frame is None:
                if self.debug:
                    print("Failed to read frame")
                self.mutex.unlock()
                return None
            
            # Update stats
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            
            if time_diff >= 1.0:
                self.fps = self.frame_count / time_diff
                self.frame_count = 0
                self.last_frame_time = current_time
            
            self.frame_count += 1
            self.frame = frame
            
            self.mutex.unlock()
            return frame.copy()
        
        except Exception as e:
            if self.debug:
                print(f"Error getting frame: {str(e)}")
            
            if self.mutex.tryLock():
                self.mutex.unlock()
            
            return None


class TurretCalibration:
    def __init__(self, camera_client, turret_client):
        self.camera = camera_client
        self.turret = turret_client
        self.calibration_data = []
        self.board_size = (9, 6)  # Default checkerboard size
        self.pan_model = None
        self.tilt_model = None
        self.is_calibrated = False
        self.poly_features = None
    
    def detect_checkerboard(self, frame):
        """Detect checkerboard in the given frame"""
        if frame is None:
            print("No frame provided to detect_checkerboard")
            return None, None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        
        # Try with the configured board size
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # If detection fails, try with other common sizes
        if not ret:
            print(f"Failed to detect with size {self.board_size}, trying alternatives...")
            board_sizes_to_try = [
                (self.board_size[0]-1, self.board_size[1]),
                (self.board_size[0], self.board_size[1]-1),
                (self.board_size[0]+1, self.board_size[1]),
                (self.board_size[0], self.board_size[1]+1),
                (8, 6),
                (9, 6),
                (7, 5),
                (6, 9)
            ]
            
            for size in board_sizes_to_try:
                print(f"Trying with size {size}...")
                ret, corners = cv2.findChessboardCorners(gray, size, 
                                                        cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                        cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret:
                    print(f"Checkerboard found with pattern size {size}")
                    self.board_size = size
                    break
        
        if ret and corners is not None and len(corners) > 0:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            try:
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                corners = refined_corners
            except Exception as e:
                print(f"Warning: Corner refinement failed: {str(e)}")
            
            # Draw the corners on display frame
            cv2.drawChessboardCorners(display_frame, self.board_size, corners, ret)
            
            # Return corners as a list of points
            corners_list = [(corner[0][0], corner[0][1]) for corner in corners]
            print(f"Checkerboard pattern found! Detected {len(corners_list)} corners.")
            return corners_list, display_frame
        else:
            print("Failed to detect any checkerboard pattern.")
            return None, display_frame
    
    def build_transformation_model(self):
        """Build the transformation model from calibration data"""
        if len(self.calibration_data) < 5:
            return False
        
        data = np.array(self.calibration_data)
        
        # Extract features and targets
        X = data[:, 0:2]  # Camera x,y coordinates
        Y_pan = data[:, 2]  # Pan/Yaw angles
        Y_tilt = data[:, 3]  # Tilt/Pitch angles
        
        # Use polynomial regression for better accuracy
        self.poly_features = PolynomialFeatures(degree=2)
        X_poly = self.poly_features.fit_transform(X)
        
        # Create and train the models
        self.pan_model = LinearRegression().fit(X_poly, Y_pan)
        self.tilt_model = LinearRegression().fit(X_poly, Y_tilt)
        
        self.is_calibrated = True
        return True
    
    def aim_at_target(self, target_x, target_y):
        """Aim the turret at a target point in the camera view"""
        if not self.is_calibrated:
            return False
        
        # Transform the target point
        target_point = np.array([[target_x, target_y]])
        target_poly = self.poly_features.transform(target_point)
        
        # Predict pan and tilt angles
        pan_angle = self.pan_model.predict(target_poly)[0]
        tilt_angle = self.tilt_model.predict(target_poly)[0]
        
        # Rotate turret to the predicted position
        return self.turret.rotate(pan_angle, tilt_angle)
    
    def save_calibration(self, filename):
        """Save the calibration model to a file"""
        if not self.is_calibrated:
            return False
        
        calibration_data = {
            'pan_model': self.pan_model,
            'tilt_model': self.tilt_model,
            'poly_features': self.poly_features,
            'calibration_data': self.calibration_data
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(calibration_data, f)
            return True
        except Exception as e:
            print(f"Error saving calibration: {str(e)}")
            return False
    
    def load_calibration(self, filename):
        """Load a calibration model from a file"""
        try:
            with open(filename, 'rb') as f:
                calibration_data = pickle.load(f)
            
            self.pan_model = calibration_data['pan_model']
            self.tilt_model = calibration_data['tilt_model']
            self.poly_features = calibration_data['poly_features']
            self.calibration_data = calibration_data['calibration_data']
            self.is_calibrated = True
            return True
        except Exception as e:
            print(f"Error loading calibration: {str(e)}")
            return False


class SimplifiedTurretCalibrationApp(QMainWindow):
    def __init__(self, camera_client, turret_client):
        super().__init__()
        
        # Store clients
        self.camera = camera_client
        self.turret = turret_client
        
        # Create calibration system
        self.calibration = TurretCalibration(self.camera, self.turret)
        
        # Setup variables
        self.laser_on = False
        self.selected_corner_idx = -1
        self.corners = None
        self.current_frame = None
        self.calibration_mode = "IDLE"  # IDLE, DETECTED, CORNER_SELECTED, CALIBRATED
        self.current_yaw = 0
        self.current_pitch = 0
        self.manual_laser_pos = None
        self.wasd_step_size = 0.005  # Step size for WASD movement
        self.targeting_mode = False  # Flag for click-to-target mode
        
        # Setup UI
        self.init_ui()
        
        # Timer for updating camera feed
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_feed)
        self.camera_timer.start(50)  # 20 fps update rate
        
        # Timer for status updates
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
        
        # Set focus policy to accept keyboard input
        self.setFocusPolicy(Qt.StrongFocus)
    
    def _map_label_click_to_image_coords(self, click_pos):
        """Maps a click position on the camera_view_label to image coordinates."""
        label_widget = self.camera_view_label
        label_size = label_widget.size()
        
        current_pixmap = label_widget.pixmap()
        if not current_pixmap or current_pixmap.isNull():
            self.log_message("Debug: _map_label_click_to_image_coords - No pixmap in label.")
            return None # No pixmap
            
        pixmap_size = current_pixmap.size() # Original image size
        
        if pixmap_size.width() == 0 or pixmap_size.height() == 0:
            self.log_message(f"Debug: _map_label_click_to_image_coords - Invalid pixmap size: {pixmap_size.width()}x{pixmap_size.height()}")
            return None # Invalid pixmap size

        # Calculate the offset of the pixmap within the label (due to Qt.AlignCenter)
        # QLabel displays the pixmap at its native size, centered.
        offset_x = (label_size.width() - pixmap_size.width()) / 2.0
        offset_y = (label_size.height() - pixmap_size.height()) / 2.0

        # Convert click coordinates from label space to pixmap space
        img_x_float = click_pos.x() - offset_x
        img_y_float = click_pos.y() - offset_y

        # Check if the click is within the actual bounds of the (potentially smaller or larger) pixmap
        # as mapped from the label.
        if 0 <= img_x_float < pixmap_size.width() and \
           0 <= img_y_float < pixmap_size.height():
            return int(img_x_float), int(img_y_float)
        else:
            self.log_message(f"Debug: Click ({click_pos.x()},{click_pos.y()}) mapped to ({img_x_float:.1f},{img_y_float:.1f}) is outside pixmap ({pixmap_size.width()},{pixmap_size.height()})")
            return None # Click was outside the image area on the label
    
    def init_ui(self):
        # Main window settings
        self.setWindowTitle("Turret Calibration with WASD Controls")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Left side - Camera view
        camera_frame = QWidget()
        camera_layout = QVBoxLayout()
        camera_frame.setLayout(camera_layout)
        
        # Camera view label with click event
        self.camera_view_label = ClickableLabel("Camera feed loading...")
        self.camera_view_label.clicked.connect(self.on_image_clicked)
        self.camera_view_label.setAlignment(Qt.AlignCenter)
        self.camera_view_label.setMinimumSize(800, 600)
        self.camera_view_label.setFont(QFont("Arial", 14))
        camera_layout.addWidget(self.camera_view_label)
        
        # Resolution and status label
        self.resolution_label = QLabel("Resolution: - | Use WASD to move turret (Click for targeting when calibrated)")
        self.resolution_label.setFont(QFont("Arial", 10))
        camera_layout.addWidget(self.resolution_label)
        
        main_layout.addWidget(camera_frame, 3)  # 3:1 ratio for camera to controls
        
        # Right side - Controls
        controls_frame = QWidget()
        controls_layout = QVBoxLayout()
        controls_frame.setLayout(controls_layout)
        
        # Connection group
        connection_group = QGroupBox("Connection")
        connection_group.setFont(QFont("Arial", 12, QFont.Bold))
        connection_layout = QGridLayout()
        connection_group.setLayout(connection_layout)
        
        # Camera settings
        camera_host_label = QLabel("Camera Host:")
        connection_layout.addWidget(camera_host_label, 0, 0)
        self.camera_host_edit = QLineEdit("127.0.0.1")
        connection_layout.addWidget(self.camera_host_edit, 0, 1)
        
        camera_port_label = QLabel("Camera Port:")
        connection_layout.addWidget(camera_port_label, 1, 0)
        self.camera_port_spin = QSpinBox()
        self.camera_port_spin.setRange(1, 65535)
        self.camera_port_spin.setValue(8080)
        connection_layout.addWidget(self.camera_port_spin, 1, 1)
        
        # Turret settings
        turret_host_label = QLabel("Turret Host:")
        connection_layout.addWidget(turret_host_label, 2, 0)
        self.turret_host_edit = QLineEdit("127.0.0.1")
        connection_layout.addWidget(self.turret_host_edit, 2, 1)
        
        turret_port_label = QLabel("Turret Port:")
        connection_layout.addWidget(turret_port_label, 3, 0)
        self.turret_port_spin = QSpinBox()
        self.turret_port_spin.setRange(1, 65535)
        self.turret_port_spin.setValue(8888)
        connection_layout.addWidget(self.turret_port_spin, 3, 1)
        
        # Connect button
        self.reconnect_btn = QPushButton("Reconnect")
        self.reconnect_btn.clicked.connect(self.reconnect)
        connection_layout.addWidget(self.reconnect_btn, 4, 0, 1, 2)
        
        controls_layout.addWidget(connection_group)
        
        # WASD Controls group
        wasd_group = QGroupBox("WASD Controls")
        wasd_group.setFont(QFont("Arial", 12, QFont.Bold))
        wasd_layout = QVBoxLayout()
        wasd_group.setLayout(wasd_layout)
        
        # Instructions
        wasd_instructions = QLabel("W: Pitch Up\nS: Pitch Down\nA: Yaw Left\nD: Yaw Right\nQ: Decrease Step Size\nE: Increase Step Size")
        wasd_instructions.setFont(QFont("Arial", 10))
        wasd_layout.addWidget(wasd_instructions)
        
        # Step size
        step_size_layout = QHBoxLayout()
        step_size_label = QLabel("Step Size:")
        step_size_layout.addWidget(step_size_label)
        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(0.001, 0.1)
        self.step_size_spin.setValue(self.wasd_step_size)
        self.step_size_spin.setSingleStep(0.001)
        self.step_size_spin.setDecimals(3)
        self.step_size_spin.valueChanged.connect(self.on_step_size_changed)
        step_size_layout.addWidget(self.step_size_spin)
        wasd_layout.addLayout(step_size_layout)
        
        # Current position
        self.position_label = QLabel(f"Yaw: {self.current_yaw:.3f} | Pitch: {self.current_pitch:.3f}")
        wasd_layout.addWidget(self.position_label)
        
        controls_layout.addWidget(wasd_group)
        
        # Checkerboard detection group
        checkerboard_group = QGroupBox("Checkerboard Detection")
        checkerboard_group.setFont(QFont("Arial", 12, QFont.Bold))
        checkerboard_layout = QVBoxLayout()
        checkerboard_group.setLayout(checkerboard_layout)
        
        # Checkerboard settings
        board_settings_layout = QGridLayout()
        
        board_width_label = QLabel("Board Width:")
        board_settings_layout.addWidget(board_width_label, 0, 0)
        self.board_width_spin = QSpinBox()
        self.board_width_spin.setRange(2, 20)
        self.board_width_spin.setValue(self.calibration.board_size[0])
        board_settings_layout.addWidget(self.board_width_spin, 0, 1)
        
        board_height_label = QLabel("Board Height:")
        board_settings_layout.addWidget(board_height_label, 1, 0)
        self.board_height_spin = QSpinBox()
        self.board_height_spin.setRange(2, 20)
        self.board_height_spin.setValue(self.calibration.board_size[1])
        board_settings_layout.addWidget(self.board_height_spin, 1, 1)
        
        checkerboard_layout.addLayout(board_settings_layout)
        
        # Detect checkerboard button
        self.detect_checkerboard_btn = QPushButton("Detect Checkerboard")
        self.detect_checkerboard_btn.clicked.connect(self.detect_checkerboard)
        checkerboard_layout.addWidget(self.detect_checkerboard_btn)
        
        controls_layout.addWidget(checkerboard_group)
        
        # Laser control group
        laser_group = QGroupBox("Laser Control")
        laser_group.setFont(QFont("Arial", 12, QFont.Bold))
        laser_layout = QVBoxLayout()
        laser_group.setLayout(laser_layout)
        
        # Laser toggle
        laser_toggle_layout = QHBoxLayout()
        self.laser_toggle_btn = QPushButton("Toggle Laser")
        self.laser_toggle_btn.clicked.connect(self.toggle_laser)
        laser_toggle_layout.addWidget(self.laser_toggle_btn)
        
        self.laser_status_label = QLabel("Laser: OFF")
        laser_toggle_layout.addWidget(self.laser_status_label)
        laser_layout.addLayout(laser_toggle_layout)
        
        # Corner selection
        corner_combo_layout = QHBoxLayout()
        corner_label = QLabel("Select Corner:")
        corner_combo_layout.addWidget(corner_label)
        self.corner_combo = QComboBox()
        self.corner_combo.addItem("None")
        self.corner_combo.currentIndexChanged.connect(self.update_selected_corner)
        corner_combo_layout.addWidget(self.corner_combo)
        laser_layout.addLayout(corner_combo_layout)
        
        # Position display
        self.laser_pos_label = QLabel("Position: Not selected")
        laser_layout.addWidget(self.laser_pos_label)
        
        # Capture button
        self.capture_point_btn = QPushButton("Capture Calibration Point")
        self.capture_point_btn.clicked.connect(self.capture_calibration_point)
        self.capture_point_btn.setEnabled(False)
        laser_layout.addWidget(self.capture_point_btn)
        
        controls_layout.addWidget(laser_group)
        
        # Calibration control group
        cal_control_group = QGroupBox("Calibration Management")
        cal_control_group.setFont(QFont("Arial", 12, QFont.Bold))
        cal_control_layout = QVBoxLayout()
        cal_control_group.setLayout(cal_control_layout)
        
        # Build model button
        self.build_model_btn = QPushButton("Build Calibration Model")
        self.build_model_btn.clicked.connect(self.build_calibration_model)
        self.build_model_btn.setEnabled(False)
        cal_control_layout.addWidget(self.build_model_btn)
        
        # Save/Load buttons
        save_load_layout = QHBoxLayout()
        
        self.save_cal_btn = QPushButton("Save Calibration")
        self.save_cal_btn.clicked.connect(self.save_calibration)
        self.save_cal_btn.setEnabled(False)
        save_load_layout.addWidget(self.save_cal_btn)
        
        self.load_cal_btn = QPushButton("Load Calibration")
        self.load_cal_btn.clicked.connect(self.load_calibration)
        save_load_layout.addWidget(self.load_cal_btn)
        
        cal_control_layout.addLayout(save_load_layout)
        
        # Reset calibration button
        self.reset_calibration_btn = QPushButton("Reset Calibration")
        self.reset_calibration_btn.clicked.connect(self.reset_calibration)
        cal_control_layout.addWidget(self.reset_calibration_btn)
        
        # Status 
        self.calibration_status_label = QLabel("Calibration: Not calibrated")
        cal_control_layout.addWidget(self.calibration_status_label)
        
        # Toggle targeting mode
        self.targeting_btn = QPushButton("Toggle Click-to-Target Mode")
        self.targeting_btn.clicked.connect(self.toggle_targeting_mode)
        self.targeting_btn.setEnabled(False)
        cal_control_layout.addWidget(self.targeting_btn)
        
        controls_layout.addWidget(cal_control_group)
        
        # Log area
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)
        
        controls_layout.addWidget(log_group)
        
        # Add to main layout
        main_layout.addWidget(controls_frame, 1)  # 3:1 ratio for camera to controls
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("System initialized")
    
    def update_camera_feed(self):
        """Update camera feed with the latest frame"""
        # Get new frame
        frame = self.camera.get_frame()
        if frame is None:
            return
        
        self.current_frame = frame.copy()
        
        # Make a copy for display
        display_frame = self.current_frame.copy()
        
        # Draw detected checkerboard corners if available
        if hasattr(self, 'corners') and self.corners is not None:
            # Draw all corners
            for i, (x, y) in enumerate(self.corners):
                color = (0, 255, 0)  # Default green
                if i == self.selected_corner_idx:
                    color = (0, 0, 255)  # Selected corner in red
                cv2.circle(display_frame, (int(x), int(y)), 5, color, -1)
                cv2.putText(display_frame, str(i), (int(x)+10, int(y)+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw manually selected laser position if available
        if self.manual_laser_pos:
            cv2.circle(display_frame, self.manual_laser_pos, 8, (255, 0, 255), -1)
            cv2.putText(display_frame, "Laser", (self.manual_laser_pos[0]+10, self.manual_laser_pos[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Add WASD control reminder
        if self.targeting_mode:
            cv2.putText(display_frame, "TARGETING MODE - CLICK TO AIM", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "WASD: Move | Q/E: Adjust Step", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Convert to Qt format for display
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Update display
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_view_label.setPixmap(QPixmap.fromImage(qt_image))
        
        # Update resolution info
        mode_text = "TARGETING" if self.targeting_mode else self.calibration_mode
        self.resolution_label.setText(f"Resolution: {w}x{h} | Mode: {mode_text} | Step: {self.wasd_step_size:.3f}")
    
    def on_image_clicked(self, event):
        """Handle clicks on the camera image"""
        
        image_coords = self._map_label_click_to_image_coords(event.pos())
        
        if image_coords is None:
            self.log_message("Click was outside the image area or image not available.")
            return

        img_x, img_y = image_coords

        if self.targeting_mode and self.calibration.is_calibrated:
            self.calibration.aim_at_target(img_x, img_y)
            self.log_message(f"Aiming at ({img_x}, {img_y})")
            return
            
        # This part is for manual laser marking / calibration point capture
        can_proceed_with_marking = False
        if self.calibration_mode in ["CORNER_SELECTED", "DETECTED"]:
            can_proceed_with_marking = True
        elif hasattr(self, 'corners') and self.corners is not None:
            # Mode is not ideal (e.g. IDLE), but corners are detected.
            # Allow proceeding by switching mode.
            self.calibration_mode = "DETECTED" 
            self.log_message(f"Switched to DETECTED mode as corners are present for manual marking.")
            can_proceed_with_marking = True

        if not can_proceed_with_marking:
            self.log_message("Manual marking: Please detect checkerboard first, or select a corner if already detected.")
            return
        
        # Store manual laser position using the correctly mapped coordinates
        self.manual_laser_pos = (img_x, img_y) # img_x, img_y are already int
        
        # Auto-select nearest corner logic (uses img_x, img_y from the image)
        if self.selected_corner_idx == -1 and hasattr(self, 'corners') and self.corners is not None:
            nearest_idx = -1
            min_dist = float('inf')
            
            # self.corners stores image coordinates (x,y)
            for i, (corner_img_x, corner_img_y) in enumerate(self.corners):
                dist = ((corner_img_x - img_x) ** 2 + (corner_img_y - img_y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            # Threshold for auto-selection, e.g., 50 pixels on the image
            if min_dist < 50: 
                self.selected_corner_idx = nearest_idx
                # Ensure combo box update is safe
                if nearest_idx + 1 < self.corner_combo.count():
                    self.corner_combo.setCurrentIndex(nearest_idx + 1) # +1 for "None" entry
                else:
                    self.log_message(f"Warning: nearest_idx+1 {nearest_idx+1} out of bounds for corner_combo (count {self.corner_combo.count()})")

                self.log_message(f"Auto-selected nearest corner {nearest_idx} to click at ({img_x},{img_y})")
        
        self.laser_pos_label.setText(f"Position: ({img_x}, {img_y})")
        self.capture_point_btn.setEnabled(True)
        self.log_message(f"Laser position marked at ({img_x}, {img_y})")
    
    def keyPressEvent(self, event):
        """Handle key press events for WASD control"""
        key = event.key()
        
        if key == Qt.Key_W:  # Pitch up
            new_pitch = min(self.current_pitch + self.wasd_step_size, 0.5)
            self.move_turret_to(self.current_yaw, new_pitch)
        elif key == Qt.Key_S:  # Pitch down
            new_pitch = max(self.current_pitch - self.wasd_step_size, 0.0)
            self.move_turret_to(self.current_yaw, new_pitch)
        elif key == Qt.Key_A:  # Yaw left
            new_yaw = max(self.current_yaw - self.wasd_step_size, -1.0)
            self.move_turret_to(new_yaw, self.current_pitch)
        elif key == Qt.Key_D:  # Yaw right
            new_yaw = min(self.current_yaw + self.wasd_step_size, 0.0)
            self.move_turret_to(new_yaw, self.current_pitch)
        elif key == Qt.Key_Q:  # Decrease step size
            self.wasd_step_size = max(self.wasd_step_size / 2, 0.001)
            self.step_size_spin.setValue(self.wasd_step_size)
            self.log_message(f"Step size decreased to {self.wasd_step_size:.3f}")
        elif key == Qt.Key_E:  # Increase step size
            self.wasd_step_size = min(self.wasd_step_size * 2, 0.1)
            self.step_size_spin.setValue(self.wasd_step_size)
            self.log_message(f"Step size increased to {self.wasd_step_size:.3f}")
        elif key == Qt.Key_Space:  # Toggle laser
            self.toggle_laser()
        elif key == Qt.Key_C:  # Capture point (if enabled)
            if self.capture_point_btn.isEnabled():
                self.capture_calibration_point()
        else:
            super().keyPressEvent(event)
    
    def on_step_size_changed(self, value):
        """Update step size when spin box value changes"""
        self.wasd_step_size = value
    
    def move_turret_to(self, yaw, pitch):
        """Move the turret to the specified position"""
        success = self.turret.rotate(yaw, pitch)
        
        if success:
            self.current_yaw = yaw
            self.current_pitch = pitch
            self.position_label.setText(f"Yaw: {self.current_yaw:.3f} | Pitch: {self.current_pitch:.3f}")
    
    def detect_checkerboard(self):
        """Detect checkerboard in the current frame"""
        if self.current_frame is None:
            self.log_message("No frame available")
            return
        
        # Update board size from UI
        width = self.board_width_spin.value()
        height = self.board_height_spin.value()
        self.calibration.board_size = (width, height)
        
        self.log_message(f"Detecting checkerboard with size {width}x{height}...")
        
        # Detect checkerboard
        corners, display_frame = self.calibration.detect_checkerboard(self.current_frame)
        
        if corners is not None:
            self.corners = corners
            self.log_message(f"Detected {len(corners)} checkerboard corners")
            
            # Update corner selection combo box
            self.corner_combo.clear()
            self.corner_combo.addItem("None")
            for i in range(len(self.corners)):
                self.corner_combo.addItem(f"Corner {i}")
            
            # Update calibration mode
            self.calibration_mode = "DETECTED"
        else:
            self.log_message("Failed to detect checkerboard")
    
    def update_selected_corner(self, index):
        """Update the selected corner index"""
        if index == 0:  # "None" option
            self.selected_corner_idx = -1
            self.calibration_mode = "DETECTED"
            self.capture_point_btn.setEnabled(False)
        else:
            self.selected_corner_idx = index - 1  # Adjust for "None" entry
            self.calibration_mode = "CORNER_SELECTED"
            
            # Reset manual laser position
            self.manual_laser_pos = None
            self.laser_pos_label.setText("Position: Not selected")
            self.capture_point_btn.setEnabled(False)
            
            self.log_message(f"Selected corner {self.selected_corner_idx}")
    
    def toggle_laser(self):
        """Toggle the laser on/off"""
        if self.laser_on:
            self.turret.laser_off()
            self.laser_on = False
            self.laser_status_label.setText("Laser: OFF")
            self.log_message("Laser turned off")
        else:
            self.turret.laser_on()
            self.laser_on = True
            self.laser_status_label.setText("Laser: ON")
            self.log_message("Laser turned on")
    
    def capture_calibration_point(self):
        """Capture a calibration point with the selected corner and manually marked laser position"""
        if self.corners is None:
            self.log_message("No checkerboard detected")
            return
            
        if self.manual_laser_pos is None:
            self.log_message("No laser position marked")
            return
        
        # If no corner is selected, allow capture with the laser position only
        if self.selected_corner_idx >= 0:
            # Get the selected corner coordinates
            corner_x, corner_y = self.corners[self.selected_corner_idx]
        else:
            # Use the laser position as the corner (useful for corner-less calibration)
            corner_x, corner_y = self.manual_laser_pos
        
        # Get the current turret position
        current_yaw = self.current_yaw
        current_pitch = self.current_pitch
        
        # Add to calibration data
        self.calibration.calibration_data.append((corner_x, corner_y, current_yaw, current_pitch))
        
        if self.selected_corner_idx >= 0:
            self.log_message(f"Calibration point captured: Corner {self.selected_corner_idx} "
                            f"at ({corner_x:.1f}, {corner_y:.1f}) "
                            f"with angles (Yaw={current_yaw}, Pitch={current_pitch})")
        else:
            self.log_message(f"Calibration point captured: Position ({corner_x:.1f}, {corner_y:.1f}) "
                            f"with angles (Yaw={current_yaw}, Pitch={current_pitch})")
        
        # Reset for next point
        self.manual_laser_pos = None
        self.capture_point_btn.setEnabled(False)
        
        if self.selected_corner_idx >= 0:
            self.corner_combo.setCurrentIndex(0)  # Reset corner selection
        
        # Enable build model button if we have enough points
        if len(self.calibration.calibration_data) >= 5:
            self.build_model_btn.setEnabled(True)
            self.log_message("You have enough points to build a calibration model (minimum 5)")
    
    def build_calibration_model(self):
        """Build the calibration model"""
        if len(self.calibration.calibration_data) < 5:
            self.log_message("Not enough calibration points. Need at least 5.")
            return
        
        self.log_message("Building calibration model...")
        success = self.calibration.build_transformation_model()
        
        if success:
            self.log_message("Calibration model built successfully")
            self.calibration_status_label.setText("Calibration: Calibrated")
            self.calibration_mode = "CALIBRATED"
            self.save_cal_btn.setEnabled(True)
            self.targeting_btn.setEnabled(True)
        else:
            self.log_message("Failed to build calibration model")
    
    def save_calibration(self):
        """Save calibration model to a file"""
        if not self.calibration.is_calibrated:
            self.log_message("No calibration model to save")
            return
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save Calibration Model", 
                                                 "", "Calibration Files (*.cal)")
        
        if file_path:
            if not file_path.endswith('.cal'):
                file_path += '.cal'
                
            success = self.calibration.save_calibration(file_path)
            
            if success:
                self.log_message(f"Calibration saved to {file_path}")
            else:
                self.log_message("Failed to save calibration")
    
    def load_calibration(self):
        """Load calibration model from a file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Load Calibration Model", 
                                                 "", "Calibration Files (*.cal)")
        
        if file_path:
            success = self.calibration.load_calibration(file_path)
            
            if success:
                self.log_message(f"Calibration loaded from {file_path}")
                self.calibration_status_label.setText("Calibration: Calibrated")
                self.calibration_mode = "CALIBRATED"
                self.save_cal_btn.setEnabled(True)
                self.targeting_btn.setEnabled(True)
            else:
                self.log_message("Failed to load calibration")
    
    def toggle_targeting_mode(self):
        """Toggle targeting mode for click-to-aim functionality"""
        if not self.calibration.is_calibrated:
            self.log_message("Calibration required for targeting mode")
            return
        
        self.targeting_mode = not self.targeting_mode
        
        if self.targeting_mode:
            self.log_message("Targeting mode enabled - Click on image to aim turret")
            self.targeting_btn.setText("Disable Targeting Mode")
        else:
            self.log_message("Targeting mode disabled")
            self.targeting_btn.setText("Enable Targeting Mode")
    
    def reset_calibration(self):
        """Reset the calibration data"""
        self.calibration.calibration_data = []
        self.calibration.is_calibrated = False
        self.calibration_mode = "IDLE"
        self.corners = None
        self.selected_corner_idx = -1
        self.manual_laser_pos = None
        self.corner_combo.clear()
        self.corner_combo.addItem("None")
        self.build_model_btn.setEnabled(False)
        self.capture_point_btn.setEnabled(False)
        self.save_cal_btn.setEnabled(False)
        self.targeting_btn.setEnabled(False)
        self.targeting_mode = False
        
        self.log_message("Calibration data reset")
        self.calibration_status_label.setText("Calibration: Not calibrated")
    
    def update_status(self):
        """Update status information"""
        # Update connection status
        camera_status = "Connected" if self.camera.connected else "Disconnected"
        turret_status = "Connected" if self.turret.connected else "Disconnected"
        
        status_text = f"Camera: {camera_status} | Turret: {turret_status}"
        
        if self.camera.fps > 0:
            status_text += f" | FPS: {self.camera.fps:.1f}"
        
        if self.calibration.is_calibrated:
            status_text += " | Calibrated: YES"
        else:
            status_text += f" | Calibrated: NO | Points: {len(self.calibration.calibration_data)}/5+"
        
        self.statusBar.showMessage(status_text)
    
    def reconnect(self):
        """Reconnect to the camera and turret with current settings"""
        # Disconnect first
        self.camera.disconnect()
        self.turret.disconnect()
        
        # Get new connection parameters
        camera_host = self.camera_host_edit.text()
        camera_port = self.camera_port_spin.value()
        turret_host = self.turret_host_edit.text()
        turret_port = self.turret_port_spin.value()
        
        # Reconnect
        self.log_message(f"Reconnecting to camera at {camera_host}:{camera_port}")
        if not self.camera.connect(camera_host, camera_port):
            self.log_message("Failed to connect to camera")
            QMessageBox.warning(self, "Connection Error",
                              "Failed to connect to camera")
        else:
            self.log_message("Camera connected successfully")
        
        self.log_message(f"Reconnecting to turret at {turret_host}:{turret_port}")
        self.turret.host = turret_host
        self.turret.port = turret_port
        if not self.turret.connect():
            self.log_message("Failed to connect to turret")
            QMessageBox.warning(self, "Connection Error",
                              "Failed to connect to turret")
        else:
            self.log_message("Turret connected successfully")
    
    def log_message(self, message):
        """Add a message to the log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.ensureCursorVisible()
        print(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """Handle cleanup when closing the application"""
        # Stop the timers
        if hasattr(self, 'camera_timer'):
            self.camera_timer.stop()
        
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        
        # Turn off laser if it's on
        if self.laser_on:
            self.turret.laser_off()
        
        # Disconnect clients
        self.turret.disconnect()
        self.camera.disconnect()
        
        event.accept()


class ClickableLabel(QLabel):
    """Custom QLabel that emits signal when clicked"""
    clicked = pyqtSignal(object)
    
    def __init__(self, text=""):
        super().__init__(text)
        self.setMouseTracking(True)
    
    def mousePressEvent(self, event):
        self.clicked.emit(event)
        super().mousePressEvent(event)


def main():
    app = QApplication(sys.argv)
    
    # Initialize clients
    camera_client = CameraStreamClient()
    turret_client = TurretClient()
    
    # Create and show main window
    window = SimplifiedTurretCalibrationApp(camera_client, turret_client)
    window.show()
    
    # Auto-connect to devices if possible
    try:
        print("Connecting to camera...")
        camera_client.connect()
        print("Connecting to turret...")
        turret_client.connect()
    except Exception as e:
        print(f"Auto-connect failed: {str(e)}")
        pass  # Let the user reconnect manually if auto-connect fails
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()