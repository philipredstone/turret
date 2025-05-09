import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QLineEdit, QSlider, QGroupBox, 
                            QGridLayout, QCheckBox, QStatusBar, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Import the provided client classes
from CameraStreamClient import CameraStreamClient
from TurretClient import TurretClient

class FrameWorker(QThread):
    """Worker thread for processing camera frames without blocking the UI"""
    frame_ready = pyqtSignal(np.ndarray)
    fps_update = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, camera_client):
        super().__init__()
        self.camera_client = camera_client
        self.running = False
        self.frame_counter = 0
        self.last_fps_time = 0
        self.smoothed_fps = 0
        
    def run(self):
        self.running = True
        self.last_fps_time = time.time()
        
        while self.running:
            try:
                # Get the latest frame
                frame = self.camera_client.get_frame()
                
                if frame is not None:
                    # Update FPS calculation
                    self.frame_counter += 1
                    current_time = time.time()
                    elapsed = current_time - self.last_fps_time
                    
                    if elapsed >= 1.0:  # Update FPS every second
                        current_fps = self.frame_counter / elapsed
                        # Smooth FPS to avoid jumpy display
                        if self.smoothed_fps == 0:
                            self.smoothed_fps = current_fps
                        else:
                            self.smoothed_fps = 0.8 * self.smoothed_fps + 0.2 * current_fps
                        
                        self.fps_update.emit(int(self.smoothed_fps))
                        self.frame_counter = 0
                        self.last_fps_time = current_time
                    
                    # Emit signal with the frame
                    self.frame_ready.emit(frame)
                
                # Sleep to reduce CPU usage, but not too much to keep frame rate high
                time.sleep(0.01)
                
            except Exception as e:
                self.error.emit(f"Frame processing error: {str(e)}")
                time.sleep(0.5)  # Longer delay on error
                
    def stop(self):
        self.running = False
        self.wait()


class TurretCameraClient(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize client objects
        self.camera_client = CameraStreamClient()
        self.turret_client = TurretClient()
        
        # Set up callbacks
        self.camera_client.error_callback = self.on_camera_error
        self.turret_client.response_callback = self.on_turret_response
        self.turret_client.error_callback = self.on_turret_error
        
        # Initialize frame worker
        self.frame_worker = None
        
        # Initialize UI
        self.init_ui()
        
        # Timer for periodic updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Turret Camera Control")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Connection controls section
        connection_group = QGroupBox("Connection Settings")
        connection_layout = QGridLayout()
        
        # Camera connection
        connection_layout.addWidget(QLabel("Camera Host:"), 0, 0)
        self.camera_host_input = QLineEdit("127.0.0.1")
        connection_layout.addWidget(self.camera_host_input, 0, 1)
        
        connection_layout.addWidget(QLabel("Camera Port:"), 0, 2)
        self.camera_port_input = QLineEdit("8080")
        connection_layout.addWidget(self.camera_port_input, 0, 3)
        
        self.camera_connect_btn = QPushButton("Connect Camera")
        self.camera_connect_btn.clicked.connect(self.toggle_camera_connection)
        connection_layout.addWidget(self.camera_connect_btn, 0, 4)
        
        # Turret connection
        connection_layout.addWidget(QLabel("Turret Host:"), 1, 0)
        self.turret_host_input = QLineEdit("127.0.0.1")
        connection_layout.addWidget(self.turret_host_input, 1, 1)
        
        connection_layout.addWidget(QLabel("Turret Port:"), 1, 2)
        self.turret_port_input = QLineEdit("8888")
        connection_layout.addWidget(self.turret_port_input, 1, 3)
        
        self.turret_connect_btn = QPushButton("Connect Turret")
        self.turret_connect_btn.clicked.connect(self.toggle_turret_connection)
        connection_layout.addWidget(self.turret_connect_btn, 1, 4)
        
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)
        
        # Main content area with camera view and turret controls
        splitter = QSplitter(Qt.Horizontal)
        
        # Camera view
        camera_group = QGroupBox("Camera View")
        camera_layout = QVBoxLayout()
        
        self.camera_label = QLabel("No Camera Feed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        self.camera_label.setMinimumSize(640, 480)
        camera_layout.addWidget(self.camera_label)
        
        # Camera status bar
        camera_status_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        camera_status_layout.addWidget(self.fps_label)
        
        self.resolution_label = QLabel("Resolution: --")
        camera_status_layout.addWidget(self.resolution_label)
        
        self.test_pattern_cb = QCheckBox("Test Pattern")
        self.test_pattern_cb.toggled.connect(self.toggle_test_pattern)
        camera_status_layout.addWidget(self.test_pattern_cb)
        
        camera_layout.addLayout(camera_status_layout)
        camera_group.setLayout(camera_layout)
        splitter.addWidget(camera_group)
        
        # Turret controls
        turret_group = QGroupBox("Turret Control")
        turret_layout = QVBoxLayout()
        
        # Yaw control
        yaw_layout = QHBoxLayout()
        yaw_layout.addWidget(QLabel("Yaw:"))
        
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setRange(-90, 90)
        self.yaw_slider.setValue(0)
        self.yaw_slider.setTickPosition(QSlider.TicksBelow)
        self.yaw_slider.setTickInterval(10)
        self.yaw_slider.valueChanged.connect(self.on_turret_position_change)
        yaw_layout.addWidget(self.yaw_slider)
        
        self.yaw_value = QLabel("0°")
        yaw_layout.addWidget(self.yaw_value)
        
        turret_layout.addLayout(yaw_layout)
        
        # Pitch control
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch:"))
        
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(-45, 45)
        self.pitch_slider.setValue(0)
        self.pitch_slider.setTickPosition(QSlider.TicksBelow)
        self.pitch_slider.setTickInterval(5)
        self.pitch_slider.valueChanged.connect(self.on_turret_position_change)
        pitch_layout.addWidget(self.pitch_slider)
        
        self.pitch_value = QLabel("0°")
        pitch_layout.addWidget(self.pitch_value)
        
        turret_layout.addLayout(pitch_layout)
        
        # Laser control
        laser_layout = QHBoxLayout()
        self.laser_btn = QPushButton("Laser OFF")
        self.laser_btn.setCheckable(True)
        self.laser_btn.toggled.connect(self.toggle_laser)
        self.laser_btn.setStyleSheet("QPushButton:checked { background-color: red; color: white; }")
        laser_layout.addWidget(self.laser_btn)
        
        self.ping_btn = QPushButton("Ping Turret")
        self.ping_btn.clicked.connect(self.ping_turret)
        laser_layout.addWidget(self.ping_btn)
        
        turret_layout.addLayout(laser_layout)
        
        # Communication log
        self.comm_log = QLabel("Not connected")
        self.comm_log.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        self.comm_log.setWordWrap(True)
        self.comm_log.setMinimumHeight(150)
        self.comm_log.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        turret_layout.addWidget(QLabel("Communication Log:"))
        turret_layout.addWidget(self.comm_log)
        
        turret_group.setLayout(turret_layout)
        splitter.addWidget(turret_group)
        
        # Set initial splitter sizes
        splitter.setSizes([600, 400])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Use test pattern initially
        self.using_test_pattern = False
        self.test_pattern_timer = QTimer()
        self.test_pattern_timer.timeout.connect(self.update_test_pattern)
        
    def toggle_camera_connection(self):
        """Connect/disconnect from camera"""
        if self.camera_client.video_capture is None:
            # Connect
            host = self.camera_host_input.text()
            try:
                port = int(self.camera_port_input.text())
            except ValueError:
                self.statusBar.showMessage("Invalid camera port number")
                return
                
            self.statusBar.showMessage(f"Connecting to camera at {host}:{port}...")
            if self.camera_client.connect(host=host, port=port):
                self.statusBar.showMessage("Camera connected")
                self.camera_connect_btn.setText("Disconnect Camera")
                
                # Start the frame worker
                self.frame_worker = FrameWorker(self.camera_client)
                self.frame_worker.frame_ready.connect(self.update_frame)
                self.frame_worker.fps_update.connect(self.update_fps)
                self.frame_worker.error.connect(self.on_worker_error)
                self.frame_worker.start()
            else:
                self.statusBar.showMessage("Failed to connect to camera")
        else:
            # Disconnect
            if self.frame_worker is not None:
                self.frame_worker.stop()
                self.frame_worker = None
                
            self.camera_client.disconnect()
            self.camera_connect_btn.setText("Connect Camera")
            self.statusBar.showMessage("Camera disconnected")
            self.camera_label.setText("No Camera Feed")
            self.fps_label.setText("FPS: --")
            self.resolution_label.setText("Resolution: --")
            
    def toggle_turret_connection(self):
        """Connect/disconnect from turret"""
        if not self.turret_client.connected:
            # Connect
            host = self.turret_host_input.text()
            try:
                port = int(self.turret_port_input.text())
            except ValueError:
                self.statusBar.showMessage("Invalid turret port number")
                return
                
            self.turret_client.host = host
            self.turret_client.port = port
            
            self.statusBar.showMessage(f"Connecting to turret at {host}:{port}...")
            if self.turret_client.connect():
                self.statusBar.showMessage("Turret connected")
                self.turret_connect_btn.setText("Disconnect Turret")
                self.comm_log.setText("Connected to turret controller")
                
                # Send initial position
                self.on_turret_position_change()
            else:
                self.statusBar.showMessage("Failed to connect to turret")
        else:
            # Disconnect
            self.turret_client.disconnect()
            self.turret_connect_btn.setText("Connect Turret")
            self.statusBar.showMessage("Turret disconnected")
            self.comm_log.setText("Not connected")
            self.laser_btn.setChecked(False)
            
    def toggle_laser(self, checked):
        """Turn laser on/off"""
        if not self.turret_client.connected:
            self.statusBar.showMessage("Turret not connected")
            self.laser_btn.setChecked(False)
            return
            
        if checked:
            success = self.turret_client.laser_on()
            self.laser_btn.setText("Laser ON")
        else:
            success = self.turret_client.laser_off()
            self.laser_btn.setText("Laser OFF")
            
        if not success:
            self.laser_btn.setChecked(False)
            self.laser_btn.setText("Laser OFF")
            
    def on_turret_position_change(self):
        """Handle turret position slider changes"""
        if not self.turret_client.connected:
            return
            
        yaw = self.yaw_slider.value()
        pitch = self.pitch_slider.value()
        
        self.yaw_value.setText(f"{yaw}°")
        self.pitch_value.setText(f"{pitch}°")
        
        # Send rotation command with rate limiting to reduce network traffic
        self.turret_client.rotate(yaw, pitch)
        
    def ping_turret(self):
        """Send ping to turret"""
        if not self.turret_client.connected:
            self.statusBar.showMessage("Turret not connected")
            return
            
        self.turret_client.ping()
        
    def on_camera_error(self, error_msg):
        """Handle camera error callback"""
        self.statusBar.showMessage(f"Camera error: {error_msg}")
        
    def on_turret_response(self, response):
        """Handle turret response callback"""
        # Update the communication log with latest message at the top
        current_text = self.comm_log.text()
        lines = current_text.split("\n")
        lines.insert(0, f"← {response}")
        # Keep only the last 10 messages
        if len(lines) > 10:
            lines = lines[:10]
        self.comm_log.setText("\n".join(lines))
        
    def on_turret_error(self, error_msg):
        """Handle turret error callback"""
        self.statusBar.showMessage(f"Turret error: {error_msg}")
        # Also add to the log
        current_text = self.comm_log.text()
        lines = current_text.split("\n")
        lines.insert(0, f"! ERROR: {error_msg}")
        if len(lines) > 10:
            lines = lines[:10]
        self.comm_log.setText("\n".join(lines))
        
    def on_worker_error(self, error_msg):
        """Handle frame worker error"""
        self.statusBar.showMessage(f"Frame worker error: {error_msg}")
        
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """Update the camera display with a new frame"""
        if frame is None:
            return
            
        # Convert the image to RGB format (from BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width, channels = rgb_frame.shape
        self.resolution_label.setText(f"Resolution: {width}x{height}")
        
        # Create QImage from the frame
        bytes_per_line = channels * width
        q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit the label while preserving aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.width(), 
            self.camera_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
    @pyqtSlot(int)
    def update_fps(self, fps):
        """Update the FPS display"""
        self.fps_label.setText(f"FPS: {fps}")
        
    def toggle_test_pattern(self, checked):
        """Toggle between camera feed and test pattern"""
        self.using_test_pattern = checked
        
        if checked:
            # Start test pattern timer if no camera connected
            if self.frame_worker is None:
                self.test_pattern_timer.start(33)  # ~30 FPS
        else:
            self.test_pattern_timer.stop()
            
    def update_test_pattern(self):
        """Update the display with a test pattern"""
        if self.camera_client.video_capture is None:
            # Only show test pattern if no camera connected
            test_frame = self.camera_client.get_test_pattern()
            self.update_frame(test_frame)
            self.fps_label.setText("FPS: ~30 (Test)")
            
    def update_status(self):
        """Periodic status updates"""
        # Update turret connection status
        if self.turret_client.connected:
            # Send ping every few seconds to keep connection alive
            self.turret_client.ping()
            
        # Update camera status if not getting frames
        if self.camera_client.video_capture is not None and self.frame_worker is None:
            self.toggle_camera_connection()  # Reconnect
            
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up resources
        if self.frame_worker is not None:
            self.frame_worker.stop()
            
        if self.camera_client.video_capture is not None:
            self.camera_client.disconnect()
            
        if self.turret_client.connected:
            # Turn off laser before disconnecting
            self.turret_client.laser_off()
            self.turret_client.disconnect()
            
        event.accept()


# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TurretCameraClient()
    window.show()
    sys.exit(app.exec_())