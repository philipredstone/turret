#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Turret Camera Control Application with Calibration
-------------------------------------------------
This application provides a GUI for controlling a 2-axis turret and camera
with calibration between the two coordinate systems using Gaussian Process Regression.
"""

import sys
import os
import cv2
import time
import threading
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout
from PyQt5.QtWidgets import QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox, QStatusBar
from PyQt5.QtWidgets import QTabWidget, QRadioButton, QButtonGroup, QSlider
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from abc import ABC, abstractmethod

# Import TurretClient for turret control
from turret_client import TurretClient

# Import our calibration classes
from turret_calibrator import TurretCameraCalibrator
from calibration_thread import CalibrationThread

# Define frame processor base class and red dot detector
class FrameProcessor(ABC):
    """Abstract base class for custom frame processors"""
    
    @abstractmethod
    def process_frame(self, frame):
        """Process a frame (in BGR format) and return the processed frame"""
        pass
    
    @property
    def name(self):
        """Return the name of the processor"""
        return self.__class__.__name__
    
    def cleanup(self):
        """Cleanup resources when processor is no longer needed"""
        pass

# Enhanced Red Dot Detector with multiple detection methods
class EnhancedRedDotDetector(FrameProcessor):
    # Detection methods
    HSV_METHOD = 0
    RGB_METHOD = 1
    BRIGHTNESS_METHOD = 2
    COMBINED_METHOD = 3
    
    def __init__(self, 
                 # Detection method
                 detection_method=COMBINED_METHOD,
                 
                 # Shape parameters
                 min_area=500, max_area=100000, min_circularity=0.6, 
                 
                 # HSV thresholds (much tighter ranges for red ball)
                 hue_low1=0, hue_high1=10, hue_low2=170, hue_high2=180,
                 sat_low=120, sat_high=255, val_low=100, val_high=255,
                 
                 # RGB thresholds (stronger red component)
                 red_threshold=150, green_max=80, blue_max=80,
                 
                 # Brightness thresholds
                 brightness_threshold=180,
                 
                 # Debug options
                 show_mask=True, show_intermediate=True):
        """Initialize with detection parameters"""
        self.detection_method = detection_method
        
        # Shape parameters
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        
        # HSV color thresholds 
        self.hue_low1 = hue_low1
        self.hue_high1 = hue_high1
        self.hue_low2 = hue_low2
        self.hue_high2 = hue_high2
        self.sat_low = sat_low
        self.sat_high = sat_high
        self.val_low = val_low
        self.val_high = val_high
        
        # RGB thresholds
        self.red_threshold = red_threshold
        self.green_max = green_max
        self.blue_max = blue_max
        
        # Brightness thresholds
        self.brightness_threshold = brightness_threshold
        
        # Debug options
        self.show_mask = show_mask
        self.show_intermediate = show_intermediate
        
        self.detected_dots = []
    
    def update_method(self, method):
        """Update detection method"""
        self.detection_method = method
    
    def update_shape_params(self, min_area, max_area, min_circularity):
        """Update shape parameters"""
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
    
    def update_hsv_params(self, hue_low1, hue_high1, hue_low2, hue_high2,
                          sat_low, sat_high, val_low, val_high):
        """Update HSV parameters"""
        self.hue_low1 = hue_low1
        self.hue_high1 = hue_high1
        self.hue_low2 = hue_low2
        self.hue_high2 = hue_high2
        self.sat_low = sat_low
        self.sat_high = sat_high
        self.val_low = val_low
        self.val_high = val_high
    
    def update_rgb_params(self, red_threshold, green_max, blue_max):
        """Update RGB parameters"""
        self.red_threshold = red_threshold
        self.green_max = green_max
        self.blue_max = blue_max
    
    def update_brightness_params(self, brightness_threshold):
        """Update brightness parameters"""
        self.brightness_threshold = brightness_threshold
    
    def update_debug_options(self, show_mask, show_intermediate):
        """Update debug options"""
        self.show_mask = show_mask
        self.show_intermediate = show_intermediate
    
    def create_hsv_mask(self, frame):
        """Create mask using HSV method"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define red color ranges (red wraps around in HSV)
        lower_red1 = np.array([self.hue_low1, self.sat_low, self.val_low])
        upper_red1 = np.array([self.hue_high1, self.sat_high, self.val_high])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([self.hue_low2, self.sat_low, self.val_low])
        upper_red2 = np.array([self.hue_high2, self.sat_high, self.val_high])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def create_rgb_mask(self, frame):
        """Create mask using RGB method"""
        # Extract channels
        b, g, r = cv2.split(frame)
        
        # Create masks for each condition
        red_high = r > self.red_threshold
        green_low = g < self.green_max
        blue_low = b < self.blue_max
        
        # Combine masks
        mask = np.logical_and(red_high, np.logical_and(green_low, blue_low))
        mask = mask.astype(np.uint8) * 255
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def create_brightness_mask(self, frame):
        """Create mask using brightness method"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find bright spots
        _, mask = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def create_combined_mask(self, frame):
        """Create mask using all methods combined"""
        hsv_mask = self.create_hsv_mask(frame)
        rgb_mask = self.create_rgb_mask(frame)
        brightness_mask = self.create_brightness_mask(frame)
        
        # Combine masks with different weights
        combined_mask = cv2.bitwise_or(hsv_mask, rgb_mask)
        combined_mask = cv2.bitwise_or(combined_mask, brightness_mask)
        
        # Apply final morphological operations
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask
    
    def find_dots_in_mask(self, mask, frame):
        """Find dots in the given mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Clear previous detections
        self.detected_dots = []
        
        # Set up result as a copy of the input frame
        result = frame.copy()
        
        # Process each contour
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_area <= area <= self.max_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Filter by circularity
                    if circularity >= self.min_circularity:
                        # Calculate centroid
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            radius = int(np.sqrt(area / np.pi))
                            
                            # Additional check for color to ensure it's truly red (for large objects)
                            if radius > 20:
                                # Sample a small region around the center
                                x1 = max(0, cx - 5)
                                y1 = max(0, cy - 5)
                                x2 = min(frame.shape[1] - 1, cx + 5)
                                y2 = min(frame.shape[0] - 1, cy + 5)
                                roi = frame[y1:y2, x1:x2]
                                
                                if roi.size > 0:
                                    # Calculate average BGR values in ROI
                                    avg_color = np.mean(roi, axis=(0,1))
                                    b_avg, g_avg, r_avg = avg_color
                                    
                                    # Skip if not actually red enough
                                    if not (r_avg > self.red_threshold and 
                                            g_avg < self.green_max and 
                                            b_avg < self.blue_max and 
                                            r_avg > (g_avg + b_avg) / 1.5):  # Red should be significantly higher
                                        continue
                            
                            # Store detection
                            self.detected_dots.append({
                                'center': (cx, cy),
                                'radius': radius,
                                'area': area,
                                'circularity': circularity
                            })
                            
                            # Draw circle and center point
                            cv2.circle(result, (cx, cy), radius, (0, 255, 0), 2)
                            cv2.circle(result, (cx, cy), 1, (0, 0, 255), 3)
                            
                            # Display information
                            label = f"A:{int(area)} C:{circularity:.2f}"
                            cv2.putText(result, label, (cx + radius, cy), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display number of dots found
        if self.detected_dots:
            cv2.putText(result, f"Dots: {len(self.detected_dots)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result
    
    def process_frame(self, frame):
        """Process frame to detect red dots"""
        # Create a copy of the input frame for the final result
        result = frame.copy()
        
        # Choose mask creation method based on setting
        if self.detection_method == self.HSV_METHOD:
            mask = self.create_hsv_mask(frame)
            method_name = "HSV"
        elif self.detection_method == self.RGB_METHOD:
            mask = self.create_rgb_mask(frame)
            method_name = "RGB"
        elif self.detection_method == self.BRIGHTNESS_METHOD:
            mask = self.create_brightness_mask(frame)
            method_name = "Brightness"
        else:  # COMBINED_METHOD
            mask = self.create_combined_mask(frame)
            method_name = "Combined"
        
        # Find dots in the mask
        result = self.find_dots_in_mask(mask, frame)
        
        # Show the detection method
        cv2.putText(result, f"Method: {method_name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show mask overlay if enabled
        if self.show_mask:
            h, w = result.shape[:2]
            # Create a color version of the mask for better visibility
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_color[:,:,0] = 0  # Remove blue channel
            mask_color[:,:,1] = mask.copy()  # Green for mask
            
            # Draw the mask name
            cv2.putText(mask_color, f"{method_name} Mask", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Scale it down for overlay
            mask_h = h // 3
            mask_w = w // 3
            mask_scaled = cv2.resize(mask_color, (mask_w, mask_h))
            
            # Create semi-transparent overlay in the bottom right
            alpha = 0.7
            roi = result[h-mask_h:h, w-mask_w:w]
            blended = cv2.addWeighted(roi, 1-alpha, mask_scaled, alpha, 0)
            result[h-mask_h:h, w-mask_w:w] = blended
        
        # Show intermediate masks if enabled
        if self.show_intermediate and self.detection_method == self.COMBINED_METHOD:
            # Generate all three masks
            hsv_mask = self.create_hsv_mask(frame)
            rgb_mask = self.create_rgb_mask(frame)
            brightness_mask = self.create_brightness_mask(frame)
            
            # Convert to color for display
            hsv_mask_color = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
            rgb_mask_color = cv2.cvtColor(rgb_mask, cv2.COLOR_GRAY2BGR)
            brightness_mask_color = cv2.cvtColor(brightness_mask, cv2.COLOR_GRAY2BGR)
            
            # Add titles
            cv2.putText(hsv_mask_color, "HSV", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(rgb_mask_color, "RGB", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(brightness_mask_color, "Bright", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Scale them down
            h, w = result.shape[:2]
            mask_h = h // 6
            mask_w = w // 6
            
            hsv_scaled = cv2.resize(hsv_mask_color, (mask_w, mask_h))
            rgb_scaled = cv2.resize(rgb_mask_color, (mask_w, mask_h))
            brightness_scaled = cv2.resize(brightness_mask_color, (mask_w, mask_h))
            
            # Place them in the top corners
            result[10:10+mask_h, 10:10+mask_w] = hsv_scaled
            result[10:10+mask_h, 20+mask_w:20+2*mask_w] = rgb_scaled
            result[10:10+mask_h, 30+2*mask_w:30+3*mask_w] = brightness_scaled
        
        return result

# Frame processing thread to avoid UI blocking
class FrameProcessingThread(QThread):
    processed_frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.processors = []
        self.current_frame = None
        self.running = False
        self.frame_lock = threading.Lock()  # For thread-safe frame access
        
    def add_processor(self, processor):
        """Add a frame processor to the chain"""
        self.processors.append(processor)
        
    def remove_processor(self, processor_name):
        """Remove a processor by name"""
        self.processors = [p for p in self.processors if p.name != processor_name]
        
    def clear_processors(self):
        """Remove all processors"""
        for processor in self.processors:
            processor.cleanup()
        self.processors = []
        
    def set_frame(self, frame):
        """Set the current frame to be processed"""
        with self.frame_lock:
            self.current_frame = frame.copy() if frame is not None else None
        
    def run(self):
        """Main processing loop"""
        self.running = True
        while self.running:
            # Check if we have a frame to process
            with self.frame_lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None
            
            if frame is not None:
                # Apply each processor in sequence
                for processor in self.processors:
                    frame = processor.process_frame(frame)
                
                # Emit the processed frame
                self.processed_frame_ready.emit(frame)
            
            # Don't hog CPU
            time.sleep(0.001)
        
    def stop(self):
        """Stop the processing thread"""
        self.running = False
        self.wait()

# Thread for handling camera stream connection and frame capture
class CameraStreamThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)
    error = pyqtSignal(str)
    
    def __init__(self, url=None):
        super().__init__()
        self.url = url
        self.running = False
        self.cap = None
        self.frame_count = 0
        self.last_fps_time = 0
        self.fps = 0
        self.latest_frame = None
        
    def set_url(self, url):
        self.url = url
        
    def stop(self):
        self.running = False
        self.wait()
        
    def run(self):
        self.running = True
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        try:
            self.cap = cv2.VideoCapture(self.url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            
            if not self.cap.isOpened():
                self.error.emit(f"Failed to open video stream at {self.url}")
                return
                
            # Main capture loop
            while self.running:
                # Just read the latest frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                    
                # Process frame and update latest_frame
                self.latest_frame = frame  # Store original frame (BGR format)
                self.frame_ready.emit(frame)  # Send original frame
                
                # Update FPS calculation
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_fps_time
                
                if elapsed >= 0.5:  # Update FPS twice per second
                    self.fps = self.frame_count / elapsed
                    self.fps_updated.emit(self.fps)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
        except Exception as e:
            self.error.emit(f"Stream error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()

class CameraViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Application state
        self.stream_thread = None
        self.processing_thread = None
        self.calibration_thread = None
        
        # Turret control
        self.turret_client = None
        self.tracking_active = False
        self.last_target_time = 0
        self.move_cooldown = 0.1  # seconds between turret moves
        
        # Setup UI
        self.setWindowTitle("Turret Camera Control")
        self.setMinimumSize(1200, 800)
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create top section with controls and preview
        self.top_section = QHBoxLayout()
        self.main_layout.addLayout(self.top_section)
        
        # Create control panel
        self.create_control_panel()
        
        # Create camera preview
        self.create_preview_area()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Setup timers for status bar updates
        self.ui_update_timer = QTimer()
        self.ui_update_timer.timeout.connect(self.update_ui_elements)
        self.ui_update_timer.start(200)  # Update status 5x per second
        
        # Performance tracking
        self.ui_fps = 0
        self.ui_frame_count = 0
        self.ui_last_fps_time = time.time()
        
        # Initialize processing thread
        self.processing_thread = FrameProcessingThread()
        self.processing_thread.processed_frame_ready.connect(self.update_display)
        self.processing_thread.start()
        
        # Initialize all parameters for the processor
        self.update_processor_settings()
        
        # Turret tracking timer
        self.tracking_timer = QTimer()
        self.tracking_timer.timeout.connect(self.update_turret_tracking)
        self.tracking_timer.start(50)  # Update 20 times per second
        
        # Initialize calibration-related attributes
        self.calibration_in_progress = False
        self.show_calibration_vis = False
        
        # Add calibration UI
        self.add_calibration_ui()
        
    def create_control_panel(self):
        # Control panel container
        self.control_panel = QWidget()
        self.control_panel.setFixedWidth(300)
        self.control_layout = QVBoxLayout(self.control_panel)
        self.top_section.addWidget(self.control_panel)
        
        # Camera connection group
        self.connection_group = QGroupBox("Camera Connection")
        self.connection_layout = QVBoxLayout(self.connection_group)
        self.control_layout.addWidget(self.connection_group)
        
        # Host input
        host_layout = QHBoxLayout()
        host_layout.addWidget(QLabel("Host:"))
        self.host_input = QLineEdit("127.0.0.1")
        host_layout.addWidget(self.host_input)
        self.connection_layout.addLayout(host_layout)
        
        # Port input
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(8080)
        port_layout.addWidget(self.port_input)
        self.connection_layout.addLayout(port_layout)
        
        # Path input
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Path:"))
        self.path_input = QLineEdit("/camera")
        path_layout.addWidget(self.path_input)
        self.connection_layout.addLayout(path_layout)
        
        # Connect button
        self.connect_button = QPushButton("Connect Camera")
        self.connect_button.clicked.connect(self.toggle_connection)
        self.connection_layout.addWidget(self.connect_button)
        
        # Turret Control Group
        self.turret_group = QGroupBox("Turret Control")
        self.turret_layout = QVBoxLayout(self.turret_group)
        self.control_layout.addWidget(self.turret_group)
        
        # Turret host/port
        turret_host_layout = QHBoxLayout()
        turret_host_layout.addWidget(QLabel("Host:"))
        self.turret_host_input = QLineEdit("127.0.0.1")
        turret_host_layout.addWidget(self.turret_host_input)
        self.turret_layout.addLayout(turret_host_layout)
        
        turret_port_layout = QHBoxLayout()
        turret_port_layout.addWidget(QLabel("Port:"))
        self.turret_port_input = QSpinBox()
        self.turret_port_input.setRange(1, 65535)
        self.turret_port_input.setValue(8888)
        turret_port_layout.addWidget(self.turret_port_input)
        self.turret_layout.addLayout(turret_port_layout)
        
        # Connect/Disconnect button
        self.turret_connect_button = QPushButton("Connect Turret")
        self.turret_connect_button.clicked.connect(self.toggle_turret_connection)
        self.turret_layout.addWidget(self.turret_connect_button)
        
        # Turret manual controls
        turret_controls_layout = QHBoxLayout()
        
        # Left column - Rotation controls
        rotation_layout = QVBoxLayout()
        rotation_layout.addWidget(QLabel("Rotation Controls"))
        
        # Direction buttons in a grid
        direction_layout = QGridLayout()
        self.up_button = QPushButton("▲")
        self.up_button.clicked.connect(lambda: self.adjust_turret_position(0, 0.1))
        self.down_button = QPushButton("▼")
        self.down_button.clicked.connect(lambda: self.adjust_turret_position(0, -0.1))
        self.left_button = QPushButton("◀")
        self.left_button.clicked.connect(lambda: self.adjust_turret_position(-0.1, 0))
        self.right_button = QPushButton("▶")
        self.right_button.clicked.connect(lambda: self.adjust_turret_position(0.1, 0))
        
        direction_layout.addWidget(self.up_button, 0, 1)
        direction_layout.addWidget(self.left_button, 1, 0)
        direction_layout.addWidget(self.right_button, 1, 2)
        direction_layout.addWidget(self.down_button, 2, 1)
        
        rotation_layout.addLayout(direction_layout)
        turret_controls_layout.addLayout(rotation_layout)
        
        # Right column - Laser controls
        laser_layout = QVBoxLayout()
        laser_layout.addWidget(QLabel("Laser Controls"))
        
        self.laser_on_button = QPushButton("Laser ON")
        self.laser_on_button.clicked.connect(self.laser_on)
        laser_layout.addWidget(self.laser_on_button)
        
        self.laser_off_button = QPushButton("Laser OFF")
        self.laser_off_button.clicked.connect(self.laser_off)
        laser_layout.addWidget(self.laser_off_button)
        
        turret_controls_layout.addLayout(laser_layout)
        self.turret_layout.addLayout(turret_controls_layout)
        
        # Tracking toggle
        self.tracking_checkbox = QCheckBox("Auto-Track Red Ball")
        self.tracking_checkbox.stateChanged.connect(self.toggle_tracking)
        self.turret_layout.addWidget(self.tracking_checkbox)
        
        # Frame Processing group
        self.processing_group = QGroupBox("Red Ball Detection")
        self.processing_layout = QVBoxLayout(self.processing_group)
        self.control_layout.addWidget(self.processing_group)
        
        # Processing toggle
        self.enable_processing_checkbox = QCheckBox("Enable Detection")
        self.enable_processing_checkbox.setChecked(True)
        self.enable_processing_checkbox.stateChanged.connect(self.toggle_processing)
        self.processing_layout.addWidget(self.enable_processing_checkbox)
        
        # Detection method - simplified to just use combined method
        self.method_group = QButtonGroup()
        self.combined_method_radio = QRadioButton("Combined Method")
        self.combined_method_radio.setChecked(True)
        self.method_group.addButton(self.combined_method_radio, EnhancedRedDotDetector.COMBINED_METHOD)
        self.processing_layout.addWidget(self.combined_method_radio)
        
        # Advanced Settings button
        self.advanced_settings_button = QPushButton("Advanced Settings")
        self.advanced_settings_button.setCheckable(True)
        self.advanced_settings_button.clicked.connect(self.toggle_advanced_settings)
        self.processing_layout.addWidget(self.advanced_settings_button)
        
        # Advanced settings widget (hidden by default)
        self.advanced_settings_widget = QWidget()
        self.advanced_settings_layout = QVBoxLayout(self.advanced_settings_widget)
        self.advanced_settings_widget.setVisible(False)
        self.processing_layout.addWidget(self.advanced_settings_widget)
        
        # Add basic circularity and min area settings
        basic_params_layout = QGridLayout()
        
        basic_params_layout.addWidget(QLabel("Min Area:"), 0, 0)
        self.min_area_input = QSpinBox()
        self.min_area_input.setRange(1, 5000)
        self.min_area_input.setValue(500)
        self.min_area_input.valueChanged.connect(self.update_processor_settings)
        basic_params_layout.addWidget(self.min_area_input, 0, 1)
        
        basic_params_layout.addWidget(QLabel("Circularity:"), 1, 0)
        self.circularity_input = QDoubleSpinBox()
        self.circularity_input.setRange(0.1, 1.0)
        self.circularity_input.setSingleStep(0.05)
        self.circularity_input.setValue(0.6)
        self.circularity_input.valueChanged.connect(self.update_processor_settings)
        basic_params_layout.addWidget(self.circularity_input, 1, 1)
        
        self.advanced_settings_layout.addLayout(basic_params_layout)
        
        # Hidden parameters - set but not shown in UI to keep it simple
        self.max_area_input = QSpinBox()
        self.max_area_input.setValue(100000)
        
        self.hue_low1_input = QSpinBox()
        self.hue_low1_input.setValue(0)
        self.hue_high1_input = QSpinBox()
        self.hue_high1_input.setValue(10)
        
        self.hue_low2_input = QSpinBox()
        self.hue_low2_input.setValue(170)
        self.hue_high2_input = QSpinBox()
        self.hue_high2_input.setValue(179)
        
        self.sat_low_input = QSpinBox()
        self.sat_low_input.setValue(120)
        self.sat_high_input = QSpinBox()
        self.sat_high_input.setValue(255)
        
        self.val_low_input = QSpinBox()
        self.val_low_input.setValue(100)
        self.val_high_input = QSpinBox()
        self.val_high_input.setValue(255)
        
        self.red_threshold_input = QSpinBox()
        self.red_threshold_input.setValue(150)
        self.green_max_input = QSpinBox()
        self.green_max_input.setValue(80)
        self.blue_max_input = QSpinBox()
        self.blue_max_input.setValue(80)
        
        self.brightness_threshold_input = QSpinBox()
        self.brightness_threshold_input.setValue(180)
        
        # Debug options visibility
        self.show_mask_checkbox = QCheckBox("Show Detection Overlays")
        self.show_mask_checkbox.setChecked(True)
        self.show_mask_checkbox.stateChanged.connect(self.update_processor_settings)
        self.advanced_settings_layout.addWidget(self.show_mask_checkbox)
        
        # Since we're hiding everything else, just set this value without showing UI
        self.show_intermediate_checkbox = QCheckBox()
        self.show_intermediate_checkbox.setChecked(False)
        
        # Debug options
        self.debug_group = QGroupBox("Status")
        self.debug_layout = QVBoxLayout(self.debug_group)
        self.control_layout.addWidget(self.debug_group)
        
        self.show_fps_checkbox = QCheckBox("Show FPS")
        self.show_fps_checkbox.setChecked(True)
        self.debug_layout.addWidget(self.show_fps_checkbox)
        
        # Add spacer to push controls to top
        self.control_layout.addStretch()
    
    def toggle_advanced_settings(self, checked):
        """Show or hide advanced settings"""
        self.advanced_settings_widget.setVisible(checked)
        
    def create_preview_area(self):
        # Preview container
        self.preview_container = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_container)
        self.top_section.addWidget(self.preview_container, 1)  # Give it stretch priority
        
        # Preview label (will hold the camera image)
        self.preview_label = QLabel("No Camera Feed")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #222222; color: white; font-size: 14px;")
        self.preview_layout.addWidget(self.preview_label, 1)
        
        # FPS label overlay
        self.fps_label = QLabel()
        self.fps_label.setStyleSheet("color: yellow; background-color: rgba(0,0,0,127); padding: 5px;")
        self.fps_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
    def toggle_connection(self):
        if self.stream_thread and self.stream_thread.running:
            # Disconnect
            self.disconnect_camera()
        else:
            # Connect
            self.connect_camera()
    
    def connect_camera(self):
        # Get connection parameters
        host = self.host_input.text()
        port = self.port_input.value()
        path = self.path_input.text()
        # No longer using stream_type dropdown in simplified UI
        
        # Construct URL
        url = f"http://{host}:{port}{path}"
        
        # Create stream thread if needed
        if not self.stream_thread:
            self.stream_thread = CameraStreamThread(url)
            self.stream_thread.frame_ready.connect(self.handle_new_frame)
            self.stream_thread.fps_updated.connect(self.update_stream_fps)
            self.stream_thread.error.connect(self.handle_stream_error)
        else:
            self.stream_thread.set_url(url)
        
        # Start the thread
        self.stream_thread.start()
        
        # Update UI
        self.connect_button.setText("Disconnect")
        self.status_bar.showMessage(f"Connected to {url}")
    
    def disconnect_camera(self):
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread = None
        
        # Reset UI
        self.connect_button.setText("Connect")
        self.preview_label.setText("No Camera Feed")
        self.status_bar.showMessage("Disconnected")
    
    def handle_new_frame(self, frame):
        """Handle new frames from camera - forward to processor or display directly"""
        # Forward frame to calibration thread if active
        if hasattr(self, 'calibration_in_progress') and self.calibration_in_progress:
            if hasattr(self, 'calibration_thread'):
                self.calibration_thread.set_frame(frame)
        
        # Show calibration visualization if enabled
        if hasattr(self, 'show_calibration_vis') and self.show_calibration_vis:
            # Create visualization
            if hasattr(self, 'calibrator') and self.calibrator:
                vis_img = self.calibrator.visualize_calibration(frame.shape[1], frame.shape[0])
                if vis_img is not None:
                    # Show visualization instead of camera frame
                    self.update_display(vis_img)
                    return
        
        # Update the processing thread with the new frame
        if self.processing_thread:
            self.processing_thread.set_frame(frame)
            
        # If processing is disabled, update display directly
        if not self.enable_processing_checkbox.isChecked():
            self.update_display(frame)
    
    def update_display(self, frame):
        """Update the UI with a frame (either processed or raw)"""
        try:
            # Convert to QImage and display
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            
            # Create pixmap and scale only if needed
            pixmap = QPixmap.fromImage(qt_image)
            
            # Only scale if necessary, using fast transformation
            preview_size = self.preview_label.size()
            if preview_size.width() < pixmap.width() or preview_size.height() < pixmap.height():
                pixmap = pixmap.scaled(preview_size, Qt.KeepAspectRatio, Qt.FastTransformation)
            
            # Update preview
            self.preview_label.setPixmap(pixmap)
            
            # Update UI performance metrics
            self.ui_frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.ui_last_fps_time
            
            if elapsed >= 0.5:  # Update twice per second
                self.ui_fps = self.ui_frame_count / elapsed
                self.ui_frame_count = 0
                self.ui_last_fps_time = current_time
        
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def toggle_processing(self, state):
        """Toggle frame processing on/off"""
        if state == Qt.Checked:
            # Create detector with current settings
            self.update_processor_settings()
        else:
            # Clear all processors
            self.processing_thread.clear_processors()
    
    def update_processor_settings(self):
        """Update processor settings when UI controls change"""
        # Only update if processing is enabled
        if self.enable_processing_checkbox.isChecked():
            # Re-create processor with new settings
            self.processing_thread.clear_processors()
            
            # Always use COMBINED_METHOD in simplified UI
            method = EnhancedRedDotDetector.COMBINED_METHOD
            
            # Create detector with all parameters
            detector = EnhancedRedDotDetector(
                # Detection method
                detection_method=method,
                
                # Shape parameters
                min_area=self.min_area_input.value(),
                max_area=self.max_area_input.value(),
                min_circularity=self.circularity_input.value(),
                
                # HSV parameters
                hue_low1=self.hue_low1_input.value(),
                hue_high1=self.hue_high1_input.value(),
                hue_low2=self.hue_low2_input.value(),
                hue_high2=self.hue_high2_input.value(),
                sat_low=self.sat_low_input.value(),
                sat_high=self.sat_high_input.value(),
                val_low=self.val_low_input.value(),
                val_high=self.val_high_input.value(),
                
                # RGB parameters
                red_threshold=self.red_threshold_input.value(),
                green_max=self.green_max_input.value(),
                blue_max=self.blue_max_input.value(),
                
                # Brightness parameters
                brightness_threshold=self.brightness_threshold_input.value(),
                
                # Debug options
                show_mask=self.show_mask_checkbox.isChecked(),
                show_intermediate=self.show_intermediate_checkbox.isChecked()
            )
            
            self.processing_thread.add_processor(detector)
    
    def update_stream_fps(self, fps):
        """Update the FPS value from the stream thread"""
        self.camera_fps = fps
    
    def update_ui_elements(self):
        """Update dynamic UI elements periodically"""
        if hasattr(self, 'camera_fps') and self.show_fps_checkbox.isChecked():
            # Update status bar with FPS information
            if self.stream_thread and self.stream_thread.running:
                processing_status = "ON" if self.enable_processing_checkbox.isChecked() else "OFF"
                
                # Include turret status if connected
                turret_status = " | Turret: Connected" if self.turret_client and self.turret_client.connected else ""
                tracking_status = " (Tracking)" if self.tracking_active else ""
                
                # Include calibration status
                calibration_status = " | Calibrating..." if self.calibration_in_progress else ""
                
                status_text = f"Connected | Camera: {self.camera_fps:.1f} FPS | UI: {self.ui_fps:.1f} FPS | Processing: {processing_status}{turret_status}{tracking_status}{calibration_status}"
                self.status_bar.showMessage(status_text)
    
    def handle_stream_error(self, error_msg):
        """Handle errors from the stream thread"""
        print(f"Stream error: {error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")
    
    def closeEvent(self, event):
        """Clean up resources when window is closed"""
        if self.stream_thread:
            self.stream_thread.stop()
        if self.processing_thread:
            self.processing_thread.stop()
        if self.turret_client:
            self.turret_client.disconnect()
        if hasattr(self, 'calibration_thread') and self.calibration_thread:
            self.calibration_thread.stop()
        event.accept()
    
    # Turret Control Methods
    def toggle_turret_connection(self):
        """Connect or disconnect from the turret controller"""
        if self.turret_client and self.turret_client.connected:
            # Disconnect
            self.turret_client.disconnect()
            self.turret_client = None
            self.turret_connect_button.setText("Connect Turret")
            self.status_bar.showMessage("Turret disconnected")
            self.tracking_checkbox.setChecked(False)
            self.tracking_checkbox.setEnabled(False)
        else:
            # Connect
            host = self.turret_host_input.text()
            port = self.turret_port_input.value()
            
            self.turret_client = TurretClient(host, port)
            self.turret_client.response_callback = self.handle_turret_response
            self.turret_client.error_callback = self.handle_turret_error
            
            if self.turret_client.connect():
                self.turret_connect_button.setText("Disconnect Turret")
                self.status_bar.showMessage(f"Connected to turret at {host}:{port}")
                self.tracking_checkbox.setEnabled(True)
            else:
                self.turret_client = None
                self.status_bar.showMessage(f"Failed to connect to turret at {host}:{port}")
    
    def handle_turret_response(self, response):
        """Handle responses from the turret controller"""
        print(f"Turret: {response}")
        
        # Parse position updates if included
        if response.startswith("ROTATE:"):
            try:
                # Format is typically "POSITION:yaw,pitch"
                pos_part = response.split(":", 1)[1]
                yaw_str, pitch_str = pos_part.split(",")
                self.current_yaw = float(yaw_str)
                self.current_pitch = float(pitch_str)
                print(f"Updated position: yaw={self.current_yaw}, pitch={self.current_pitch}")
            except Exception as e:
                print(f"Error parsing position: {e}")
        
    def handle_turret_error(self, error_msg):
        """Handle errors from the turret controller"""
        print(f"Turret error: {error_msg}")
        self.status_bar.showMessage(f"Turret error: {error_msg}")
        
        # Disconnect on error
        if self.turret_client:
            self.turret_client = None
            self.turret_connect_button.setText("Connect Turret")
            self.tracking_checkbox.setChecked(False)
            self.tracking_checkbox.setEnabled(False)
    
    def move_turret(self, yaw, pitch):
        """Move the turret by the specified angles (relative movement)"""
        if self.turret_client and self.turret_client.connected:
            self.turret_client.rotate(yaw, pitch)
    
    def adjust_turret_position(self, yaw_delta, pitch_delta):
        """Adjust turret position in absolute coordinates (-1 to 0 yaw, 0 to 1 pitch)"""
        if not self.turret_client or not self.turret_client.connected:
            return
            
        # Store current position (or initialize to middle position)
        if not hasattr(self, 'current_yaw'):
            self.current_yaw = -0.5  # Default center position
        if not hasattr(self, 'current_pitch'):
            self.current_pitch = 0.5  # Default center position
            
        # Calculate new position
        new_yaw = max(-1.0, min(0.0, self.current_yaw + yaw_delta))
        new_pitch = max(0.0, min(1.0, self.current_pitch + pitch_delta))
        
        # Update stored position
        self.current_yaw = new_yaw
        self.current_pitch = new_pitch
        
        # Send absolute position command
        self.turret_client.send_command(f"ROTATE:{new_yaw:.4f},{new_pitch:.4f}")
    
    def laser_on(self):
        """Turn on the laser"""
        if self.turret_client and self.turret_client.connected:
            self.turret_client.laser_on()
    
    def laser_off(self):
        """Turn off the laser"""
        if self.turret_client and self.turret_client.connected:
            self.turret_client.laser_off()
    
    def toggle_tracking(self, state):
        """Toggle automatic tracking of red dots"""
        self.tracking_active = (state == Qt.Checked)
        
        if self.tracking_active:
            self.status_bar.showMessage("Auto-tracking enabled")
        else:
            self.status_bar.showMessage("Auto-tracking disabled")
    
    def update_turret_tracking(self):
        """Update turret position based on detected red dots, using calibration if available"""
        if not self.tracking_active or not self.turret_client or not self.turret_client.connected:
            return
        
        # Check if we have any red dots detected
        if not hasattr(self, 'processing_thread') or not self.processing_thread.processors:
            return
            
        for processor in self.processing_thread.processors:
            if isinstance(processor, EnhancedRedDotDetector) and processor.detected_dots:
                # Find the largest dot (likely the main target)
                largest_dot = max(processor.detected_dots, key=lambda x: x['radius'])
                
                # Get dot position
                cx, cy = largest_dot['center']
                
                # Check if enough time has passed since last move
                current_time = time.time()
                if (current_time - self.last_target_time) >= self.move_cooldown:
                    # Use calibration if enabled and available
                    if (hasattr(self, 'use_calibration_checkbox') and 
                        self.use_calibration_checkbox.isChecked() and 
                        hasattr(self, 'calibrator') and 
                        self.calibrator):
                        
                        # Predict turret angles using calibration
                        yaw, pitch = self.calibrator.predict_angles(cx, cy)
                        
                        if yaw is not None and pitch is not None:
                            # Move turret to the predicted position
                            self.turret_client.send_command(f"ROTATE:{yaw:.4f},{pitch:.4f}")
                            self.last_target_time = current_time
                            return
                    
                    # Fall back to original tracking if calibration is not used or fails
                    # Get frame dimensions from the preview
                    frame_width = self.preview_label.width()
                    frame_height = self.preview_label.height()
                    
                    if frame_width <= 0 or frame_height <= 0:
                        return
                        
                    # Calculate normalized position from center
                    center_x = frame_width / 2
                    center_y = frame_height / 2
                    
                    norm_x = (cx - center_x) / center_x
                    norm_y = (cy - center_y) / center_y
                    
                    # Only move if outside deadzone
                    if abs(norm_x) > 0.05 or abs(norm_y) > 0.05:
                        # Map normalized coordinates to turret angles
                        yaw = -0.5 - (norm_x * 0.5)
                        pitch = 0.5 - (norm_y * 0.5)
                        
                        # Ensure values are within bounds
                        yaw = max(-1.0, min(0.0, yaw))
                        pitch = max(0.0, min(1.0, pitch))
                        
                        # Scale the movement
                        movement_scale = max(abs(norm_x), abs(norm_y))
                        yaw_adjustment = (yaw - (-0.5)) * movement_scale * 0.2
                        pitch_adjustment = (pitch - 0.5) * movement_scale * 0.2
                        
                        # Fine-tune the movement
                        yaw = -0.5 + yaw_adjustment
                        pitch = 0.5 + pitch_adjustment
                        
                        # Move the turret
                        self.turret_client.send_command(f"ROTATE:{yaw:.4f},{pitch:.4f}")
                        self.last_target_time = current_time
                break
    
    # Calibration UI and functionality
    def add_calibration_ui(self):
        """Add calibration UI elements to the application"""
        # Create calibration tab in the control panel
        self.calibration_group = QGroupBox("Turret-Camera Calibration")
        self.calibration_layout = QVBoxLayout(self.calibration_group)
        self.control_layout.addWidget(self.calibration_group)
        
        # Initialize calibrator
        self.calibrator = TurretCameraCalibrator()
        self.calibrator.log_callback = self.log_calibration_message
        self.calibrator.progress_callback = self.update_calibration_progress
        
        # Grid size controls
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid Size:"))
        self.grid_rows_input = QSpinBox()
        self.grid_rows_input.setRange(3, 10)
        self.grid_rows_input.setValue(5)
        grid_layout.addWidget(self.grid_rows_input)
        grid_layout.addWidget(QLabel("×"))
        self.grid_cols_input = QSpinBox()
        self.grid_cols_input.setRange(3, 10)
        self.grid_cols_input.setValue(5)
        grid_layout.addWidget(self.grid_cols_input)
        self.calibration_layout.addLayout(grid_layout)
        
        # Connect grid size inputs to update calibrator
        self.grid_rows_input.valueChanged.connect(self.update_calibration_grid)
        self.grid_cols_input.valueChanged.connect(self.update_calibration_grid)
        
        # Wait time controls
        settle_layout = QHBoxLayout()
        settle_layout.addWidget(QLabel("Settle Time:"))
        self.settle_time_input = QDoubleSpinBox()
        self.settle_time_input.setRange(0.1, 5.0)
        self.settle_time_input.setSingleStep(0.1)
        self.settle_time_input.setValue(1.0)
        self.settle_time_input.setSuffix(" s")
        settle_layout.addWidget(self.settle_time_input)
        self.calibration_layout.addLayout(settle_layout)
        
        # Connect settle time input to update calibrator
        self.settle_time_input.valueChanged.connect(self.update_calibration_settings)
        
        # Calibration buttons
        buttons_layout = QHBoxLayout()
        self.start_calibration_button = QPushButton("Start Calibration")
        self.start_calibration_button.clicked.connect(self.start_calibration)
        buttons_layout.addWidget(self.start_calibration_button)
        
        self.stop_calibration_button = QPushButton("Stop Calibration")
        self.stop_calibration_button.clicked.connect(self.stop_calibration)
        self.stop_calibration_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_calibration_button)
        self.calibration_layout.addLayout(buttons_layout)
        
        # Save/Load calibration buttons
        save_load_layout = QHBoxLayout()
        self.save_calibration_button = QPushButton("Save Calibration")
        self.save_calibration_button.clicked.connect(self.save_calibration)
        save_load_layout.addWidget(self.save_calibration_button)
        
        self.load_calibration_button = QPushButton("Load Calibration")
        self.load_calibration_button.clicked.connect(self.load_calibration)
        save_load_layout.addWidget(self.load_calibration_button)
        self.calibration_layout.addLayout(save_load_layout)
        
        # Calibration status and progress
        self.calibration_status_label = QLabel("Calibration not started")
        self.calibration_layout.addWidget(self.calibration_status_label)
        
        self.calibration_progress = QSlider(Qt.Horizontal)
        self.calibration_progress.setRange(0, 100)
        self.calibration_progress.setValue(0)
        self.calibration_progress.setEnabled(False)
        self.calibration_layout.addWidget(self.calibration_progress)
        
        # Add visualization checkbox
        self.show_calibration_vis_checkbox = QCheckBox("Show Calibration Visualization")
        self.show_calibration_vis_checkbox.setChecked(False)
        self.show_calibration_vis_checkbox.stateChanged.connect(self.toggle_calibration_visualization)
        self.calibration_layout.addWidget(self.show_calibration_vis_checkbox)
        
        # Use calibration for tracking checkbox
        self.use_calibration_checkbox = QCheckBox("Use Calibration for Tracking")
        self.use_calibration_checkbox.setChecked(False)
        self.use_calibration_checkbox.setEnabled(False)
        self.calibration_layout.addWidget(self.use_calibration_checkbox)
        
        # Try to load existing calibration
        if os.path.exists(self.calibrator.calibration_file):
            self.load_calibration()

    def update_calibration_grid(self):
        """Update calibration grid size"""
        rows = self.grid_rows_input.value()
        cols = self.grid_cols_input.value()
        self.calibrator.calibration_grid_size = (rows, cols)
        self.log_calibration_message(f"Set calibration grid to {rows}×{cols}")

    def update_calibration_settings(self):
        """Update calibration timing settings"""
        self.calibrator.settle_time = self.settle_time_input.value()
        self.log_calibration_message(f"Set settle time to {self.calibrator.settle_time} seconds")

    def start_calibration(self):
        """Start the calibration process in a separate thread"""
        if not self.turret_client or not self.turret_client.connected:
            self.log_calibration_message("Error: Turret not connected")
            return
            
        # Update settings from UI
        self.update_calibration_grid()
        self.update_calibration_settings()
        
        # Get the red dot detector
        detector = None
        if self.processing_thread and self.processing_thread.processors:
            for processor in self.processing_thread.processors:
                if isinstance(processor, EnhancedRedDotDetector):
                    detector = processor
                    break
                    
        if not detector:
            self.log_calibration_message("Error: No red dot detector available")
            return
        
        # Make sure processing is enabled
        if not self.enable_processing_checkbox.isChecked():
            self.log_calibration_message("Enabling detection for calibration")
            self.enable_processing_checkbox.setChecked(True)
            self.toggle_processing(Qt.Checked)
        
        # Initialize calibration thread
        self.calibration_thread = CalibrationThread(self.calibrator, self.turret_client)
        self.calibration_thread.set_detector(detector)
        
        # Connect signals
        self.calibration_thread.progress_updated.connect(self.update_calibration_progress)
        self.calibration_thread.status_updated.connect(self.log_calibration_message)
        self.calibration_thread.calibration_completed.connect(self.handle_calibration_completed)
        
        # Update UI
        self.start_calibration_button.setEnabled(False)
        self.stop_calibration_button.setEnabled(True)
        self.calibration_progress.setValue(0)
        self.calibration_progress.setEnabled(True)
        self.use_calibration_checkbox.setEnabled(False)
        
        # Set flag for frame forwarding
        self.calibration_in_progress = True
        
        # Start the thread
        self.calibration_thread.start()
        self.log_calibration_message("Calibration thread started")

    def handle_calibration_completed(self, success):
        """Handle calibration completion"""
        self.calibration_in_progress = False
        self.start_calibration_button.setEnabled(True)
        self.stop_calibration_button.setEnabled(False)
        
        if success:
            self.use_calibration_checkbox.setEnabled(True)
            self.log_calibration_message("Calibration completed successfully")
            # Save the calibration automatically
            self.save_calibration()
        else:
            self.log_calibration_message("Calibration did not complete successfully")

    def stop_calibration(self):
        """Stop the calibration process"""
        if hasattr(self, 'calibration_thread') and self.calibration_thread and self.calibration_thread.isRunning():
            self.calibration_thread.stop()
        
        self.calibration_in_progress = False
        self.start_calibration_button.setEnabled(True)
        self.stop_calibration_button.setEnabled(False)
        self.log_calibration_message("Calibration stopped by user")

    def save_calibration(self):
        """Save calibration data to file"""
        success = self.calibrator.save_calibration()
        if success:
            self.use_calibration_checkbox.setEnabled(True)

    def load_calibration(self):
        """Load calibration data from file"""
        success = self.calibrator.load_calibration()
        if success:
            self.use_calibration_checkbox.setEnabled(True)
            self.log_calibration_message("Calibration loaded successfully")

    def log_calibration_message(self, message):
        """Log calibration messages to status label"""
        self.calibration_status_label.setText(message)
        print(f"Calibration: {message}")

    def update_calibration_progress(self, progress):
        """Update calibration progress bar"""
        self.calibration_progress.setValue(int(progress))

    def toggle_calibration_visualization(self, state):
        """Toggle showing the calibration visualization"""
        self.show_calibration_vis = state == Qt.Checked

if __name__ == "__main__":
    # Start the application
    app = QApplication(sys.argv)
    
    # Create a single instance of the application
    viewer = CameraViewerApp()
    viewer.show()
    
    # Start the event loop
    sys.exit(app.exec_())