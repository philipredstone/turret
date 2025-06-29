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
                            QTextEdit, QMessageBox, QSplitter, QStatusBar, QFileDialog,
                            QSlider, QCheckBox, QTabWidget, QDockWidget, QMenuBar,
                            QMenu, QAction, QDialog, QDialogButtonBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent, QIcon
from PyQt5.QtCore import Qt, QTimer, QMutex, pyqtSignal

from clickable_label import ClickableLabel
from calibration import TurretCalibration
from automatic_calibrator import AutomaticCalibrator
from object_detector import YoloObjectDetector


class ConnectionSettingsDialog(QDialog):
    """
    Dialog for configuring camera and turret connection settings.
    
    This dialog allows users to specify the network addresses and ports
    for both the camera stream and turret controller, and provides
    a "Reconnect Now" button for immediate connection testing.
    """
    
    def __init__(self, parent=None):
        """Initialize the connection settings dialog."""
        super().__init__(parent)
        self.setWindowTitle("Connection Settings")
        self.setModal(True)  # Block interaction with parent window
        self.setMinimumWidth(400)
        self.setWindowIcon(QIcon('logo.png'))
        
        # Create main layout
        layout = QVBoxLayout()
        
        # Connection settings group box
        connection_group = QGroupBox("Connection Settings")
        connection_group.setFont(QFont("Arial", 12, QFont.Bold))
        connection_grid_layout = QGridLayout()
        connection_group.setLayout(connection_grid_layout)
        
        # Camera settings controls
        camera_host_label = QLabel("Camera Host:")
        connection_grid_layout.addWidget(camera_host_label, 0, 0)
        self.camera_host_edit = QLineEdit("127.0.0.1")  # Default localhost
        connection_grid_layout.addWidget(self.camera_host_edit, 0, 1)
        
        camera_port_label = QLabel("Camera Port:")
        connection_grid_layout.addWidget(camera_port_label, 1, 0)
        self.camera_port_spin = QSpinBox()
        self.camera_port_spin.setRange(1, 65535)  # Valid port range
        self.camera_port_spin.setValue(8080)      # Default HTTP port
        connection_grid_layout.addWidget(self.camera_port_spin, 1, 1)
        
        # Turret settings controls
        turret_host_label = QLabel("Turret Host:")
        connection_grid_layout.addWidget(turret_host_label, 2, 0)
        self.turret_host_edit = QLineEdit("127.0.0.1")  # Default localhost
        connection_grid_layout.addWidget(self.turret_host_edit, 2, 1)
        
        turret_port_label = QLabel("Turret Port:")
        connection_grid_layout.addWidget(turret_port_label, 3, 0)
        self.turret_port_spin = QSpinBox()
        self.turret_port_spin.setRange(1, 65535)  # Valid port range
        self.turret_port_spin.setValue(8888)      # Default turret port
        connection_grid_layout.addWidget(self.turret_port_spin, 3, 1)
        
        layout.addWidget(connection_group)
        
        # Dialog buttons with custom reconnect button
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self
        )
        self.reconnect_btn = QPushButton("Reconnect Now")
        button_box.addButton(self.reconnect_btn, QDialogButtonBox.ActionRole)
        
        # Connect standard button signals
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
        self.setLayout(layout)
    
    def get_settings(self):
        """
        Return the current connection settings as a dictionary.
        
        Returns:
            dict: Connection settings with keys: camera_host, camera_port, turret_host, turret_port
        """
        return {
            'camera_host': self.camera_host_edit.text(),
            'camera_port': self.camera_port_spin.value(),
            'turret_host': self.turret_host_edit.text(),
            'turret_port': self.turret_port_spin.value()
        }
    
    def set_settings(self, settings):
        """
        Set the dialog controls from a settings dictionary.
        
        Args:
            settings: Dictionary with connection settings
        """
        self.camera_host_edit.setText(settings['camera_host'])
        self.camera_port_spin.setValue(settings['camera_port'])
        self.turret_host_edit.setText(settings['turret_host'])
        self.turret_port_spin.setValue(settings['turret_port'])


class AdvancedSettingsDialog(QDialog):
    """
    Dialog for configuring advanced laser detection parameters.
    
    This dialog allows fine-tuning of the laser detection algorithm
    parameters, including color thresholds, size filters, and
    debug visualization options.
    """
    
    def __init__(self, parent=None):
        """Initialize the advanced settings dialog."""
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setWindowIcon(QIcon('logo.png'))
        
        # Create main layout
        layout = QVBoxLayout()
        
        # Laser Detection Settings group
        detection_group = QGroupBox("Laser Detection Settings")
        detection_group.setFont(QFont("Arial", 12, QFont.Bold))
        detection_layout = QVBoxLayout()
        detection_group.setLayout(detection_layout)
        
        # Grid layout for parameter controls
        param_layout = QGridLayout()
        
        # Red dominance multiplier - how much red should exceed other colors
        param_layout.addWidget(QLabel("Red Dominance:"), 0, 0)
        self.red_multiplier_spin = QDoubleSpinBox()
        self.red_multiplier_spin.setRange(1.0, 2.0)
        self.red_multiplier_spin.setValue(1.3)  # Default: red must be 1.3x green/blue
        self.red_multiplier_spin.setSingleStep(0.1)
        param_layout.addWidget(self.red_multiplier_spin, 0, 1)
        
        # Red threshold - minimum red channel value
        param_layout.addWidget(QLabel("Red Threshold:"), 1, 0)
        self.red_threshold_spin = QSpinBox()
        self.red_threshold_spin.setRange(0, 255)  # Full color range
        self.red_threshold_spin.setValue(100)     # Default minimum red value
        param_layout.addWidget(self.red_threshold_spin, 1, 1)
        
        # Minimum area filter - smallest detectable laser spot
        param_layout.addWidget(QLabel("Min Area:"), 2, 0)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 1000)
        self.min_area_spin.setValue(5)  # Default 5 pixels minimum
        param_layout.addWidget(self.min_area_spin, 2, 1)
        
        # Circularity threshold - how round the laser spot should be
        param_layout.addWidget(QLabel("Circularity:"), 3, 0)
        self.circularity_spin = QDoubleSpinBox()
        self.circularity_spin.setRange(0.0, 1.0)  # 0=line, 1=perfect circle
        self.circularity_spin.setValue(0.3)       # Default fairly permissive
        self.circularity_spin.setSingleStep(0.1)
        param_layout.addWidget(self.circularity_spin, 3, 1)
        
        detection_layout.addLayout(param_layout)
        
        # Debug visualization option
        self.debug_viz_checkbox = QCheckBox("Show Debug Windows")
        detection_layout.addWidget(self.debug_viz_checkbox)
        
        # Test laser detection button
        self.test_laser_detection_btn = QPushButton("Test Laser Detection")
        detection_layout.addWidget(self.test_laser_detection_btn)
        
        layout.addWidget(detection_group)
        
        # Standard dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
        self.setLayout(layout)
    
    def get_settings(self):
        """
        Return the current advanced settings as a dictionary.
        
        Returns:
            dict: Advanced settings for laser detection
        """
        return {
            'red_multiplier': self.red_multiplier_spin.value(),
            'red_threshold': self.red_threshold_spin.value(),
            'min_area': self.min_area_spin.value(),
            'circularity': self.circularity_spin.value(),
            'debug_viz': self.debug_viz_checkbox.isChecked()
        }
    
    def set_settings(self, settings):
        """
        Set the dialog controls from an advanced settings dictionary.
        
        Args:
            settings: Dictionary with advanced laser detection settings
        """
        self.red_multiplier_spin.setValue(settings['red_multiplier'])
        self.red_threshold_spin.setValue(settings['red_threshold'])
        self.min_area_spin.setValue(settings['min_area'])
        self.circularity_spin.setValue(settings['circularity'])
        self.debug_viz_checkbox.setChecked(settings['debug_viz'])


class CheckerboardSettingsDialog(QDialog):
    """
    Dialog for configuring checkerboard pattern dimensions.
    
    This dialog allows users to specify the width and height of the
    checkerboard pattern used for calibration. This is important
    because different checkerboards have different internal corner counts.
    """
    
    def __init__(self, parent=None):
        """Initialize the checkerboard settings dialog."""
        super().__init__(parent)
        self.setWindowTitle("Checkerboard Settings")
        self.setModal(True)
        self.setWindowIcon(QIcon('logo.png'))
        
        # Create main layout
        layout = QVBoxLayout()
        
        # Checkerboard dimensions group
        settings_group = QGroupBox("Checkerboard Dimensions")
        settings_group.setFont(QFont("Arial", 12, QFont.Bold))
        settings_layout = QGridLayout()
        settings_group.setLayout(settings_layout)
        
        # Board width (number of internal corners horizontally)
        settings_layout.addWidget(QLabel("Board Width:"), 0, 0)
        self.board_width_spin = QSpinBox()
        self.board_width_spin.setRange(2, 20)  # Reasonable range for checkerboards
        self.board_width_spin.setValue(10)     # Default 10 internal corners wide
        settings_layout.addWidget(self.board_width_spin, 0, 1)
        
        # Board height (number of internal corners vertically)
        settings_layout.addWidget(QLabel("Board Height:"), 1, 0)
        self.board_height_spin = QSpinBox()
        self.board_height_spin.setRange(2, 20)  # Reasonable range for checkerboards
        self.board_height_spin.setValue(7)      # Default 7 internal corners tall
        settings_layout.addWidget(self.board_height_spin, 1, 1)
        
        layout.addWidget(settings_group)
        
        # Standard dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
        self.setLayout(layout)
    
    def get_settings(self):
        """
        Return the current checkerboard settings.
        
        Returns:
            dict: Checkerboard dimensions with keys 'width' and 'height'
        """
        return {
            'width': self.board_width_spin.value(),
            'height': self.board_height_spin.value()
        }
    
    def set_settings(self, settings):
        """
        Set the dialog controls from a checkerboard settings dictionary.
        
        Args:
            settings: Dictionary with checkerboard dimensions
        """
        self.board_width_spin.setValue(settings['width'])
        self.board_height_spin.setValue(settings['height'])


class SimplifiedTurretCalibrationApp(QMainWindow):
    """
    Main application window for the Turret Calibration System.
    
    This is the primary user interface that combines all functionality:
    - Live camera feed display with click interaction
    - Manual turret control via WASD keys
    - Checkerboard detection for calibration reference points
    - Manual and automatic calibration workflows
    - Calibration model building and validation
    - Click-to-target functionality for aiming
    - Object detection and automatic targeting with YOLO
    - Settings dialogs for configuration
    - Real-time status monitoring and logging
    """
    
    def __init__(self, camera_client, turret_client):
        """
        Initialize the main application window.
        
        Args:
            camera_client: Camera interface for video stream
            turret_client: Turret interface for movement control
        """
        super().__init__()
        
        # Store hardware interface clients
        self.camera = camera_client
        self.turret = turret_client
        
        # Create calibration system with hardware interfaces
        self.calibration = TurretCalibration(self.camera, self.turret)
        
        # Create automatic calibrator for hands-free operation
        self.auto_calibrator = AutomaticCalibrator(
            camera_client, 
            turret_client, 
            self.calibration
        )
        
        # Create object detector for YOLO-based targeting
        self.object_detector = YoloObjectDetector(model_name="yolo11n")
        self.object_detector.status_callback = self.log_message
        self.object_detector.detection_callback = self.on_object_detected
        
        # Object targeting state
        self.auto_targeting_enabled = False
        self.targeting_loop_timer = QTimer(self)
        self.targeting_loop_timer.timeout.connect(self.targeting_loop)
        
        # Application state variables
        self.laser_on = False                 # Current laser state
        self.selected_corner_idx = -1         # Currently selected checkerboard corner
        self.corners = None                   # Detected checkerboard corners
        self.current_frame = None             # Latest camera frame
        self.calibration_mode = "IDLE"        # Current calibration state
        self.current_yaw = 0                  # Current turret yaw position
        self.current_pitch = 0                # Current turret pitch position
        self.manual_laser_pos = None          # Manually marked laser position
        self.wasd_step_size = 0.005           # Step size for keyboard movement
        self.targeting_mode = False           # Click-to-target mode flag
        self.show_debug_windows = False       # Debug visualization flag
        
        # Connection settings storage for dialogs
        self.connection_settings = {
            'camera_host': '127.0.0.1',
            'camera_port': 8080,
            'turret_host': '127.0.0.1',
            'turret_port': 8888
        }
        
        # Advanced detection settings storage
        self.advanced_settings = {
            'red_multiplier': 1.3,
            'red_threshold': 100,
            'min_area': 5,
            'circularity': 0.3,
            'debug_viz': False
        }
        
        # Initialize user interface
        self.init_ui()
        
        # Create application menu bar
        self.create_menu_bar()
        
        # Timer for updating camera feed display (20 FPS)
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_feed)
        self.camera_timer.start(50)  # 50ms = 20 FPS
        
        # Timer for status updates (1 Hz)
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
        
        # Enable keyboard input for WASD control
        self.setFocusPolicy(Qt.StrongFocus)
    
    def create_menu_bar(self):
        """Create the application menu bar with File, Settings, View, and Help menus."""
        menubar = self.menuBar()
        
        # File menu for calibration save/load operations
        file_menu = menubar.addMenu('File')
        
        # Save calibration action
        save_action = QAction('Save Calibration', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_calibration)
        file_menu.addAction(save_action)
        
        # Load calibration action
        load_action = QAction('Load Calibration', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_calibration)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        # Exit application action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu for configuration dialogs
        settings_menu = menubar.addMenu('Settings')
        
        # Connection settings dialog
        connection_action = QAction('Connection Settings', self)
        connection_action.triggered.connect(self.show_connection_settings)
        settings_menu.addAction(connection_action)
        
        # Advanced laser detection settings dialog
        advanced_action = QAction('Advanced Settings', self)
        advanced_action.triggered.connect(self.show_advanced_settings)
        settings_menu.addAction(advanced_action)
        
        # View menu for display options
        view_menu = menubar.addMenu('View')
        
        # Toggle debug windows action
        debug_action = QAction('Toggle Debug Windows', self)
        debug_action.setCheckable(True)
        debug_action.triggered.connect(self.toggle_debug_from_menu)
        view_menu.addAction(debug_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # About dialog action
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def show_connection_settings(self):
        """Display the connection settings dialog."""
        dialog = ConnectionSettingsDialog(self)
        dialog.set_settings(self.connection_settings)
        
        # Connect the reconnect button to trigger immediate reconnection
        dialog.reconnect_btn.clicked.connect(lambda: self.reconnect_from_dialog(dialog))
        
        if dialog.exec_() == QDialog.Accepted:
            # Update stored settings when dialog is accepted
            self.connection_settings = dialog.get_settings()
            self.log_message("Connection settings updated")
    
    def reconnect_from_dialog(self, dialog):
        """
        Reconnect using settings from dialog and close dialog.
        
        Args:
            dialog: ConnectionSettingsDialog instance
        """
        settings = dialog.get_settings()
        self.connection_settings = settings
        self.reconnect()
        dialog.accept()
    
    def show_advanced_settings(self):
        """Display the advanced laser detection settings dialog."""
        dialog = AdvancedSettingsDialog(self)
        dialog.set_settings(self.advanced_settings)
        
        # Connect the test button to laser detection testing
        dialog.test_laser_detection_btn.clicked.connect(self.test_laser_detection)
        
        if dialog.exec_() == QDialog.Accepted:
            # Update settings and apply them immediately
            self.advanced_settings = dialog.get_settings()
            self.update_detection_parameters()
            self.show_debug_windows = self.advanced_settings['debug_viz']
            if not self.show_debug_windows:
                cv2.destroyAllWindows()  # Close debug windows if disabled
            self.log_message("Advanced settings updated")
    
    def toggle_debug_from_menu(self, checked):
        """
        Toggle debug windows from menu action.
        
        Args:
            checked: Whether debug visualization is enabled
        """
        self.show_debug_windows = checked
        if not self.show_debug_windows:
            cv2.destroyAllWindows()
        self.log_message(f"Debug visualization: {'enabled' if self.show_debug_windows else 'disabled'}")
    
    def show_about(self):
        """Display the about dialog with application information."""
        QMessageBox.about(self, "About Turret Calibration", 
                         "Turret Calibration System\n\n"
                         "Version 1.0\n"
                         "A system for calibrating turret aiming with camera feedback.")
    
    def init_ui(self):
        """Initialize the main user interface layout and controls."""
        # Main window configuration
        self.setWindowTitle("Turret Calibration with Automatic Calibration")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('logo.png'))
        
        # Create central widget and main horizontal layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Left side - Camera view (takes 3/4 of horizontal space)
        camera_frame = QWidget()
        camera_layout = QVBoxLayout()
        camera_frame.setLayout(camera_layout)
        
        # Main camera display with click handling
        self.camera_view_label = ClickableLabel("Camera feed loading...")
        self.camera_view_label.clicked.connect(self.on_image_clicked)
        self.camera_view_label.setAlignment(Qt.AlignCenter)
        self.camera_view_label.setMinimumSize(800, 600)
        self.camera_view_label.setFont(QFont("Arial", 14))
        camera_layout.addWidget(self.camera_view_label)
        
        # Status label below camera view
        self.resolution_label = QLabel("Resolution: - | Use WASD to move turret (Click for targeting when calibrated)")
        self.resolution_label.setFont(QFont("Arial", 10))
        camera_layout.addWidget(self.resolution_label)
        
        main_layout.addWidget(camera_frame, 3)  # 3:1 ratio for camera to controls
        
        # Right side - Tabbed control panel (takes 1/4 of horizontal space)
        control_tabs = QTabWidget()
        control_tabs.setMaximumWidth(400)  # Limit width of control panel

        # Custom styling for tab headers
        tab_bar_style = """
        QTabBar::tab {
            background-color: #f0f0f0;
            color: black;
            padding: 12px 25px;
            margin-right: 2px;
            min-width: 120px;
            font-size: 14px;
            font-weight: bold;
            border: 1px solid #cccccc;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }

        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 1px solid white;
        }

        QTabBar::tab:hover {
            background-color: #e0e0e0;
        }
        """

        # Apply styling to tab bar
        control_tabs.tabBar().setStyleSheet(tab_bar_style)
        
        # Tab 1: Manual Control
        manual_tab = QWidget()
        manual_layout = QVBoxLayout()
        manual_tab.setLayout(manual_layout)
        
        # WASD Controls group
        wasd_group = QGroupBox("WASD Controls")
        wasd_group.setFont(QFont("Arial", 12, QFont.Bold))
        wasd_layout = QVBoxLayout()
        wasd_group.setLayout(wasd_layout)
        
        # Control instructions
        wasd_instructions = QLabel("W: Pitch Up\nS: Pitch Down\nA: Yaw Left\nD: Yaw Right\nQ: Decrease Step\nE: Increase Step\nSpace: Toggle Laser")
        wasd_instructions.setFont(QFont("Arial", 10))
        wasd_layout.addWidget(wasd_instructions)
        
        # Step size control
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
        
        # Current position display
        self.position_label = QLabel(f"Yaw: {self.current_yaw:.3f} | Pitch: {self.current_pitch:.3f}")
        wasd_layout.addWidget(self.position_label)
        
        manual_layout.addWidget(wasd_group)
        
        # Laser control group
        laser_group = QGroupBox("Laser Control")
        laser_group.setFont(QFont("Arial", 12, QFont.Bold))
        laser_layout = QVBoxLayout()
        laser_group.setLayout(laser_layout)
        
        # Laser toggle controls
        laser_toggle_layout = QHBoxLayout()
        self.laser_toggle_btn = QPushButton("Toggle Laser")
        self.laser_toggle_btn.clicked.connect(self.toggle_laser)
        laser_toggle_layout.addWidget(self.laser_toggle_btn)
        
        self.laser_status_label = QLabel("Laser: OFF")
        laser_toggle_layout.addWidget(self.laser_status_label)
        laser_layout.addLayout(laser_toggle_layout)
        
        manual_layout.addWidget(laser_group)
        manual_layout.addStretch()  # Push content to top
        
        # Tab 2: Calibration
        calibration_tab = QWidget()
        calibration_layout = QVBoxLayout()
        calibration_tab.setLayout(calibration_layout)
        
        # Primary Calibration section
        primary_group = QGroupBox("Primary Calibration")
        primary_group.setFont(QFont("Arial", 12, QFont.Bold))
        primary_layout = QVBoxLayout()
        primary_group.setLayout(primary_layout)
        
        # Detect checkerboard button - large and prominent
        self.detect_checkerboard_btn = QPushButton("DETECT CHECKERBOARD")
        self.detect_checkerboard_btn.setMinimumHeight(50)
        self.detect_checkerboard_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.detect_checkerboard_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b3d;
            }
        """)
        self.detect_checkerboard_btn.clicked.connect(self.detect_checkerboard)
        primary_layout.addWidget(self.detect_checkerboard_btn)
        
        # Checkerboard settings button
        self.checkerboard_settings_btn = QPushButton("Checkerboard Settings...")
        self.checkerboard_settings_btn.clicked.connect(self.show_checkerboard_settings)
        primary_layout.addWidget(self.checkerboard_settings_btn)
        
        # Automatic calibration button - single toggle button
        self.auto_calibrate_btn = QPushButton("START")
        self.auto_calibrate_btn.setMinimumHeight(40)
        self.auto_calibrate_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.auto_calibrate_btn.clicked.connect(self.toggle_automatic_calibration)
        self.auto_calibrate_btn.setEnabled(False)  # Disabled until checkerboard detected
        primary_layout.addWidget(self.auto_calibrate_btn)
        
        # Build and Reset buttons in a horizontal layout
        build_reset_layout = QHBoxLayout()
        
        self.build_model_btn = QPushButton("Build Model")
        self.build_model_btn.clicked.connect(self.build_calibration_model)
        self.build_model_btn.setEnabled(False)  # Disabled until enough points collected
        build_reset_layout.addWidget(self.build_model_btn)
        
        self.reset_calibration_btn = QPushButton("Reset")
        self.reset_calibration_btn.clicked.connect(self.reset_calibration)
        build_reset_layout.addWidget(self.reset_calibration_btn)
        
        primary_layout.addLayout(build_reset_layout)
        
        # Calibration status display
        self.calibration_status_label = QLabel("Calibration: Not calibrated")
        self.calibration_status_label.setAlignment(Qt.AlignCenter)
        self.calibration_status_label.setFont(QFont("Arial", 10))
        primary_layout.addWidget(self.calibration_status_label)
        
        calibration_layout.addWidget(primary_group)
        
        # Targeting Mode section
        targeting_group = QGroupBox("Targeting Mode")
        targeting_group.setFont(QFont("Arial", 12, QFont.Bold))
        targeting_layout = QVBoxLayout()
        targeting_group.setLayout(targeting_layout)
        
        # Toggle targeting mode button
        self.targeting_btn = QPushButton("Enable Click-to-Target Mode")
        self.targeting_btn.setMinimumHeight(35)
        self.targeting_btn.setFont(QFont("Arial", 11))
        self.targeting_btn.clicked.connect(self.toggle_targeting_mode)
        self.targeting_btn.setEnabled(False)  # Disabled until calibrated
        targeting_layout.addWidget(self.targeting_btn)
        
        # Targeting instructions
        targeting_instructions = QLabel("When enabled, click anywhere on the image to aim the turret at that position.")
        targeting_instructions.setWordWrap(True)
        targeting_instructions.setFont(QFont("Arial", 10))
        targeting_layout.addWidget(targeting_instructions)
        
        calibration_layout.addWidget(targeting_group)
        
        # Manual Calibration section
        manual_group = QGroupBox("Manual Calibration")
        manual_group.setFont(QFont("Arial", 12, QFont.Bold))
        manual_layout = QVBoxLayout()
        manual_group.setLayout(manual_layout)
        
        # Corner selection controls
        corner_combo_layout = QHBoxLayout()
        corner_label = QLabel("Select Corner:")
        corner_combo_layout.addWidget(corner_label)
        self.corner_combo = QComboBox()
        self.corner_combo.addItem("None")
        self.corner_combo.currentIndexChanged.connect(self.update_selected_corner)
        corner_combo_layout.addWidget(self.corner_combo)
        manual_layout.addLayout(corner_combo_layout)
        
        # Position display
        self.laser_pos_label = QLabel("Position: Not selected")
        manual_layout.addWidget(self.laser_pos_label)
        
        # Capture button
        self.capture_point_btn = QPushButton("Capture Calibration Point")
        self.capture_point_btn.clicked.connect(self.capture_calibration_point)
        self.capture_point_btn.setEnabled(False)
        manual_layout.addWidget(self.capture_point_btn)
        
        # Manual calibration instructions
        manual_instructions = QLabel("1. Select a corner\n2. Click on the image to mark laser position\n3. Capture the calibration point")
        manual_instructions.setFont(QFont("Arial", 10))
        manual_layout.addWidget(manual_instructions)
        
        calibration_layout.addWidget(manual_group)
        
        calibration_layout.addStretch()  # Push content to top

        # Tab 3: Object Detection
        object_detection_tab = QWidget()
        object_detection_layout = QVBoxLayout()
        object_detection_tab.setLayout(object_detection_layout)

        # Model loading group
        model_group = QGroupBox("YOLO Model")
        model_group.setFont(QFont("Arial", 12, QFont.Bold))
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)

        # Model path and load button
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("Model Path:"))
        self.model_path_edit = QLineEdit("yolov5s.pt")  # Default path
        model_path_layout.addWidget(self.model_path_edit)

        model_browse_btn = QPushButton("Browse")
        model_browse_btn.clicked.connect(self.browse_yolo_model)
        model_path_layout.addWidget(model_browse_btn)

        model_layout.addLayout(model_path_layout)

        # Load model button
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_yolo_model)
        model_layout.addWidget(self.load_model_btn)

        # Model info display
        self.model_info_label = QLabel("Model not loaded")
        model_layout.addWidget(self.model_info_label)

        object_detection_layout.addWidget(model_group)

        # Detection Settings group
        detection_group = QGroupBox("Detection Settings")
        detection_group.setFont(QFont("Arial", 12, QFont.Bold))
        detection_layout = QVBoxLayout()
        detection_group.setLayout(detection_layout)

        # Confidence threshold control
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setRange(0.1, 0.99)
        self.confidence_threshold_spin.setValue(0.5)
        self.confidence_threshold_spin.setSingleStep(0.05)
        self.confidence_threshold_spin.valueChanged.connect(self.update_confidence_threshold)
        confidence_layout.addWidget(self.confidence_threshold_spin)
        detection_layout.addLayout(confidence_layout)

        # Target classes selection
        target_classes_layout = QVBoxLayout()
        target_classes_layout.addWidget(QLabel("Target Classes:"))
        self.target_classes_edit = QLineEdit("person,car,dog,cat")  # Default classes
        target_classes_layout.addWidget(self.target_classes_edit)
        detection_layout.addLayout(target_classes_layout)

        # Targeting mode selection
        targeting_mode_layout = QHBoxLayout()
        targeting_mode_layout.addWidget(QLabel("Targeting Mode:"))
        self.targeting_mode_combo = QComboBox()
        self.targeting_mode_combo.addItems(["largest", "center", "closest"])
        self.targeting_mode_combo.currentTextChanged.connect(self.update_targeting_mode)
        targeting_mode_layout.addWidget(self.targeting_mode_combo)
        detection_layout.addLayout(targeting_mode_layout)

        # Apply settings button
        self.apply_detection_settings_btn = QPushButton("Apply Settings")
        self.apply_detection_settings_btn.clicked.connect(self.apply_detection_settings)
        detection_layout.addWidget(self.apply_detection_settings_btn)

        object_detection_layout.addWidget(detection_group)

        # Automatic targeting group
        auto_targeting_group = QGroupBox("Automatic Targeting")
        auto_targeting_group.setFont(QFont("Arial", 12, QFont.Bold))
        auto_targeting_layout = QVBoxLayout()
        auto_targeting_group.setLayout(auto_targeting_layout)

        # Toggle automatic targeting
        self.auto_targeting_btn = QPushButton("Start Automatic Targeting")
        self.auto_targeting_btn.setMinimumHeight(40)
        self.auto_targeting_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.auto_targeting_btn.clicked.connect(self.toggle_auto_targeting)
        self.auto_targeting_btn.setEnabled(False)  # Disabled until model and calibration ready
        auto_targeting_layout.addWidget(self.auto_targeting_btn)

        # Laser during auto targeting option
        self.auto_laser_checkbox = QCheckBox("Turn on laser during targeting")
        self.auto_laser_checkbox.setChecked(True)
        auto_targeting_layout.addWidget(self.auto_laser_checkbox)

        # Auto targeting instructions
        auto_targeting_instructions = QLabel(
            "Automatic targeting will aim the turret at detected objects based on the selected targeting mode."
        )
        auto_targeting_instructions.setWordWrap(True)
        auto_targeting_layout.addWidget(auto_targeting_instructions)

        object_detection_layout.addWidget(auto_targeting_group)
        object_detection_layout.addStretch()  # Push content to top
        
        # Add tabs to tab widget
        control_tabs.addTab(manual_tab, "Manual Control")
        control_tabs.addTab(calibration_tab, "Calibration")
        control_tabs.addTab(object_detection_tab, "Object Detection")
        
        main_layout.addWidget(control_tabs, 1)  # Takes 1/4 of horizontal space
        
        # Bottom area - Log as a dock widget for flexibility
        log_dock = QDockWidget("Log", self)
        log_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        self.log_text.setMaximumHeight(150)
        
        log_dock.setWidget(self.log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)
        
        # Status bar at bottom
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("System initialized")
    
    def show_checkerboard_settings(self):
        """Display the checkerboard dimensions settings dialog."""
        dialog = CheckerboardSettingsDialog(self)
        dialog.set_settings({
            'width': self.calibration.board_size[0],
            'height': self.calibration.board_size[1]
        })
        
        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.get_settings()
            self.calibration.board_size = (settings['width'], settings['height'])
            self.log_message(f"Checkerboard dimensions updated to {settings['width']}x{settings['height']}")
    
    def toggle_automatic_calibration(self):
        """Toggle automatic calibration on/off with visual button state changes."""
        if hasattr(self, 'auto_calibrator') and self.auto_calibrator.is_calibrating:
            # Stop calibration
            self.stop_automatic_calibration()
            self.auto_calibrate_btn.setText("START")
            self.auto_calibrate_btn.setStyleSheet("")  # Reset to default style
        else:
            # Start calibration
            self.start_automatic_calibration()
            self.auto_calibrate_btn.setText("STOP")
            self.auto_calibrate_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
                QPushButton:pressed {
                    background-color: #b91400;
                }
            """)
    
    def update_detection_parameters(self):
        """Update laser detection parameters from advanced settings."""
        detector = self.auto_calibrator.laser_detector
        detector.red_multiplier = self.advanced_settings['red_multiplier']
        detector.red_threshold = self.advanced_settings['red_threshold']
        detector.min_area = self.advanced_settings['min_area']
        detector.circularity_threshold = self.advanced_settings['circularity']
        detector.update_color_ranges()
        self.log_message("Updated detection parameters")
    
    def toggle_debug_visualization(self, state):
        """
        Toggle debug visualization windows.
        
        Args:
            state: Qt checkbox state
        """
        self.show_debug_windows = (state == Qt.Checked)
        if not self.show_debug_windows:
            cv2.destroyAllWindows()
        self.log_message(f"Debug visualization: {'enabled' if self.show_debug_windows else 'disabled'}")
    
    def start_automatic_calibration(self):
        """Start the automatic calibration process with user guidance."""
        if self.corners is None:
            self.log_message("Please detect checkerboard first")
            return
        
        # Determine starting corner
        start_corner_idx = 0
        if self.selected_corner_idx >= 0:
            start_corner_idx = self.selected_corner_idx
            self.log_message(f"Starting automatic calibration from corner {start_corner_idx}")
        else:
            # Ask user to position laser on corner 0
            reply = QMessageBox.question(
                self, 
                "Position Laser",
                "Please manually position the laser on corner 0, then click OK to start automatic calibration.",
                QMessageBox.Ok | QMessageBox.Cancel
            )
            
            if reply != QMessageBox.Ok:
                return
            
            self.log_message("Starting automatic calibration from corner 0")
        
        # Ensure laser is on for calibration
        if not self.laser_on:
            self.toggle_laser()
            time.sleep(0.5)  # Give laser time to turn on
        
        # Disable manual controls during auto calibration
        self.capture_point_btn.setEnabled(False)
        
        # Start automatic calibration from the selected corner
        self.auto_calibrator.start_automatic_calibration(
            self.corners,
            start_corner_idx=start_corner_idx,
            status_callback=self.log_message
        )
    
    def stop_automatic_calibration(self):
        """Stop the automatic calibration process and update UI state."""
        self.log_message("Stopping automatic calibration...")
        self.auto_calibrator.stop_calibration()
        
        # Re-enable manual controls
        self.capture_point_btn.setEnabled(True)
        
        # Check if we have enough points to build the model
        if len(self.calibration.calibration_data) >= 5:
            self.build_model_btn.setEnabled(True)
            self.log_message(f"Automatic calibration stopped. {len(self.calibration.calibration_data)} points collected. Ready to build model.")
        else:
            self.log_message(f"Automatic calibration stopped. {len(self.calibration.calibration_data)} points collected. Need at least 5 points to build model.")
    
    def test_laser_detection(self):
        """Test laser detection on current frame and display results."""
        if self.current_frame is None:
            self.log_message("No frame available")
            return
        
        self.log_message("Testing laser detection...")
        
        # Turn on laser if not already on
        if not self.laser_on:
            self.turret.laser_on()
            time.sleep(0.5)  # Wait for laser to turn on
        
        # Test detection methods
        detector = self.auto_calibrator.laser_detector
        
        # Enable debug visualization temporarily
        old_debug_state = self.show_debug_windows
        self.show_debug_windows = True
        
        # Try main detection method
        result = detector.detect_laser_spot(self.current_frame, debug=True)
        if result is not None:
            self.log_message(f"Main detection: Laser found at {result}")
        else:
            self.log_message("Main detection: No laser detected")
        
        # Try simple detection method
        simple_result = detector.detect_laser_spot_simple(self.current_frame)
        if simple_result is not None:
            self.log_message(f"Simple detection: Laser found at {simple_result}")
        else:
            self.log_message("Simple detection: No laser detected")
        
        # Restore debug state
        self.show_debug_windows = old_debug_state
    
    def update_camera_feed(self):
        """Update camera feed display with latest frame and overlays."""
        # Get new frame from camera
        frame = self.camera.get_frame()
        if frame is None:
            return
        
        self.current_frame = frame.copy()
        
        # Create display copy for overlays
        display_frame = self.current_frame.copy()
        
        # Draw detected checkerboard corners if available
        if hasattr(self, 'corners') and self.corners is not None:
            # Draw all corners with numbering
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

        # Draw object detection results if available
        if hasattr(self, 'object_detector') and self.object_detector.is_detecting:
            target = self.object_detector.get_target()
            if target is not None:
                center_x, center_y = target['center']
                x1, y1, x2, y2 = target['box']
                
                # Draw target box with thicker lines
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Draw target center with crosshairs
                cv2.circle(display_frame, (center_x, center_y), 8, (0, 255, 255), -1)
                cv2.line(display_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 255), 2)
                cv2.line(display_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 255), 2)
                
                # Draw target info
                label = f"TARGET: {target['class_name']} ({target['confidence']:.2f})"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add mode-specific overlay text
        if self.targeting_mode:
            cv2.putText(display_frame, "TARGETING MODE - CLICK TO AIM", (100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "WASD: Move | Q/E: Adjust Step | Space: Toggle Laser", (100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Convert BGR to RGB for Qt display
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Update display widget
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_view_label.setPixmap(QPixmap.fromImage(qt_image))
        
        # Update resolution and mode info
        mode_text = "TARGETING" if self.targeting_mode else self.calibration_mode
        self.resolution_label.setText(f"Resolution: {w}x{h} | Mode: {mode_text} | Step: {self.wasd_step_size:.3f}")
    
    def on_image_clicked(self, event):
        """
        Handle clicks on the camera image for targeting or manual calibration.
        
        Args:
            event: Mouse click event containing position information
        """
        # Convert click position from widget coordinates to image coordinates
        image_coords = self._map_label_click_to_image_coords(event.pos())
        
        if image_coords is None:
            self.log_message("Click was outside the image area or image not available.")
            return

        img_x, img_y = image_coords

        # Handle targeting mode (aim turret at clicked position)
        if self.targeting_mode and self.calibration.is_calibrated:
            self.calibration.aim_at_target(img_x, img_y)
            self.log_message(f"Aiming at ({img_x}, {img_y})")
            return
            
        # Handle manual laser marking for calibration
        can_proceed_with_marking = False
        if self.calibration_mode in ["CORNER_SELECTED", "DETECTED"]:
            can_proceed_with_marking = True
        elif hasattr(self, 'corners') and self.corners is not None:
            self.calibration_mode = "DETECTED" 
            self.log_message(f"Switched to DETECTED mode as corners are present for manual marking.")
            can_proceed_with_marking = True

        if not can_proceed_with_marking:
            self.log_message("Manual marking: Please detect checkerboard first, or select a corner if already detected.")
            return
        
        # Store manual laser position
        self.manual_laser_pos = (img_x, img_y)
        
        # Auto-select nearest corner if none is selected
        if self.selected_corner_idx == -1 and hasattr(self, 'corners') and self.corners is not None:
            nearest_idx = -1
            min_dist = float('inf')
            
            # Find nearest corner to click position
            for i, (corner_img_x, corner_img_y) in enumerate(self.corners):
                dist = ((corner_img_x - img_x) ** 2 + (corner_img_y - img_y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            # Auto-select if click is reasonably close to a corner
            if min_dist < 50: 
                self.selected_corner_idx = nearest_idx
                if nearest_idx + 1 < self.corner_combo.count():
                    self.corner_combo.setCurrentIndex(nearest_idx + 1)
                else:
                    self.log_message(f"Warning: nearest_idx+1 {nearest_idx+1} out of bounds for corner_combo (count {self.corner_combo.count()})")
                self.log_message(f"Auto-selected nearest corner {nearest_idx} to click at ({img_x},{img_y})")
        
        # Update UI and enable capture
        self.laser_pos_label.setText(f"Position: ({img_x}, {img_y})")
        self.capture_point_btn.setEnabled(True)
        self.log_message(f"Laser position marked at ({img_x}, {img_y})")
    
    def keyPressEvent(self, event):
        """
        Handle key press events for WASD turret control.
        
        Args:
            event: Key press event
        """
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
            super().keyPressEvent(event)  # Pass unhandled keys to parent
    
    def on_step_size_changed(self, value):
        """
        Update step size when spin box value changes.
        
        Args:
            value: New step size value
        """
        self.wasd_step_size = value
    
    def move_turret_to(self, yaw, pitch):
        """
        Move the turret to the specified position and update UI.
        
        Args:
            yaw: Target yaw angle
            pitch: Target pitch angle
        """
        success = self.turret.rotate(yaw, pitch)
        
        if success:
            self.current_yaw = yaw
            self.current_pitch = pitch
            self.position_label.setText(f"Yaw: {self.current_yaw:.3f} | Pitch: {self.current_pitch:.3f}")
    
    def detect_checkerboard(self):
        """Detect checkerboard pattern in current frame and update UI."""
        if self.current_frame is None:
            self.log_message("No frame available")
            return
        
        self.log_message(f"Detecting checkerboard with size {self.calibration.board_size[0]}x{self.calibration.board_size[1]}...")
        
        # Attempt checkerboard detection
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
            
            # Enable automatic calibration button
            self.auto_calibrate_btn.setEnabled(True)
        else:
            self.log_message("Failed to detect checkerboard")
    
    def update_selected_corner(self, index):
        """
        Update the selected corner index from combo box.
        
        Args:
            index: Selected index in combo box (0 = "None", 1+ = corner indices)
        """
        if index == 0:  # "None" option
            self.selected_corner_idx = -1
            self.calibration_mode = "DETECTED"
            self.capture_point_btn.setEnabled(False)
        else:
            self.selected_corner_idx = index - 1  # Adjust for "None" entry
            self.calibration_mode = "CORNER_SELECTED"
            
            # Reset manual laser position when corner changes
            self.manual_laser_pos = None
            self.laser_pos_label.setText("Position: Not selected")
            self.capture_point_btn.setEnabled(False)
            
            self.log_message(f"Selected corner {self.selected_corner_idx}")
    
    def toggle_laser(self):
        """Toggle the laser on/off and update UI status."""
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
        """Capture a calibration point with the selected corner and manually marked laser position."""
        if self.corners is None:
            self.log_message("No checkerboard detected")
            return
            
        if self.manual_laser_pos is None:
            self.log_message("No laser position marked")
            return
        
        # Use selected corner position, or manual position if no corner selected
        if self.selected_corner_idx >= 0:
            corner_x, corner_y = self.corners[self.selected_corner_idx]
        else:
            corner_x, corner_y = self.manual_laser_pos
        
        # Get the current turret position
        current_yaw = self.current_yaw
        current_pitch = self.current_pitch
        
        # Add to calibration data
        self.calibration.calibration_data.append((corner_x, corner_y, current_yaw, current_pitch))
        
        # Log the capture
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
        """Build the calibration model from collected data points."""
        if len(self.calibration.calibration_data) < 5:
            self.log_message("Not enough calibration points. Need at least 5.")
            return
        
        self.log_message("Building calibration model...")
        success = self.calibration.build_transformation_model()
        
        if success:
            self.log_message("Calibration model built successfully")
            self.calibration_status_label.setText("Calibration: Calibrated")
            self.calibration_mode = "CALIBRATED"
            self.targeting_btn.setEnabled(True)  # Enable targeting mode
            
            # Enable auto targeting button if model is loaded
            if self.calibration.is_calibrated and hasattr(self, 'object_detector') and self.object_detector.model is not None:
                self.auto_targeting_btn.setEnabled(True)
        else:
            self.log_message("Failed to build calibration model")
    
    def save_calibration(self):
        """Save calibration model to a file using file dialog."""
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
        """Load calibration model from a file using file dialog."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Load Calibration Model", 
                                                 "", "Calibration Files (*.cal)")
        
        if file_path:
            success = self.calibration.load_calibration(file_path)
            
            if success:
                self.log_message(f"Calibration loaded from {file_path}")
                self.calibration_status_label.setText("Calibration: Calibrated")
                self.calibration_mode = "CALIBRATED"
                self.targeting_btn.setEnabled(True)  # Enable targeting mode
                
                # Enable auto targeting button if model is loaded
                if self.object_detector.model is not None:
                    self.auto_targeting_btn.setEnabled(True)
            else:
                self.log_message("Failed to load calibration")
    
    def toggle_targeting_mode(self):
        """Toggle targeting mode for click-to-aim functionality."""
        if not self.calibration.is_calibrated:
            self.log_message("Calibration required for targeting mode")
            return
        
        self.targeting_mode = not self.targeting_mode
        
        if self.targeting_mode:
            self.log_message("Targeting mode enabled - Click on image to aim turret")
            self.targeting_btn.setText("Disable Click-to-Target Mode")
            self.targeting_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
            """)
        else:
            self.log_message("Targeting mode disabled")
            self.targeting_btn.setText("Enable Click-to-Target Mode")
            self.targeting_btn.setStyleSheet("")  # Reset to default style
    
    def reset_calibration(self):
        """Reset all calibration data and return to initial state."""
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
        self.targeting_btn.setEnabled(False)
        self.targeting_mode = False
        
        # Reset auto targeting
        if self.auto_targeting_enabled:
            self.toggle_auto_targeting()
        
        self.log_message("Calibration data reset")
        self.calibration_status_label.setText("Calibration: Not calibrated")
    
    def browse_yolo_model(self):
        """Browse for YOLO model file using file dialog."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select YOLO Model", "", "PyTorch Models (*.pt);;All Files (*)"
        )
        
        if file_path:
            self.model_path_edit.setText(file_path)

    def load_yolo_model(self):
        """Load YOLO model from specified path."""
        model_path = self.model_path_edit.text().strip()
        
        if not model_path:
            self.log_message("Please specify a model path")
            return
        
        success = self.object_detector.load_model(model_path)
        
        if success:
            self.model_info_label.setText(
                f"Model loaded with {len(self.object_detector.class_names)} classes"
            )
            # Enable auto targeting if system is calibrated
            self.auto_targeting_btn.setEnabled(self.calibration.is_calibrated)
            self.apply_detection_settings()
        else:
            self.model_info_label.setText("Failed to load model")

    def update_confidence_threshold(self, value):
        """Update confidence threshold when spin box value changes."""
        self.object_detector.set_confidence_threshold(value)

    def update_targeting_mode(self, mode):
        """Update targeting mode when combo box value changes."""
        self.object_detector.set_targeting_mode(mode)

    def apply_detection_settings(self):
        """Apply detection settings from UI controls."""
        # Update confidence threshold
        threshold = self.confidence_threshold_spin.value()
        self.object_detector.set_confidence_threshold(threshold)
        
        # Update target classes
        classes_text = self.target_classes_edit.text()
        if classes_text:
            classes = [c.strip() for c in classes_text.split(",")]
            self.object_detector.set_target_classes(classes)
        
        # Update targeting mode
        mode = self.targeting_mode_combo.currentText()
        self.object_detector.set_targeting_mode(mode)
        
        self.log_message("Detection settings applied")

    def toggle_auto_targeting(self):
        """Toggle automatic targeting on/off."""
        if self.auto_targeting_enabled:
            # Stop automatic targeting
            self.auto_targeting_enabled = False
            self.auto_targeting_btn.setText("Start Automatic Targeting")
            self.auto_targeting_btn.setStyleSheet("")  # Reset to default style
            
            # Stop targeting loop
            self.targeting_loop_timer.stop()
            
            # Stop detection if running
            if self.object_detector.is_detecting:
                self.object_detector.stop_detection()
            
            # Turn off laser if it was on for targeting
            if self.laser_on and self.auto_laser_checkbox.isChecked():
                self.toggle_laser()
            
            self.log_message("Automatic targeting stopped")
        else:
            # Check if calibrated
            if not self.calibration.is_calibrated:
                self.log_message("Calibration required for automatic targeting")
                QMessageBox.warning(self, "Not Calibrated", 
                                  "Please calibrate the system before using automatic targeting")
                return
            
            # Check if model is loaded
            if self.object_detector.model is None:
                self.log_message("Please load a YOLO model first")
                QMessageBox.warning(self, "No Model", 
                                  "Please load a YOLO model before using automatic targeting")
                return
            
            # Start automatic targeting
            self.auto_targeting_enabled = True
            self.auto_targeting_btn.setText("Stop Automatic Targeting")
            self.auto_targeting_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
                QPushButton:pressed {
                    background-color: #b91400;
                }
            """)
            
            # Turn on laser if checked
            if not self.laser_on and self.auto_laser_checkbox.isChecked():
                self.toggle_laser()
            
            # Start detection if not already running
            if not self.object_detector.is_detecting:
                self.object_detector.start_detection(self.camera.get_frame)
            
            # Start targeting loop
            self.targeting_loop_timer.start(100)  # 10 Hz targeting loop
            
            self.log_message("Automatic targeting started")

    def targeting_loop(self):
        """Main loop for automatic targeting."""
        if not self.auto_targeting_enabled:
            return
        
        # Get the best target
        target = self.object_detector.get_target()
        
        if target is None:
            return
        
        # Get target center point
        center_x, center_y = target['center']
        
        # Use calibration to aim at target
        self.calibration.aim_at_target(center_x, center_y)
        
        # Log targeting info occasionally
        if time.time() % 2 < 0.1:  # Log roughly every 2 seconds
            self.log_message(
                f"Targeting {target['class_name']} at ({center_x}, {center_y}) "
                f"with confidence {target['confidence']:.2f}"
            )

    def on_object_detected(self, detections, vis_frame):
        """
        Callback when objects are detected.
        
        Args:
            detections: List of detected objects
            vis_frame: Visualization frame with detection overlays
        """
        # Use this for additional visualization or feedback if needed
        pass
    
    def update_status(self):
        """Update status information and check for automatic calibration completion."""
        # Check if automatic calibration is done
        if hasattr(self, 'auto_calibrator') and not self.auto_calibrator.is_calibrating and self.auto_calibrate_btn.text() == "STOP":
            # Auto calibration finished - update UI
            self.auto_calibrate_btn.setText("START")
            self.auto_calibrate_btn.setStyleSheet("")  # Reset to default style
            
            # Enable build model button if we have enough points
            if len(self.calibration.calibration_data) >= 5:
                self.build_model_btn.setEnabled(True)
                
            if self.calibration.is_calibrated:
                self.targeting_btn.setEnabled(True)
                self.calibration_status_label.setText("Calibration: Calibrated")
                self.calibration_mode = "CALIBRATED"
        
        # Update connection status in status bar
        camera_status = "Connected" if self.camera.connected else "Disconnected"
        turret_status = "Connected" if self.turret.connected else "Disconnected"
        
        status_text = f"Camera: {camera_status} | Turret: {turret_status}"
        
        # Add FPS if available
        if self.camera.fps > 0:
            status_text += f" | FPS: {self.camera.fps:.1f}"
        
        # Add calibration status
        if self.calibration.is_calibrated:
            status_text += " | Calibrated: YES"
        else:
            status_text += f" | Calibrated: NO | Points: {len(self.calibration.calibration_data)}/5+"
        
        self.statusBar.showMessage(status_text)
    
    def reconnect(self):
        """Reconnect to the camera and turret with current settings."""
        # Disconnect first
        self.camera.disconnect()
        self.turret.disconnect()
        
        # Get connection parameters from stored settings
        camera_host = self.connection_settings['camera_host']
        camera_port = self.connection_settings['camera_port']
        turret_host = self.connection_settings['turret_host']
        turret_port = self.connection_settings['turret_port']
        
        # Reconnect camera
        self.log_message(f"Reconnecting to camera at {camera_host}:{camera_port}")
        if not self.camera.connect(camera_host, camera_port):
            self.log_message("Failed to connect to camera")
            QMessageBox.warning(self, "Connection Error",
                              "Failed to connect to camera")
        else:
            self.log_message("Camera connected successfully")
        
        # Reconnect turret
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
        """
        Add a timestamped message to the log display and console.
        
        Args:
            message: Message to log
        """
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.ensureCursorVisible()  # Auto-scroll to bottom
        print(f"[{timestamp}] {message}")    # Also print to console
    
    def closeEvent(self, event):
        """
        Handle cleanup when closing the application.
        
        Args:
            event: Close event
        """
        # Stop automatic calibration if running
        if hasattr(self, 'auto_calibrator') and self.auto_calibrator.is_calibrating:
            self.auto_calibrator.stop_calibration()
        
        # Stop object detection if running
        if hasattr(self, 'object_detector') and self.object_detector.is_detecting:
            self.object_detector.stop_detection()

        if hasattr(self, 'targeting_loop_timer'):
            self.targeting_loop_timer.stop()
        
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
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        event.accept()
    
    def _map_label_click_to_image_coords(self, click_pos):
        """
        Map a click position on the camera_view_label to image coordinates.
        
        This handles the coordinate transformation from widget space to image space,
        accounting for image scaling and centering within the label.
        
        Args:
            click_pos: QPoint containing click position in widget coordinates
            
        Returns:
            tuple or None: (x, y) image coordinates, or None if click outside image
        """
        label_widget = self.camera_view_label
        label_size = label_widget.size()
        
        current_pixmap = label_widget.pixmap()
        if not current_pixmap or current_pixmap.isNull():
            self.log_message("Debug: _map_label_click_to_image_coords - No pixmap in label.")
            return None # No pixmap available
            
        pixmap_size = current_pixmap.size() # Original image size
        
        if pixmap_size.width() == 0 or pixmap_size.height() == 0:
            self.log_message(f"Debug: _map_label_click_to_image_coords - Invalid pixmap size: {pixmap_size.width()}x{pixmap_size.height()}")
            return None # Invalid pixmap size

        # Calculate the offset of the pixmap within the label (due to Qt.AlignCenter)
        offset_x = (label_size.width() - pixmap_size.width()) / 2.0
        offset_y = (label_size.height() - pixmap_size.height()) / 2.0

        # Convert click coordinates from label space to pixmap space
        img_x_float = click_pos.x() - offset_x
        img_y_float = click_pos.y() - offset_y

        # Check if the click is within the actual bounds of the pixmap
        if 0 <= img_x_float < pixmap_size.width() and \
           0 <= img_y_float < pixmap_size.height():
            return int(img_x_float), int(img_y_float)
        else:
            self.log_message(f"Debug: Click ({click_pos.x()},{click_pos.y()}) mapped to ({img_x_float:.1f},{img_y_float:.1f}) is outside pixmap ({pixmap_size.width()},{pixmap_size.height()})")
            return None
    
    def show_debug_visualization(self, frame, debug_info):
        """
        Display debug visualization windows for laser detection analysis.
        
        This shows various stages of the detection process in separate OpenCV windows,
        helping to debug and tune detection parameters.
        
        Args:
            frame: Original camera frame
            debug_info: Dictionary containing debug visualization data
        """
        if not self.show_debug_windows:
            return
        
        # Show different detection masks and stages
        if 'red_mask' in debug_info and debug_info['red_mask'] is not None:
            cv2.imshow("Red Mask", debug_info['red_mask'])
        
        if 'red_dominant' in debug_info and debug_info['red_dominant'] is not None:
            cv2.imshow("Red Dominant", debug_info['red_dominant'])
        
        if 'combined_mask' in debug_info and debug_info['combined_mask'] is not None:
            cv2.imshow("Combined Mask", debug_info['combined_mask'])
        
        if 'final_mask1' in debug_info and debug_info['final_mask1'] is not None:
            cv2.imshow("Final Mask (with brightness)", debug_info['final_mask1'])
        
        if 'final_mask2' in debug_info and debug_info['final_mask2'] is not None:
            cv2.imshow("Final Mask (without brightness)", debug_info['final_mask2'])
        
        # Show detection result with contours and final position
        if frame is not None:
            result_frame = frame.copy()
            
            # Draw all detected contours
            if 'contours1' in debug_info:
                cv2.drawContours(result_frame, debug_info['contours1'], -1, (0, 255, 0), 1)
            if 'contours2' in debug_info:
                cv2.drawContours(result_frame, debug_info['contours2'], -1, (0, 255, 255), 1)
            
            # Draw best selected contour
            if 'best_contour' in debug_info and debug_info['best_contour'] is not None:
                cv2.drawContours(result_frame, [debug_info['best_contour']], -1, (255, 0, 0), 2)
            
            # Draw detected laser position
            if 'laser_pos' in debug_info and debug_info['laser_pos'] is not None:
                pos = debug_info['laser_pos']
                cv2.circle(result_frame, pos, 10, (0, 0, 255), 2)
                cv2.putText(result_frame, f"Laser: {pos}", (pos[0]+15, pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow("Detection Result", result_frame)