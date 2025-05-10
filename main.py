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
                            QSlider, QCheckBox, QTabWidget, QDockWidget)
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeyEvent
from PyQt5.QtCore import Qt, QTimer, QMutex, pyqtSignal


from turret_client import TurretClient
from camera_client import CameraStreamClient

from ui import SimplifiedTurretCalibrationApp

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