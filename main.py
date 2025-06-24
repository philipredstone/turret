import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from turret_client import TurretClient
from camera_client import CameraStreamClient

from ui import SimplifiedTurretCalibrationApp

def main():
    """
    Main entry point for the Turret Calibration application.
    
    This function initializes the Qt application, creates the necessary
    client objects for camera and turret communication, and sets up
    the main application window.
    """
    # Create Qt application instance
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('logo.png'))  # Set application icon if available
    
    # Create client instances for hardware communication
    camera_client = CameraStreamClient()  # Handles camera video stream
    turret_client = TurretClient()        # Handles turret control commands
    
    # Create main application window with hardware clients
    window = SimplifiedTurretCalibrationApp(camera_client, turret_client)
    window.show()
    
    # Attempt automatic connection to hardware
    # This is done after window creation so status can be displayed
    try:
        camera_client.connect()  # Connect to camera stream
        turret_client.connect()  # Connect to turret controller
    except Exception as e:
        print(f"Auto-connect failed: {str(e)}")
        pass  # Continue even if auto-connect fails - user can reconnect manually
    
    # Start Qt event loop and exit when window is closed
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()