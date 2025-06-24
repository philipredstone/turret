import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from turret_client import TurretClient
from camera_client import CameraStreamClient

from ui import SimplifiedTurretCalibrationApp

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('logo.png'))
    
    camera_client = CameraStreamClient()
    turret_client = TurretClient()
    
    window = SimplifiedTurretCalibrationApp(camera_client, turret_client)
    window.show()
    
    try:
        camera_client.connect()
        turret_client.connect()
    except Exception as e:
        print(f"Auto-connect failed: {str(e)}")
        pass
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()