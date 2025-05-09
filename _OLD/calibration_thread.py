
import threading
from PyQt5.QtCore import QThread, pyqtSignal

class CalibrationThread(QThread):
    """Thread class for handling calibration process without blocking the UI"""
    # Define signals for communication with the main thread
    progress_updated = pyqtSignal(float)  # For progress updates
    status_updated = pyqtSignal(str)      # For status message updates
    calibration_completed = pyqtSignal(bool)  # True if successful, False otherwise
    
    def __init__(self, calibrator, turret_client):
        super().__init__()
        self.calibrator = calibrator
        self.turret_client = turret_client
        self.running = False
        self.detector = None     # Will store the detector
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def set_detector(self, detector):
        """Set the detector to use for laser dot detection"""
        self.detector = detector
        
    def set_frame(self, frame):
        """Set the current frame to process"""
        with self.frame_lock:
            if frame is not None:
                self.current_frame = frame.copy()
    
    def stop(self):
        """Stop the calibration process"""
        self.running = False
        self.calibrator.stop_calibration()
        # Wait for thread to finish
        self.wait()
        
    def run(self):
        """Main calibration thread process"""
        self.running = True
        
        # Prepare for calibration
        success = self.calibrator.start_calibration(self.turret_client)
        if not success:
            self.status_updated.emit("Failed to start calibration")
            self.calibration_completed.emit(False)
            return
            
        self.status_updated.emit("Calibration started")
        
        # Main calibration loop
        while self.running and self.calibrator.calibration_running:
            # Make sure we have a frame and detector
            if self.detector is None:
                self.status_updated.emit("Error: No detector available")
                self.calibration_completed.emit(False)
                return
                
            # Get the latest frame
            with self.frame_lock:
                if self.current_frame is not None:
                    frame_copy = self.current_frame.copy()
                else:
                    # If no frame is available, wait for the next one
                    import time
                    time.sleep(0.1)
                    continue
                    
            # Process calibration step
            try:
                step_success = self.calibrator.process_calibration_step(frame_copy, self.detector)
                
                # Emit progress signal
                if self.calibrator.calibration_points:
                    progress = (self.calibrator.current_point_index / len(self.calibrator.calibration_points)) * 100
                    self.progress_updated.emit(progress)
                
                # Short sleep to avoid CPU hogging and allow UI updates
                import time
                time.sleep(0.1)
                
            except Exception as e:
                self.status_updated.emit(f"Error during calibration: {str(e)}")
                self.calibration_completed.emit(False)
                return
        
        # Calibration finished or stopped
        if not self.running:
            self.status_updated.emit("Calibration stopped by user")
            self.calibration_completed.emit(False)
        else:
            # Train the model now that we've collected all data points
            success = self.calibrator.train_calibration_model()
            if success:
                self.status_updated.emit("Calibration completed successfully")
                self.calibration_completed.emit(True)
            else:
                self.status_updated.emit("Failed to train calibration model")
                self.calibration_completed.emit(False)