import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pickle
import os
import time
import cv2

class TurretCameraCalibrator:
    """
    Calibration system that maps camera coordinates to turret angles using Gaussian Process Regression
    """
    def __init__(self, turret_client=None):
        # Turret control
        self.turret_client = turret_client
        
        # Calibration parameters
        self.calibration_grid_size = (5, 5)  # 5x5 grid of calibration points
        self.yaw_range = (-0.9, -0.1)  # Range of turret yaw values (-1 to 0)
        self.pitch_range = (0.1, 0.9)  # Range of turret pitch values (0 to 1)
        self.settle_time = 1.0  # Time to wait for turret to settle at each position
        self.laser_warmup_time = 0.5  # Time to wait after turning on laser
        
        # Calibration state
        self.calibration_running = False
        self.current_point_index = 0
        self.calibration_points = []  # List of (yaw, pitch) points to visit
        self.calibration_data = []  # List of (yaw, pitch, pixel_x, pixel_y) tuples
        
        # Current point being calibrated
        self.current_yaw = None
        self.current_pitch = None
        self.current_phase = 0  # 0: moving, 1: settling, 2: laser on, 3: detecting
        self.last_phase_time = 0
        
        # Gaussian Process Regression models
        self.gp_yaw = None
        self.gp_pitch = None
        
        # Default calibration file path
        self.calibration_file = "turret_camera_calibration.pkl"
        
        # Callback for logging/UI updates
        self.log_callback = lambda msg: print(msg)
        self.progress_callback = lambda percent: None
        
    def generate_calibration_grid(self):
        """Generate a grid of calibration points in turret coordinates"""
        rows, cols = self.calibration_grid_size
        
        # Create evenly spaced grid
        yaws = np.linspace(self.yaw_range[0], self.yaw_range[1], cols)
        pitches = np.linspace(self.pitch_range[0], self.pitch_range[1], rows)
        
        # Create grid of points
        points = []
        for pitch in pitches:
            for yaw in yaws:
                points.append((yaw, pitch))
        
        return points
    
    def start_calibration(self, turret_client):
        """Start the calibration process"""
        self.turret_client = turret_client
        
        if not self.turret_client or not self.turret_client.connected:
            self.log_callback("Error: Turret not connected")
            return False
        
        # Reset calibration state
        self.calibration_running = True
        self.current_point_index = 0
        self.calibration_data = []
        self.current_phase = 0
        self.last_phase_time = time.time()
        
        # Generate grid of points to visit
        self.calibration_points = self.generate_calibration_grid()
        self.log_callback(f"Starting calibration with {len(self.calibration_points)} points")
        
        # Ensure laser is off at start
        self.turret_client.laser_off()
        
        return True
    
    def stop_calibration(self):
        """Stop the calibration process"""
        self.calibration_running = False
        if self.turret_client and self.turret_client.connected:
            self.turret_client.laser_off()
        self.log_callback("Calibration stopped")
    
    def process_calibration_step(self, frame, detector):
        """Process one step of the calibration - returns True to continue, False if done"""
        if not self.calibration_running:
            return False
            
        if self.current_point_index >= len(self.calibration_points):
            # Calibration complete
            self.calibration_running = False
            if self.turret_client and self.turret_client.connected:
                self.turret_client.laser_off()
            self.log_callback("Calibration data collection complete")
            return False
        
        # Get current time to check for phase transitions
        current_time = time.time()
        
        # Phase 0: Move to next position
        if self.current_phase == 0:
            # Get current calibration point
            self.current_yaw, self.current_pitch = self.calibration_points[self.current_point_index]
            
            # Move turret to position
            if self.turret_client and self.turret_client.connected:
                self.turret_client.send_command(f"ROTATE:{self.current_yaw:.4f},{self.current_pitch:.4f}")
                self.log_callback(f"Moving to calibration point {self.current_point_index+1}/{len(self.calibration_points)}: yaw={self.current_yaw:.2f}, pitch={self.current_pitch:.2f}")
            
            # Move to next phase
            self.current_phase = 1
            self.last_phase_time = current_time
            return True
        
        # Phase 1: Wait for turret to settle
        elif self.current_phase == 1:
            if current_time - self.last_phase_time >= self.settle_time:
                # Move to next phase
                self.current_phase = 2
                self.last_phase_time = current_time
                
                # Turn on laser
                if self.turret_client and self.turret_client.connected:
                    self.turret_client.laser_on()
                    self.log_callback("Laser on, waiting for warmup")
            return True
        
        # Phase 2: Wait for laser warmup
        elif self.current_phase == 2:
            if current_time - self.last_phase_time >= self.laser_warmup_time:
                # Move to next phase
                self.current_phase = 3
                self.last_phase_time = current_time
                self.log_callback("Detecting laser dot")
            return True
        
        # Phase 3: Detect laser dot
        elif self.current_phase == 3:
            # Process frame to detect laser dot
            if detector and detector.detected_dots:
                # Find the brightest/most likely laser dot
                laser_dot = max(detector.detected_dots, key=lambda x: x['area'])
                
                # Get dot position
                cx, cy = laser_dot['center']
                
                # Add to calibration data
                self.calibration_data.append((self.current_yaw, self.current_pitch, cx, cy))
                self.log_callback(f"Detected laser at ({cx}, {cy}) for yaw={self.current_yaw:.2f}, pitch={self.current_pitch:.2f}")
                
                # Turn off laser before moving to next point
                if self.turret_client and self.turret_client.connected:
                    self.turret_client.laser_off()
                
                # Move to next point
                self.current_point_index += 1
                
                # Update progress
                progress = (self.current_point_index / len(self.calibration_points)) * 100
                self.progress_callback(progress)
                
                # Reset phase for next point
                self.current_phase = 0
                return True
            else:
                # If we've been trying to detect for too long, move on
                if current_time - self.last_phase_time >= 3.0:  # 3 seconds timeout
                    self.log_callback("No laser dot detected, skipping point")
                    
                    # Turn off laser
                    if self.turret_client and self.turret_client.connected:
                        self.turret_client.laser_off()
                        
                    # Move to next point
                    self.current_point_index += 1
                    
                    # Update progress
                    progress = (self.current_point_index / len(self.calibration_points)) * 100
                    self.progress_callback(progress)
                    
                    # Reset phase for next point
                    self.current_phase = 0
                    
                return True
        
        return True
    
    def train_calibration_model(self):
        """Train Gaussian Process Regression model from collected data"""
        if len(self.calibration_data) < 4:
            self.log_callback("Not enough calibration data points")
            return False
            
        try:
            # Prepare data
            X = np.array([(x, y) for _, _, x, y in self.calibration_data])
            y_yaw = np.array([yaw for yaw, _, _, _ in self.calibration_data])
            y_pitch = np.array([pitch for _, pitch, _, _ in self.calibration_data])
            
            # Create and train model for yaw prediction
            kernel_yaw = RBF(length_scale=100.0) + WhiteKernel(noise_level=0.01)
            self.gp_yaw = GaussianProcessRegressor(kernel=kernel_yaw, normalize_y=True, n_restarts_optimizer=10)
            self.gp_yaw.fit(X, y_yaw)
            
            # Create and train model for pitch prediction
            kernel_pitch = RBF(length_scale=100.0) + WhiteKernel(noise_level=0.01)
            self.gp_pitch = GaussianProcessRegressor(kernel=kernel_pitch, normalize_y=True, n_restarts_optimizer=10)
            self.gp_pitch.fit(X, y_pitch)
            
            return True
        except Exception as e:
            self.log_callback(f"Error training model: {str(e)}")
            return False
    
    def predict_angles(self, pixel_x, pixel_y):
        """Predict turret angles for a given pixel coordinate"""
        if not self.gp_yaw or not self.gp_pitch:
            return None, None
            
        try:
            # Make prediction
            X_pred = np.array([[pixel_x, pixel_y]])
            yaw_pred, yaw_std = self.gp_yaw.predict(X_pred, return_std=True)
            pitch_pred, pitch_std = self.gp_pitch.predict(X_pred, return_std=True)
            
            # Ensure predictions are within valid ranges
            yaw = max(self.yaw_range[0], min(self.yaw_range[1], yaw_pred[0]))
            pitch = max(self.pitch_range[0], min(self.pitch_range[1], pitch_pred[0]))
            
            return yaw, pitch
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None, None
    
    def save_calibration(self, filepath=None):
        """Save calibration data and model to file"""
        if filepath is None:
            filepath = self.calibration_file
            
        try:
            # Save calibration data, parameters, and trained models
            calibration_info = {
                'calibration_data': self.calibration_data,
                'grid_size': self.calibration_grid_size,
                'yaw_range': self.yaw_range,
                'pitch_range': self.pitch_range,
                'gp_yaw': self.gp_yaw,
                'gp_pitch': self.gp_pitch
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(calibration_info, f)
                
            self.log_callback(f"Calibration saved to {filepath}")
            return True
        except Exception as e:
            self.log_callback(f"Error saving calibration: {str(e)}")
            return False
    
    def load_calibration(self, filepath=None):
        """Load calibration data and model from file"""
        if filepath is None:
            filepath = self.calibration_file
            
        if not os.path.exists(filepath):
            self.log_callback(f"Calibration file {filepath} not found")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                calibration_info = pickle.load(f)
                
            # Load calibration parameters
            self.calibration_data = calibration_info['calibration_data']
            self.calibration_grid_size = calibration_info['grid_size']
            self.yaw_range = calibration_info['yaw_range']
            self.pitch_range = calibration_info['pitch_range']
            
            # Load models
            self.gp_yaw = calibration_info['gp_yaw']
            self.gp_pitch = calibration_info['gp_pitch']
            
            self.log_callback(f"Calibration loaded from {filepath}")
            return True
        except Exception as e:
            self.log_callback(f"Error loading calibration: {str(e)}")
            return False
    
    def visualize_calibration(self, frame_width, frame_height):
        """Create a visualization of the calibration data and model"""
        if not self.calibration_data:
            return None
            
        # Create a blank image
        vis_img = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
        
        # Draw grid lines
        grid_color = (200, 200, 200)
        for i in range(0, frame_width, 50):
            cv2.line(vis_img, (i, 0), (i, frame_height), grid_color, 1)
        for i in range(0, frame_height, 50):
            cv2.line(vis_img, (0, i), (frame_width, i), grid_color, 1)
        
        # Draw calibration points (camera coordinates)
        for yaw, pitch, cx, cy in self.calibration_data:
            cv2.circle(vis_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.putText(vis_img, f"({yaw:.2f}, {pitch:.2f})", (int(cx)+10, int(cy)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # If we have trained models, visualize the mapping
        if self.gp_yaw and self.gp_pitch:
            # Create a grid of points to test
            x_grid = np.linspace(0, frame_width, 20)
            y_grid = np.linspace(0, frame_height, 15)
            
            for x in x_grid:
                for y in y_grid:
                    # Predict turret angles for this pixel
                    yaw, pitch = self.predict_angles(x, y)
                    if yaw is not None and pitch is not None:
                        # Draw a small dot
                        cv2.circle(vis_img, (int(x), int(y)), 2, (0, 128, 0), -1)
                        
                        # Draw connecting lines from predictions to actual calibration points
                        for cal_yaw, cal_pitch, cal_x, cal_y in self.calibration_data:
                            if abs(yaw - cal_yaw) < 0.05 and abs(pitch - cal_pitch) < 0.05:
                                cv2.line(vis_img, (int(x), int(y)), (int(cal_x), int(cal_y)), 
                                         (128, 128, 0), 1, cv2.LINE_AA)
        
        # Add title and info
        cv2.putText(vis_img, "Turret-Camera Calibration Map", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(vis_img, f"Calibration Points: {len(self.calibration_data)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return vis_img