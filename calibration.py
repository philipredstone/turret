import cv2
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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