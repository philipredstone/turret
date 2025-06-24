import cv2
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import griddata


class TurretCalibration:
    """
    Handles calibration of turret aiming system using multiple mathematical approaches.
    
    This class provides comprehensive calibration functionality including:
    - Checkerboard detection for reference points
    - Multiple calibration models (linear regression, RANSAC, homography)
    - Prediction of turret angles from image coordinates
    - Model persistence (save/load)
    - Calibration quality assessment
    """
    
    def __init__(self, camera_client, turret_client):
        """
        Initialize the calibration system.
        
        Args:
            camera_client: Interface to camera for getting frames
            turret_client: Interface to turret for movement commands
        """
        self.camera = camera_client
        self.turret = turret_client
        self.calibration_data = []  # List of (img_x, img_y, turret_yaw, turret_pitch) tuples
        self.board_size = (9, 6)    # Default checkerboard size (width, height)
        
        # Different calibration models - multiple approaches for robustness
        self.pan_model = None           # Linear regression for yaw/pan
        self.tilt_model = None          # Linear regression for pitch/tilt
        self.homography_matrix = None   # Homographic transformation matrix
        self.ransac_pan_model = None    # RANSAC regression for yaw (outlier-resistant)
        self.ransac_tilt_model = None   # RANSAC regression for pitch (outlier-resistant)
        
        self.is_calibrated = False
        self.poly_features = PolynomialFeatures(degree=2)  # For non-linear relationships
        
        # Store detected corners for reference
        self.detected_corners = None
        
        # Model selection flags
        self.use_ransac = True      # Enable RANSAC models for outlier resistance
        self.use_homography = True  # Enable homography transformation
    
    def detect_checkerboard(self, frame):
        """
        Detect checkerboard pattern in the given frame.
        
        This method attempts to find a checkerboard pattern using OpenCV's built-in
        detector. If the configured size fails, it tries common alternative sizes.
        
        Args:
            frame: Input image frame from camera
            
        Returns:
            tuple: (corners_list, display_frame) where corners_list is [(x,y), ...] 
                   and display_frame shows detected corners, or (None, display_frame) if failed
        """
        if frame is None:
            print("No frame provided to detect_checkerboard")
            return None, None
            
        # Convert to grayscale for better corner detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        
        # Try detection with the configured board size
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # If detection fails, try alternative common board sizes
        if not ret:
            print(f"Failed to detect with size {self.board_size}, trying alternatives...")
            board_sizes_to_try = [
                (self.board_size[0]-1, self.board_size[1]),      # One less column
                (self.board_size[0], self.board_size[1]-1),      # One less row
                (8, 6), (9, 6), (7, 5), (6, 9)                  # Common sizes
            ]
            
            for size in board_sizes_to_try:
                ret, corners = cv2.findChessboardCorners(gray, size, 
                                                        cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                        cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret:
                    print(f"Checkerboard found with pattern size {size}")
                    self.board_size = size  # Update to working size
                    break
        
        if ret and corners is not None and len(corners) > 0:
            # Refine corner positions to subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            try:
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                corners = refined_corners
            except Exception as e:
                print(f"Warning: Corner refinement failed: {str(e)}")
            
            # Draw the detected corners on display frame for visualization
            cv2.drawChessboardCorners(display_frame, self.board_size, corners, ret)
            
            # Convert to simple list format
            corners_list = [(corner[0][0], corner[0][1]) for corner in corners]
            self.detected_corners = corners_list
            print(f"Checkerboard pattern found! Detected {len(corners_list)} corners.")
            return corners_list, display_frame
        else:
            print("Failed to detect any checkerboard pattern.")
            return None, display_frame
    
    def compute_homography(self):
        """
        Compute homography matrix from calibration data.
        
        Homography provides a direct geometric transformation between image coordinates
        and turret coordinates, which can be more accurate than polynomial regression
        for planar transformations.
        
        Returns:
            bool: True if homography computed successfully, False otherwise
        """
        if len(self.calibration_data) < 4:  # Minimum 4 points needed for homography
            return False
        
        # Extract image points and corresponding turret points
        image_points = np.float32([(d[0], d[1]) for d in self.calibration_data])
        turret_points = np.float32([(d[2], d[3]) for d in self.calibration_data])
        
        # Compute homography using RANSAC for outlier resistance
        self.homography_matrix, mask = cv2.findHomography(image_points, turret_points, cv2.RANSAC, 5.0)
        
        return self.homography_matrix is not None
    
    def predict_position_homography(self, image_x, image_y):
        """
        Predict turret position using homography transformation.
        
        Args:
            image_x, image_y: Target position in image coordinates
            
        Returns:
            tuple or None: (pan, tilt) angles, or None if prediction failed
        """
        if self.homography_matrix is None:
            return None
        
        # Convert point to homogeneous coordinates for matrix multiplication
        point = np.array([[image_x], [image_y], [1.0]], dtype=np.float32)
        
        # Apply homography transformation
        transformed = self.homography_matrix @ point
        
        # Convert back to 2D coordinates
        if transformed[2, 0] != 0:
            pan = transformed[0, 0] / transformed[2, 0]
            tilt = transformed[1, 0] / transformed[2, 0]
            return pan, tilt
        
        return None
    
    def build_transformation_model(self):
        """
        Build transformation models using multiple approaches for robustness.
        
        This method creates several different models:
        1. Standard polynomial regression
        2. RANSAC regression (outlier-resistant)
        3. Homography transformation (geometric)
        
        Returns:
            bool: True if model building successful, False otherwise
        """
        if len(self.calibration_data) < 5:  # Need minimum points for reliable model
            return False
        
        # Convert calibration data to numpy array for processing
        data = np.array(self.calibration_data)
        
        # Extract features (image coordinates) and targets (turret angles)
        X = data[:, 0:2]        # Image x,y coordinates
        Y_pan = data[:, 2]      # Pan/Yaw angles
        Y_tilt = data[:, 3]     # Tilt/Pitch angles
        
        # Create polynomial features for non-linear relationships
        # This allows the model to capture distortion and non-linear mapping
        X_poly = self.poly_features.fit_transform(X)
        
        # Build standard linear regression models
        self.pan_model = LinearRegression().fit(X_poly, Y_pan)
        self.tilt_model = LinearRegression().fit(X_poly, Y_tilt)
        
        # Build RANSAC regression models for outlier resistance
        if self.use_ransac:
            self.ransac_pan_model = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=0.5,          # Use 50% of data for model fitting
                residual_threshold=0.001  # Points within this threshold are inliers
            ).fit(X_poly, Y_pan)
            
            self.ransac_tilt_model = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=0.5,
                residual_threshold=0.001
            ).fit(X_poly, Y_tilt)
        
        # Compute homography matrix if enabled
        if self.use_homography:
            self.compute_homography()
        
        self.is_calibrated = True

        # Print detailed statistics about model quality
        self.print_calibration_stats()

        return True
    
    def predict_position(self, image_x, image_y, method='auto'):
        """
        Predict turret position using the best available method.
        
        Args:
            image_x, image_y: Target position in image coordinates
            method: Prediction method ('auto', 'linear', 'ransac', 'homography')
            
        Returns:
            tuple or None: (pan_angle, tilt_angle) or None if prediction failed
        """
        if not self.is_calibrated:
            return None
        
        # Auto-select best available method
        if method == 'auto':
            # Prefer homography if available (often most accurate)
            if self.use_homography and self.homography_matrix is not None:
                result = self.predict_position_homography(image_x, image_y)
                if result is not None:
                    return result
            
            # Fall back to RANSAC if available (outlier-resistant)
            if self.use_ransac and self.ransac_pan_model is not None:
                method = 'ransac'
            else:
                method = 'linear'
        
        # Transform input point to polynomial features
        target_point = np.array([[image_x, image_y]])
        target_poly = self.poly_features.transform(target_point)
        
        # Predict using specified method
        if method == 'ransac' and self.ransac_pan_model is not None:
            pan_angle = self.ransac_pan_model.predict(target_poly)[0]
            tilt_angle = self.ransac_tilt_model.predict(target_poly)[0]
        else:
            pan_angle = self.pan_model.predict(target_poly)[0]
            tilt_angle = self.tilt_model.predict(target_poly)[0]
        
        return pan_angle, tilt_angle
    
    def interpolate_position(self, image_x, image_y):
        """
        Interpolate position using calibrated points (alternative method).
        
        This method uses scipy's griddata for interpolation, which can be useful
        when you want a purely data-driven approach without model fitting.
        
        Args:
            image_x, image_y: Target position in image coordinates
            
        Returns:
            tuple or None: (pan, tilt) angles or None if interpolation failed
        """
        if len(self.calibration_data) < 4:
            return None
        
        # Extract calibration points
        points = np.array([(d[0], d[1]) for d in self.calibration_data])
        pan_values = np.array([d[2] for d in self.calibration_data])
        tilt_values = np.array([d[3] for d in self.calibration_data])
        
        # Attempt interpolation with different methods
        try:
            # Try cubic interpolation first (smoothest)
            pan = griddata(points, pan_values, (image_x, image_y), method='cubic')
            tilt = griddata(points, tilt_values, (image_x, image_y), method='cubic')
            
            # Fall back to linear if cubic produces NaN
            if np.isnan(pan) or np.isnan(tilt):
                pan = griddata(points, pan_values, (image_x, image_y), method='linear')
                tilt = griddata(points, tilt_values, (image_x, image_y), method='linear')
            
            return pan, tilt
        except:
            return None
    
    def estimate_calibration_error(self, test_point):
        """
        Estimate the calibration error at a given point.
        
        This provides a rough estimate of how accurate the calibration might be
        at a specific image location, based on distance to nearest calibrated point.
        
        Args:
            test_point: Image coordinates (x, y) to test
            
        Returns:
            float: Estimated error magnitude (lower is better)
        """
        if not self.is_calibrated:
            return float('inf')
        
        # Get prediction for this point
        predicted = self.predict_position(test_point[0], test_point[1])
        if predicted is None:
            return float('inf')
        
        # Find nearest calibrated point
        min_distance = float('inf')
        nearest_actual = None
        
        for cal_point in self.calibration_data:
            dist = np.sqrt((cal_point[0] - test_point[0])**2 + (cal_point[1] - test_point[1])**2)
            if dist < min_distance:
                min_distance = dist
                nearest_actual = (cal_point[2], cal_point[3])
        
        if nearest_actual is None:
            return float('inf')
        
        # Estimate error based on prediction vs nearest actual
        error = np.sqrt((predicted[0] - nearest_actual[0])**2 + (predicted[1] - nearest_actual[1])**2)
        
        # Scale error by distance to nearest calibrated point
        # Points far from calibrated data are less reliable
        scaled_error = error * (1 + min_distance / 100)
        
        return scaled_error
    
    def get_essential_calibration_points(self):
        """
        Get the most important points for calibration.
        
        This identifies key points that should be calibrated for best results:
        - Four corners (most important for geometric stability)
        - Edge midpoints (for edge distortion)
        - Center point (for central accuracy)
        
        Returns:
            list: Indices of essential calibration points
        """
        if self.detected_corners is None:
            return []
        
        corners = self.detected_corners
        num_corners = len(corners)
        essential = []
        
        # Four corners - most critical for geometric accuracy
        if num_corners >= 4:
            essential.extend([
                0,                                    # Top-left
                self.board_size[0]-1,                # Top-right
                num_corners-self.board_size[0],      # Bottom-left
                num_corners-1                        # Bottom-right
            ])
        
        # Middle of edges (if board is large enough)
        if self.board_size[0] >= 5 and self.board_size[1] >= 5:
            # Top middle
            essential.append(self.board_size[0] // 2)
            # Bottom middle
            essential.append(num_corners - self.board_size[0] // 2 - 1)
            # Left middle
            essential.append((self.board_size[1] // 2) * self.board_size[0])
            # Right middle
            essential.append((self.board_size[1] // 2) * self.board_size[0] + self.board_size[0] - 1)
        
        # Center point
        center_row = self.board_size[1] // 2
        center_col = self.board_size[0] // 2
        center_idx = center_row * self.board_size[0] + center_col
        if center_idx < num_corners:
            essential.append(center_idx)
        
        return list(set(essential))  # Remove duplicates
    
    def aim_at_target(self, target_x, target_y):
        """
        Aim the turret at a target point in the camera view.
        
        Args:
            target_x, target_y: Target position in image coordinates
            
        Returns:
            bool: True if aiming command sent successfully, False otherwise
        """
        if not self.is_calibrated:
            return False
        
        # Predict turret angles for target position
        result = self.predict_position(target_x, target_y)
        if result is None:
            return False
        
        pan_angle, tilt_angle = result
        
        # Send rotation command to turret
        return self.turret.rotate(pan_angle, tilt_angle)
    
    def save_calibration(self, filename):
        """
        Save the calibration model to a file for later use.
        
        Args:
            filename: Path to save calibration data
            
        Returns:
            bool: True if save successful, False otherwise
        """
        if not self.is_calibrated:
            return False
        
        # Package all calibration data for saving
        calibration_data = {
            'pan_model': self.pan_model,
            'tilt_model': self.tilt_model,
            'poly_features': self.poly_features,
            'calibration_data': self.calibration_data,
            'homography_matrix': self.homography_matrix,
            'ransac_pan_model': self.ransac_pan_model,
            'ransac_tilt_model': self.ransac_tilt_model,
            'board_size': self.board_size
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(calibration_data, f)
            return True
        except Exception as e:
            print(f"Error saving calibration: {str(e)}")
            return False
    
    def load_calibration(self, filename):
        """
        Load a calibration model from a file.
        
        Args:
            filename: Path to calibration file
            
        Returns:
            bool: True if load successful, False otherwise
        """
        try:
            with open(filename, 'rb') as f:
                calibration_data = pickle.load(f)
            
            # Restore all calibration components
            self.pan_model = calibration_data['pan_model']
            self.tilt_model = calibration_data['tilt_model']
            self.poly_features = calibration_data['poly_features']
            self.calibration_data = calibration_data['calibration_data']
            
            # Load additional models if available (for backward compatibility)
            self.homography_matrix = calibration_data.get('homography_matrix')
            self.ransac_pan_model = calibration_data.get('ransac_pan_model')
            self.ransac_tilt_model = calibration_data.get('ransac_tilt_model')
            self.board_size = calibration_data.get('board_size', self.board_size)
            
            self.is_calibrated = True
            return True
        except Exception as e:
            print(f"Error loading calibration: {str(e)}")
            return False
        
    def print_calibration_stats(self):
        """
        Print comprehensive statistics about the calibration model quality.
        
        This includes R² scores, RMSE, maximum errors, and RANSAC inlier statistics
        to help assess calibration quality and identify potential issues.
        """
        if not self.is_calibrated:
            print("Model is not calibrated yet.")
            return
        
        print("\n=== Calibration Statistics ===")
        print(f"Number of calibration points: {len(self.calibration_data)}")
        print(f"Checkerboard size: {self.board_size}")
        
        # Extract data for error computation
        data = np.array(self.calibration_data)
        X = data[:, 0:2]        # Camera x,y coordinates
        Y_pan = data[:, 2]      # Pan/Yaw angles
        Y_tilt = data[:, 3]     # Tilt/Pitch angles
        
        # Transform to polynomial features
        X_poly = self.poly_features.transform(X)
        
        # Linear model statistics
        if self.pan_model is not None:
            print("\n--- Linear Model Statistics ---")
            pan_pred = self.pan_model.predict(X_poly)
            tilt_pred = self.tilt_model.predict(X_poly)
            
            # Compute various error metrics
            pan_mse = np.mean((Y_pan - pan_pred) ** 2)
            tilt_mse = np.mean((Y_tilt - tilt_pred) ** 2)
            pan_abs_max = np.max(np.abs(Y_pan - pan_pred))
            tilt_abs_max = np.max(np.abs(Y_tilt - tilt_pred))
            
            # R² score (coefficient of determination) - closer to 1.0 is better
            pan_r2 = self.pan_model.score(X_poly, Y_pan)
            tilt_r2 = self.tilt_model.score(X_poly, Y_tilt)
            
            print(f"Pan R² score: {pan_r2:.4f}")
            print(f"Tilt R² score: {tilt_r2:.4f}")
            print(f"Pan RMSE: {np.sqrt(pan_mse):.6f} degrees")
            print(f"Tilt RMSE: {np.sqrt(tilt_mse):.6f} degrees")
            print(f"Pan max absolute error: {pan_abs_max:.6f} degrees")
            print(f"Tilt max absolute error: {tilt_abs_max:.6f} degrees")
        
        # RANSAC model statistics
        if self.use_ransac and self.ransac_pan_model is not None:
            print("\n--- RANSAC Model Statistics ---")
            ransac_pan_pred = self.ransac_pan_model.predict(X_poly)
            ransac_tilt_pred = self.ransac_tilt_model.predict(X_poly)
            
            # Compute error metrics for RANSAC models
            ransac_pan_mse = np.mean((Y_pan - ransac_pan_pred) ** 2)
            ransac_tilt_mse = np.mean((Y_tilt - ransac_tilt_pred) ** 2)
            ransac_pan_abs_max = np.max(np.abs(Y_pan - ransac_pan_pred))
            ransac_tilt_abs_max = np.max(np.abs(Y_tilt - ransac_tilt_pred))
            
            # R² scores for RANSAC models
            ransac_pan_r2 = self.ransac_pan_model.score(X_poly, Y_pan)
            ransac_tilt_r2 = self.ransac_tilt_model.score(X_poly, Y_tilt)
            
            # Inlier statistics - shows how many points were considered good data
            pan_inlier_mask = self.ransac_pan_model.inlier_mask_
            tilt_inlier_mask = self.ransac_tilt_model.inlier_mask_
            pan_inliers = np.sum(pan_inlier_mask)
            tilt_inliers = np.sum(tilt_inlier_mask)
            
            print(f"Pan R² score: {ransac_pan_r2:.4f}")
            print(f"Tilt R² score: {ransac_tilt_r2:.4f}")
            print(f"Pan RMSE: {np.sqrt(ransac_pan_mse):.6f} degrees")
            print(f"Tilt RMSE: {np.sqrt(ransac_tilt_mse):.6f} degrees")
            print(f"Pan max absolute error: {ransac_pan_abs_max:.6f} degrees")
            print(f"Tilt max absolute error: {ransac_tilt_abs_max:.6f} degrees")
            print(f"Pan inliers: {pan_inliers}/{len(X)} ({pan_inliers/len(X)*100:.1f}%)")
            print(f"Tilt inliers: {tilt_inliers}/{len(X)} ({tilt_inliers/len(X)*100:.1f}%)")
        
        # Homography matrix statistics
        if self.use_homography and self.homography_matrix is not None:
            print("\n--- Homography Matrix Statistics ---")
            try:
                # Calculate condition number - indicates numerical stability
                singular_values = np.linalg.svd(self.homography_matrix, compute_uv=False)
                condition_number = float(singular_values[0] / singular_values[-1])
                
                # Calculate reprojection error using homography
                homography_errors = []
                for point in self.calibration_data:
                    image_x, image_y, pan, tilt = point
                    predicted = self.predict_position_homography(image_x, image_y)
                    if predicted is not None:
                        pred_pan, pred_tilt = predicted
                        error = np.sqrt((pan - pred_pan)**2 + (tilt - pred_tilt)**2)
                        homography_errors.append(error)
                
                if homography_errors:
                    avg_error = np.mean(homography_errors)
                    max_error = np.max(homography_errors)
                    print(f"Homography matrix condition number: {condition_number:.2f}")
                    print(f"Homography average reprojection error: {avg_error:.6f} degrees")
                    print(f"Homography maximum reprojection error: {max_error:.6f} degrees")
            except Exception as e:
                print(f"Error calculating homography statistics: {str(e)}")