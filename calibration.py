import cv2
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import griddata


class TurretCalibration:
    def __init__(self, camera_client, turret_client):
        self.camera = camera_client
        self.turret = turret_client
        self.calibration_data = []
        self.board_size = (9, 6)  # Default checkerboard size
        
        # Models for different approaches
        self.pan_model = None
        self.tilt_model = None
        self.homography_matrix = None
        self.ransac_pan_model = None
        self.ransac_tilt_model = None
        
        self.is_calibrated = False
        self.poly_features = PolynomialFeatures(degree=2)
        
        # Store corners for reference
        self.detected_corners = None
        
        # Model selection
        self.use_ransac = True
        self.use_homography = True
    
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
                (8, 6), (9, 6), (7, 5), (6, 9)
            ]
            
            for size in board_sizes_to_try:
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
            self.detected_corners = corners_list
            print(f"Checkerboard pattern found! Detected {len(corners_list)} corners.")
            return corners_list, display_frame
        else:
            print("Failed to detect any checkerboard pattern.")
            return None, display_frame
    
    def compute_homography(self):
        """Compute homography matrix from calibration data"""
        if len(self.calibration_data) < 4:
            return False
        
        # Extract image points and turret points
        image_points = np.float32([(d[0], d[1]) for d in self.calibration_data])
        turret_points = np.float32([(d[2], d[3]) for d in self.calibration_data])
        
        # Find homography
        self.homography_matrix, mask = cv2.findHomography(image_points, turret_points, cv2.RANSAC, 5.0)
        
        return self.homography_matrix is not None
    
    def predict_position_homography(self, image_x, image_y):
        """Predict turret position using homography"""
        if self.homography_matrix is None:
            return None
        
        # Convert point to homogeneous coordinates
        point = np.array([[image_x], [image_y], [1.0]], dtype=np.float32)
        
        # Apply homography
        transformed = self.homography_matrix @ point
        
        # Convert back to 2D
        if transformed[2, 0] != 0:
            pan = transformed[0, 0] / transformed[2, 0]
            tilt = transformed[1, 0] / transformed[2, 0]
            return pan, tilt
        
        return None
    
    def build_transformation_model(self):
        """Build transformation models using multiple approaches"""
        if len(self.calibration_data) < 5:
            return False
        
        data = np.array(self.calibration_data)
        
        # Extract features and targets
        X = data[:, 0:2]  # Camera x,y coordinates
        Y_pan = data[:, 2]  # Pan/Yaw angles
        Y_tilt = data[:, 3]  # Tilt/Pitch angles
        
        # Polynomial features
        X_poly = self.poly_features.fit_transform(X)
        
        # Standard regression
        self.pan_model = LinearRegression().fit(X_poly, Y_pan)
        self.tilt_model = LinearRegression().fit(X_poly, Y_tilt)
        
        # RANSAC regression for robustness
        if self.use_ransac:
            self.ransac_pan_model = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=0.5,
                residual_threshold=0.001
            ).fit(X_poly, Y_pan)
            
            self.ransac_tilt_model = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=0.5,
                residual_threshold=0.001
            ).fit(X_poly, Y_tilt)
        
        # Compute homography if enabled
        if self.use_homography:
            self.compute_homography()
        
        self.is_calibrated = True

        self.print_calibration_stats()

        return True
    
    def predict_position(self, image_x, image_y, method='auto'):
        """Predict turret position using the best available method"""
        if not self.is_calibrated:
            return None
        
        # Auto-select best method
        if method == 'auto':
            if self.use_homography and self.homography_matrix is not None:
                result = self.predict_position_homography(image_x, image_y)
                if result is not None:
                    return result
            
            if self.use_ransac and self.ransac_pan_model is not None:
                method = 'ransac'
            else:
                method = 'linear'
        
        # Transform the point
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
        """Interpolate position using calibrated points"""
        if len(self.calibration_data) < 4:
            return None
        
        # Extract calibration points
        points = np.array([(d[0], d[1]) for d in self.calibration_data])
        pan_values = np.array([d[2] for d in self.calibration_data])
        tilt_values = np.array([d[3] for d in self.calibration_data])
        
        # Interpolate
        try:
            pan = griddata(points, pan_values, (image_x, image_y), method='cubic')
            tilt = griddata(points, tilt_values, (image_x, image_y), method='cubic')
            
            if np.isnan(pan) or np.isnan(tilt):
                # Fallback to linear interpolation
                pan = griddata(points, pan_values, (image_x, image_y), method='linear')
                tilt = griddata(points, tilt_values, (image_x, image_y), method='linear')
            
            return pan, tilt
        except:
            return None
    
    def estimate_calibration_error(self, test_point):
        """Estimate the calibration error at a given point"""
        if not self.is_calibrated:
            return float('inf')
        
        # Predict position
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
        scaled_error = error * (1 + min_distance / 100)
        
        return scaled_error
    
    def get_essential_calibration_points(self):
        """Get the most important points for calibration"""
        if self.detected_corners is None:
            return []
        
        corners = self.detected_corners
        num_corners = len(corners)
        
        # Essential points: corners
        essential = []
        
        # Four corners
        if num_corners >= 4:
            essential.extend([0, self.board_size[0]-1, 
                            num_corners-self.board_size[0], num_corners-1])
        
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
        """Aim the turret at a target point in the camera view"""
        if not self.is_calibrated:
            return False
        
        # Predict position
        result = self.predict_position(target_x, target_y)
        if result is None:
            return False
        
        pan_angle, tilt_angle = result
        
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
        """Load a calibration model from a file"""
        try:
            with open(filename, 'rb') as f:
                calibration_data = pickle.load(f)
            
            self.pan_model = calibration_data['pan_model']
            self.tilt_model = calibration_data['tilt_model']
            self.poly_features = calibration_data['poly_features']
            self.calibration_data = calibration_data['calibration_data']
            
            # Load additional models if available
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
        """Print statistics about the calibration model"""
        if not self.is_calibrated:
            print("Model is not calibrated yet.")
            return
        
        print("\n=== Calibration Statistics ===")
        print(f"Number of calibration points: {len(self.calibration_data)}")
        print(f"Checkerboard size: {self.board_size}")
        
        # Extract data for error computation
        data = np.array(self.calibration_data)
        X = data[:, 0:2]  # Camera x,y coordinates
        Y_pan = data[:, 2]  # Pan/Yaw angles
        Y_tilt = data[:, 3]  # Tilt/Pitch angles
        
        # Transform to polynomial features
        X_poly = self.poly_features.transform(X)
        
        # Linear model statistics
        if self.pan_model is not None:
            print("\n--- Linear Model Statistics ---")
            pan_pred = self.pan_model.predict(X_poly)
            tilt_pred = self.tilt_model.predict(X_poly)
            
            # Compute errors
            pan_mse = np.mean((Y_pan - pan_pred) ** 2)
            tilt_mse = np.mean((Y_tilt - tilt_pred) ** 2)
            pan_abs_max = np.max(np.abs(Y_pan - pan_pred))
            tilt_abs_max = np.max(np.abs(Y_tilt - tilt_pred))
            
            # R² score
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
            
            # Compute errors
            ransac_pan_mse = np.mean((Y_pan - ransac_pan_pred) ** 2)
            ransac_tilt_mse = np.mean((Y_tilt - ransac_tilt_pred) ** 2)
            ransac_pan_abs_max = np.max(np.abs(Y_pan - ransac_pan_pred))
            ransac_tilt_abs_max = np.max(np.abs(Y_tilt - ransac_tilt_pred))
            
            # R² score
            ransac_pan_r2 = self.ransac_pan_model.score(X_poly, Y_pan)
            ransac_tilt_r2 = self.ransac_tilt_model.score(X_poly, Y_tilt)
            
            # Inlier statistics
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
                # Calculate condition number properly
                singular_values = np.linalg.svd(self.homography_matrix, compute_uv=False)
                condition_number = float(singular_values[0] / singular_values[-1])
                
                # Reprojection error using homography
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