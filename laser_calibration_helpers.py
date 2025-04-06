import numpy as np
import cv2
import time
import os
import json
from scipy.optimize import minimize

class LaserCalibrationHelpers:
    """Helper functions for laser turret calibration"""
    
    def __init__(self):
        # Calibration parameters
        self.laser_dot_min_area = 1  # Minimum area for laser dot detection
        self.laser_dot_max_area = 600  # Maximum area for laser dot detection
        self.reference_points = []  # List of (turret_yaw, turret_pitch, pixel_x, pixel_y)
        self.calibration = None  # Calibration result
        
    def detect_laser_dot(self, frame):
        """Detect a laser dot in the frame using multiple methods
        
        Args:
            frame: CV2 image frame
            
        Returns:
            tuple: (x, y) of detected laser dot center, or None if not found
        """
        if frame is None:
            return None
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multi-level detection for red laser with enhanced sensitivity
        # Level 1: Standard bright red laser detection
        lower_red1 = np.array([0, 80, 150])  # More sensitive saturation and value thresholds
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 150])
        upper_red2 = np.array([179, 255, 255])
        
        # Level 2: Detect fainter reds
        lower_red3 = np.array([0, 50, 100])  # Even more sensitive for faint dots
        upper_red3 = np.array([10, 255, 255])
        lower_red4 = np.array([160, 50, 100])
        upper_red4 = np.array([179, 255, 255])
        
        # Create masks for each level
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
        mask4 = cv2.inRange(hsv, lower_red4, upper_red4)
        
        # Combine masks with different weights
        # First combine each level
        mask_bright = cv2.bitwise_or(mask1, mask2)
        mask_faint = cv2.bitwise_or(mask3, mask4)
        
        # Apply more aggressive filtering to the faint mask to reduce noise
        kernel_small = np.ones((2, 2), np.uint8)
        mask_faint = cv2.morphologyEx(mask_faint, cv2.MORPH_OPEN, kernel_small)
        
        # Try both masks, prioritizing the bright one
        for current_mask in [mask_bright, mask_faint]:
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            processed_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel)
            processed_mask = cv2.dilate(processed_mask, kernel, iterations=1)
            
            # Try to find the dot using blob detection
            # This works better for small dots
            params = cv2.SimpleBlobDetector_Params()
            # Filter by color (bright)
            params.filterByColor = True
            params.blobColor = 255
            # Filter by size
            params.filterByArea = True
            params.minArea = max(1, self.laser_dot_min_area)
            params.maxArea = self.laser_dot_max_area
            # Filter by circularity (laser dots tend to be circular)
            params.filterByCircularity = True
            params.minCircularity = 0.5  # Allow for some distortion
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(processed_mask)
            
            if keypoints:
                # Return the position of the first (largest) blob
                x = int(keypoints[0].pt[0])
                y = int(keypoints[0].pt[1])
                return (x, y)
            
            # If blob detection fails, try moments
            M = cv2.moments(processed_mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Check area to validate detection
                area = cv2.countNonZero(processed_mask)
                if self.laser_dot_min_area <= area <= self.laser_dot_max_area:
                    return (cx, cy)
            
            # Fall back to contour method
            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the best contour using a more flexible criterion
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1:  # Skip empty contours
                    continue
                    
                # Calculate a score based on area and circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Weighted score favoring circular objects but allowing some flexibility
                score = area * (0.5 + 0.5 * circularity)
                
                if score > best_score and self.laser_dot_min_area <= area <= self.laser_dot_max_area:
                    best_score = score
                    best_contour = contour
            
            if best_contour is not None:
                # Get center of contour
                M = cv2.moments(best_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        
        # Last resort: Detect bright spots directly using peak finding on intensity
        # This can help with very small dots
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Find the maximum intensity pixel
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        # Check if it's significantly brighter than the surroundings
        if max_val > 200 and max_val > np.mean(blurred) * 1.5:
            # Verify it's likely a red dot by checking original color
            x, y = max_loc
            if frame[y, x, 2] > frame[y, x, 1] * 1.3 and frame[y, x, 2] > frame[y, x, 0] * 1.3:
                return max_loc
                
        return None
    
    def add_reference_point(self, yaw, pitch, pixel_x, pixel_y):
        """Add a calibration reference point
        
        Args:
            yaw: Turret yaw position (-1.0 to 1.0)
            pitch: Turret pitch position (-1.0 to 1.0)
            pixel_x: X coordinate of laser dot in image
            pixel_y: Y coordinate of laser dot in image
        """
        self.reference_points.append((yaw, pitch, pixel_x, pixel_y))
    
    def add_unique_reference_point(self, yaw, pitch, pixel_position, min_turret_distance=0.03, min_pixel_distance=10):
        """Add a reference point if it's not too close to existing points
        
        Args:
            yaw: Turret yaw position (-1.0 to 1.0)
            pitch: Turret pitch position (-1.0 to 1.0)
            pixel_position: Tuple of (x, y) pixel coordinates
            min_turret_distance: Minimum distance in turret space to consider unique
            min_pixel_distance: Minimum distance in pixel space to consider unique
            
        Returns:
            bool: True if point was added, False if too close to existing point
        """
        if pixel_position is None:
            return False
            
        pixel_x, pixel_y = pixel_position
        
        # Check if we already have a similar point
        for ref_yaw, ref_pitch, ref_x, ref_y in self.reference_points:
            # Skip if turret positions are too close
            if (abs(ref_yaw - yaw) < min_turret_distance and 
                abs(ref_pitch - pitch) < min_turret_distance):
                return False
                
            # Skip if pixel positions are too close
            if (abs(ref_x - pixel_x) < min_pixel_distance and 
                abs(ref_y - pixel_y) < min_pixel_distance):
                return False
        
        # Add to reference points
        self.reference_points.append((yaw, pitch, pixel_x, pixel_y))
        return True
    
    def clear_reference_points(self):
        """Clear all reference points"""
        self.reference_points = []
        self.calibration = None
    
    def compute_polynomial_calibration(self, image_width, image_height):
        """Compute polynomial calibration from reference points
        
        Args:
            image_width: Width of camera image
            image_height: Height of camera image
            
        Returns:
            bool: True if calibration was successful, False otherwise
        """
        if len(self.reference_points) < 4:
            print("Need at least 4 reference points for calibration")
            return False
        
        # Extract data
        turret_points = []
        pixel_points = []
        
        for yaw, pitch, px, py in self.reference_points:
            turret_points.append((yaw, pitch))
            pixel_points.append((px, py))
        
        # Convert to numpy arrays
        turret_points = np.array(turret_points, dtype=np.float32)
        pixel_points = np.array(pixel_points, dtype=np.float32)
        
        # Normalize pixel coordinates to [-1, 1] range
        norm_pixels = np.zeros_like(pixel_points, dtype=np.float32)
        norm_pixels[:, 0] = 2 * (pixel_points[:, 0] / image_width) - 1
        norm_pixels[:, 1] = 2 * (pixel_points[:, 1] / image_height) - 1
        
        # Create feature matrix for 2nd degree polynomial
        X = np.column_stack([
            np.ones_like(norm_pixels[:, 0]),  # Constant term
            norm_pixels[:, 0],                # x
            norm_pixels[:, 1],                # y
            norm_pixels[:, 0] * norm_pixels[:, 1],  # xy
            norm_pixels[:, 0]**2,             # x²
            norm_pixels[:, 1]**2              # y²
        ])
        
        # Use robust fitting if we have enough points
        if len(self.reference_points) >= 6:
            # Function to fit model with a subset of points
            def fit_model(X_subset, y_subset):
                return np.linalg.lstsq(X_subset, y_subset, rcond=None)[0]
            
            # Function to calculate error for all points
            def calculate_errors(model, X_data, y_data):
                y_pred = np.dot(X_data, model)
                return np.abs(y_pred - y_data)
            
            # RANSAC-like algorithm for yaw and pitch
            best_yaw_model = None
            min_yaw_error = float('inf')
            best_pitch_model = None
            min_pitch_error = float('inf')
            
            # Number of iterations for robust fitting
            n_iterations = 15
            # Threshold for inlier classification
            inlier_threshold = 0.05
            
            for _ in range(n_iterations):
                # Randomly select a subset of points (80%)
                n_samples = len(X)
                n_subset = int(0.8 * n_samples)
                subset_indices = np.random.choice(n_samples, n_subset, replace=False)
                
                X_subset = X[subset_indices]
                
                # Fit yaw model
                yaw_subset = turret_points[subset_indices, 0]
                yaw_model = fit_model(X_subset, yaw_subset)
                
                # Calculate error for all points
                yaw_errors = calculate_errors(yaw_model, X, turret_points[:, 0])
                
                # Calculate overall error metric
                yaw_error = np.mean(yaw_errors)
                
                # Update best model if this one is better
                if yaw_error < min_yaw_error:
                    min_yaw_error = yaw_error
                    best_yaw_model = yaw_model
                
                # Fit pitch model
                pitch_subset = turret_points[subset_indices, 1]
                pitch_model = fit_model(X_subset, pitch_subset)
                
                # Calculate error for all points
                pitch_errors = calculate_errors(pitch_model, X, turret_points[:, 1])
                
                # Calculate overall error metric
                pitch_error = np.mean(pitch_errors)
                
                # Update best model if this one is better
                if pitch_error < min_pitch_error:
                    min_pitch_error = pitch_error
                    best_pitch_model = pitch_model
            
            # Identify inliers and outliers
            yaw_errors = calculate_errors(best_yaw_model, X, turret_points[:, 0])
            yaw_inlier_mask = yaw_errors < inlier_threshold
            
            pitch_errors = calculate_errors(best_pitch_model, X, turret_points[:, 1])
            pitch_inlier_mask = pitch_errors < inlier_threshold
            
            # Refit with inliers if we have enough
            if yaw_inlier_mask.sum() >= 4:
                yaw_model = fit_model(X[yaw_inlier_mask], turret_points[yaw_inlier_mask, 0])
            else:
                yaw_model = best_yaw_model
            
            if pitch_inlier_mask.sum() >= 4:
                pitch_model = fit_model(X[pitch_inlier_mask], turret_points[pitch_inlier_mask, 1])
            else:
                pitch_model = best_pitch_model
        else:
            # Fit models for yaw and pitch separately using standard least squares
            yaw_model, _, _, _ = np.linalg.lstsq(X, turret_points[:, 0], rcond=None)
            pitch_model, _, _, _ = np.linalg.lstsq(X, turret_points[:, 1], rcond=None)
        
        # Store the transformation model
        self.calibration = {
            'model_type': 'polynomial',
            'yaw_model': yaw_model,
            'pitch_model': pitch_model,
            'image_width': image_width,
            'image_height': image_height
        }
        
        # Calculate final error
        yaw_pred = np.dot(X, yaw_model)
        pitch_pred = np.dot(X, pitch_model)
        
        yaw_rmse = np.sqrt(np.mean((yaw_pred - turret_points[:, 0])**2))
        pitch_rmse = np.sqrt(np.mean((pitch_pred - turret_points[:, 1])**2))
        
        print(f"Polynomial calibration computed:")
        print(f"Yaw RMSE: {yaw_rmse:.4f}")
        print(f"Pitch RMSE: {pitch_rmse:.4f}")
        
        return True
    
    def compute_pinhole_calibration(self, image_width, image_height):
        """Compute pinhole camera calibration from reference points
        
        Args:
            image_width: Width of camera image
            image_height: Height of camera image
            
        Returns:
            bool: True if calibration was successful, False otherwise
        """
        if len(self.reference_points) < 8:
            print("Need at least 8 reference points for pinhole camera calibration")
            return False
        
        # Extract data
        turret_points = []
        pixel_points = []
        
        for yaw, pitch, px, py in self.reference_points:
            turret_points.append((yaw, pitch))
            pixel_points.append((px, py))
        
        # Convert to numpy arrays
        turret_points = np.array(turret_points, dtype=np.float32)
        pixel_points = np.array(pixel_points, dtype=np.float32)
        
        # Normalize pixel coordinates
        norm_pixels = np.zeros_like(pixel_points, dtype=np.float32)
        norm_pixels[:, 0] = 2 * (pixel_points[:, 0] / image_width) - 1
        norm_pixels[:, 1] = 2 * (pixel_points[:, 1] / image_height) - 1
        
        # Initialize optimization with reasonable values
        # [cx, cy, fx, fy, k1, k2]
        initial_params = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        
        def projection_error(params, norm_pixels, turret_points):
            cx, cy, fx, fy, k1, k2 = params
            
            # Convert turret angles to 3D rays
            rays = np.zeros((len(turret_points), 3), dtype=np.float32)
            for i, (yaw, pitch) in enumerate(turret_points):
                # Convert from turret coordinates to 3D Cartesian
                angle_yaw = yaw * np.pi/2  # Convert from [-1,1] to [-π/2,π/2]
                angle_pitch = pitch * np.pi/2
                
                # Calculate 3D vector from angles
                x = np.tan(angle_yaw)
                y = np.tan(angle_pitch) * np.cos(angle_yaw)
                z = 1.0  # Fixed distance in z direction
                
                # Normalize to unit vector
                magnitude = np.sqrt(x*x + y*y + z*z)
                rays[i] = [x/magnitude, y/magnitude, z/magnitude]
            
            # Project 3D rays to 2D using camera model
            projected = np.zeros((len(rays), 2), dtype=np.float32)
            for i, (x, y, z) in enumerate(rays):
                if z <= 0:  # Behind the camera
                    projected[i] = [0, 0]
                    continue
                    
                # Perspective division
                xp = x / z
                yp = y / z
                
                # Apply radial distortion
                r2 = xp*xp + yp*yp
                distortion = 1 + k1*r2 + k2*r2*r2
                
                # Apply camera intrinsics
                u = cx + fx * xp * distortion
                v = cy + fy * yp * distortion
                
                projected[i] = [u, v]
            
            # Calculate error between projected points and normalized pixels
            error = np.sum((projected - norm_pixels)**2)
            return error
        
        # Use scipy's optimization to find the best parameters
        try:
            result = minimize(
                projection_error,
                initial_params,
                args=(norm_pixels, turret_points),
                method='Powell',
                options={'maxiter': 500}
            )
            
            if not result.success:
                print("Pinhole calibration optimization failed")
                return False
            
            # Get optimized parameters
            cx, cy, fx, fy, k1, k2 = result.x
            
            # Store the camera model
            self.calibration = {
                'model_type': 'pinhole',
                'cx': cx,
                'cy': cy,
                'fx': fx,
                'fy': fy,
                'k1': k1,
                'k2': k2,
                'image_width': image_width,
                'image_height': image_height
            }
            
            # Calculate error for evaluation
            projection_err = projection_error(result.x, norm_pixels, turret_points)
            avg_err = np.sqrt(projection_err / len(norm_pixels))
            
            print(f"Pinhole camera calibration complete")
            print(f"Average error: {avg_err:.4f}")
            print(f"Camera parameters: cx={cx:.2f}, cy={cy:.2f}, fx={fx:.2f}, fy={fy:.2f}")
            print(f"Distortion: k1={k1:.4f}, k2={k2:.4f}")
            
            return True
        except Exception as e:
            print(f"Error in pinhole calibration: {str(e)}")
            return False
    
    def pixel_to_turret(self, pixel_x, pixel_y):
        """Convert pixel coordinates to turret coordinates using the current calibration
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            
        Returns:
            tuple: (yaw, pitch) turret coordinates or None if no calibration
        """
        if self.calibration is None:
            return None
        
        if self.calibration.get('model_type') == 'pinhole':
            return self.pixel_to_turret_pinhole(pixel_x, pixel_y)
        else:
            return self.pixel_to_turret_polynomial(pixel_x, pixel_y)
    
    def pixel_to_turret_polynomial(self, pixel_x, pixel_y):
        """Convert pixel coordinates to turret coordinates using polynomial model
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            
        Returns:
            tuple: (yaw, pitch) turret coordinates or None if no calibration
        """
        if self.calibration is None or self.calibration.get('model_type') != 'polynomial':
            return None
        
        # Normalize pixel coordinates
        w = self.calibration['image_width']
        h = self.calibration['image_height']
        
        norm_x = 2 * (pixel_x / w) - 1
        norm_y = 2 * (pixel_y / h) - 1
        
        # Construct feature vector
        X = np.array([
            1.0,            # Constant term
            norm_x,         # x
            norm_y,         # y
            norm_x * norm_y,  # xy
            norm_x**2,      # x²
            norm_y**2       # y²
        ])
        
        # Predict turret coordinates
        yaw = np.dot(X, self.calibration['yaw_model'])
        pitch = np.dot(X, self.calibration['pitch_model'])
        
        # Clamp to valid range
        yaw = max(-1.0, min(1.0, yaw))
        pitch = max(-1.0, min(1.0, pitch))
        
        return yaw, pitch
    
    def pixel_to_turret_pinhole(self, pixel_x, pixel_y):
        """Convert pixel coordinates to turret coordinates using pinhole model
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            
        Returns:
            tuple: (yaw, pitch) turret coordinates or None if no calibration
        """
        if self.calibration is None or self.calibration.get('model_type') != 'pinhole':
            return None
        
        # Normalize pixel coordinates
        w = self.calibration['image_width']
        h = self.calibration['image_height']
        
        norm_x = 2 * (pixel_x / w) - 1
        norm_y = 2 * (pixel_y / h) - 1
        
        # Get camera parameters
        cx = self.calibration['cx']
        cy = self.calibration['cy']
        fx = self.calibration['fx']
        fy = self.calibration['fy']
        k1 = self.calibration['k1']
        k2 = self.calibration['k2']
        
        # Pixel to normalized camera coordinates
        x = (norm_x - cx) / fx
        y = (norm_y - cy) / fy
        
        # Apply inverse radial distortion (simplified)
        r2 = x*x + y*y
        distortion = 1 + k1*r2 + k2*r2*r2
        
        # Correct for distortion (approximation)
        x_corrected = x / distortion
        y_corrected = y / distortion
        
        # Convert to 3D ray
        ray = np.array([x_corrected, y_corrected, 1.0])
        ray = ray / np.linalg.norm(ray)  # Normalize to unit vector
        
        # Convert ray to turret angles - Optimized for 180° turret range
        yaw = np.arctan2(ray[0], ray[2]) * 2/np.pi  # Maps [-π/2,π/2] to [-1,1]
        pitch = np.arctan2(ray[1], np.sqrt(ray[0]**2 + ray[2]**2)) * 2/np.pi
        
        # Clamp to valid range
        yaw = max(-1.0, min(1.0, yaw))
        pitch = max(-1.0, min(1.0, pitch))
        
        return yaw, pitch
    
    def save_calibration(self, filename='calibration_data/laser_calibration.json'):
        """Save calibration data to file
        
        Args:
            filename: Path to save calibration data
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        if not self.reference_points or self.calibration is None:
            print("No calibration data to save")
            return False
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save calibration data
            calibration_data = {
                'reference_points': self.reference_points,
                'model_type': self.calibration.get('model_type', 'polynomial')
            }
            
            # Add model parameters based on type
            if self.calibration.get('model_type') == 'pinhole':
                calibration_data.update({
                    'cx': float(self.calibration['cx']),
                    'cy': float(self.calibration['cy']),
                    'fx': float(self.calibration['fx']),
                    'fy': float(self.calibration['fy']),
                    'k1': float(self.calibration['k1']),
                    'k2': float(self.calibration['k2']),
                    'image_width': int(self.calibration['image_width']),
                    'image_height': int(self.calibration['image_height'])
                })
            else:  # polynomial model
                calibration_data.update({
                    'yaw_model': self.calibration['yaw_model'].tolist(),
                    'pitch_model': self.calibration['pitch_model'].tolist(),
                    'image_width': int(self.calibration['image_width']),
                    'image_height': int(self.calibration['image_height'])
                })
            
            with open(filename, 'w') as f:
                json.dump(calibration_data, f)
            
            print(f"Calibration data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving calibration data: {str(e)}")
            return False
    
    def load_calibration(self, filename='calibration_data/laser_calibration.json'):
        """Load calibration data from file
        
        Args:
            filename: Path to load calibration data from
            
        Returns:
            bool: True if load was successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.reference_points = data['reference_points']
            
            # Load model data
            model_type = data.get('model_type', 'polynomial')
            
            if model_type == 'pinhole':
                self.calibration = {
                    'model_type': 'pinhole',
                    'cx': data['cx'],
                    'cy': data['cy'],
                    'fx': data['fx'],
                    'fy': data['fy'],
                    'k1': data['k1'],
                    'k2': data['k2'],
                    'image_width': data['image_width'],
                    'image_height': data['image_height']
                }
                print(f"Loaded pinhole camera model with {len(self.reference_points)} reference points")
            else:  # polynomial model
                self.calibration = {
                    'model_type': 'polynomial',
                    'yaw_model': np.array(data['yaw_model']),
                    'pitch_model': np.array(data['pitch_model']),
                    'image_width': data['image_width'],
                    'image_height': data['image_height']
                }
                print(f"Loaded polynomial model with {len(self.reference_points)} reference points")
            
            return True
        except Exception as e:
            print(f"Error loading calibration data: {str(e)}")
            return False
    
    def evaluate_calibration(self):
        """Evaluate current calibration quality using reference points
        
        Returns:
            tuple: (average_error, max_error, worst_point) or None if no calibration
        """
        if not self.reference_points or not self.calibration:
            return None
            
        total_error = 0
        max_error = 0
        worst_point = None
        
        for yaw, pitch, px, py in self.reference_points:
            # Use appropriate conversion based on model type
            if self.calibration.get('model_type') == 'pinhole':
                predicted_yaw, predicted_pitch = self.pixel_to_turret_pinhole(px, py)
            else:
                predicted_yaw, predicted_pitch = self.pixel_to_turret_polynomial(px, py)
            
            # Calculate error
            yaw_error = abs(predicted_yaw - yaw)
            pitch_error = abs(predicted_pitch - pitch)
            point_error = np.sqrt(yaw_error**2 + pitch_error**2)
            
            total_error += point_error
            
            if point_error > max_error:
                max_error = point_error
                worst_point = (yaw, pitch, predicted_yaw, predicted_pitch)
        
        avg_error = total_error / len(self.reference_points)
        
        print(f"Calibration evaluation:")
        print(f"Average error: {avg_error:.4f}")
        print(f"Maximum error: {max_error:.4f}")
        
        if worst_point:
            print(f"Worst point: Actual=({worst_point[0]:.2f}, {worst_point[1]:.2f}), "
                 f"Predicted=({worst_point[2]:.2f}, {worst_point[3]:.2f})")
        
        quality = "Excellent" if avg_error < 0.05 else (
                  "Good" if avg_error < 0.1 else (
                  "Fair" if avg_error < 0.2 else "Poor"))
                  
        print(f"Calibration quality: {quality}")
        
        return avg_error, max_error, worst_point
