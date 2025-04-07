#!/usr/bin/env python3
import cv2
import numpy as np
import time
import threading
import pickle
import matplotlib.pyplot as plt
import random
import os
import collections

# --- Dependencies ---
# Essential: opencv-python, numpy, matplotlib, scipy
# Required for GPR: scikit-learn
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process import GaussianProcessRegressor
    # Common kernels: Matern, RBF, WhiteKernel, ConstantKernel
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from sklearn.model_selection import KFold
    from sklearn.linear_model import RANSACRegressor, Ridge # RANSAC for outlier rejection
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures # For RANSAC base estimator
    SKLEARN_AVAILABLE = True
except ImportError:
    print("ERROR: scikit-learn is required for this GPR-based calibrator.")
    print("Please install it: pip install scikit-learn")
    SKLEARN_AVAILABLE = False
    # Exit or raise an error if sklearn is absolutely essential
    # raise ImportError("scikit-learn not found, which is essential for GPR.")

from scipy.spatial import distance # For optimization
from scipy.stats import median_abs_deviation # For MAD outlier removal fallback
from concurrent.futures import ThreadPoolExecutor

# --- Default Configuration for GPR Focus ---
DEFAULT_CONFIG = {
    # Detection
    "hsv_lower1": (0, 120, 100),
    "hsv_upper1": (10, 255, 255),
    "hsv_lower2": (165, 120, 100),
    "hsv_upper2": (180, 255, 255),
    "brightness_thresh": 230,
    "min_contour_area": 4,
    "max_contour_area": 150,
    "subpixel_roi_size": 15,
    "subpixel_term_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),

    # Data Collection
    "laser_stabilization_time": 0.15,
    "turret_settle_time_base": 0.25,
    "turret_settle_time_factor": 1.0,

    # Kalman Filter (Temporal Integration)
    "kalman_process_noise": 1e-4,
    "kalman_measurement_noise": 5e-2,
    "kalman_error_cov_post": 1.0,
    "temporal_buffer_size": 7,
    "temporal_median_filter": True,

    # Outlier Rejection
    "use_ransac": True, # Recommended if sklearn available
    "ransac_base_poly_order": 2, # Low order poly for RANSAC base estimator
    "ransac_min_samples_ratio": 0.2, # Fraction of points for RANSAC min_samples
    "ransac_residual_threshold_factor": 3.0, # Factor times MAD for threshold

    # GPR Model
    # Matern(nu=1.5 or 2.5) often good for physical systems. ConstantKernel allows mean offset.
    "gpr_kernel": ConstantKernel(1.0) * Matern(length_scale=0.5, nu=1.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)),
    "gpr_restarts": 7, # More restarts for better hyperparameter optimization
    "gpr_alpha": 1e-5, # Regularization added to the diagonal of the kernel matrix during fitting

    # General
    "calibration_range_yaw": (-0.75, -0.40),
    "calibration_range_pitch": (0.05, 0.75),
}


class GprTurretCalibrator:
    def __init__(self, turret_client, camera_client, config=None):
        """
        Initializes the GPR-focused turret calibrator.

        Args:
            turret_client: An object to control the turret (needs rotate, laser_on, laser_off, connected attributes).
            camera_client: An object providing camera frames (needs frame_callback, streaming attributes).
            config (dict, optional): Overrides for the DEFAULT_CONFIG.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for GprTurretCalibrator.")

        self.turret = turret_client
        self.camera = camera_client

        # --- Configuration ---
        # Start with default configuration
        self.config = DEFAULT_CONFIG.copy()
        
        # Update with custom config if provided
        if config:
            # Add feature flags to accepted keys
            accepted_keys = list(self.config.keys()) + [
                'use_subpixel',
                'use_temporal',
                'use_parallel',
                'num_points'
            ]
            
            # Simple validation: check if keys in custom config exist in accepted keys
            for key in config:
                if key not in accepted_keys:
                    print(f"Warning: Custom config key '{key}' not found in defaults.")
            
            # Update config with custom values
            self.config.update(config)

        # Calibration data storage
        self.turret_positions_raw = [] # Before outlier removal
        self.image_points_raw = []
        self.turret_positions = []  # Final (yaw, pitch) pairs used in model
        self.image_points = []     # Final (x, y) pixel coordinates used in model

        # Model storage (Specific to GPR)
        self.gp_model_x = None
        self.gp_model_y = None
        self.scaler_turret = StandardScaler() # Input scaler
        self.scaler_image = StandardScaler()  # Output scaler

        # State
        self.calibration_complete = False
        self.model_type = 'gpr' # Fixed for this class
        self.last_detection_point = None
        self.repeated_detection_count = 0

        # Feature Flags
        self.use_subpixel = self.config.get('use_subpixel', True)
        self.use_temporal = self.config.get('use_temporal', True)
        self.use_parallel = self.config.get('use_parallel', True)  # Set based on system capability

        # Advanced detection parameters
        self.kalman_filter = None
        self.kalman_initialized = False
        self.detection_history = collections.deque(maxlen=10)

        # Thread synchronization
        self.current_frame = None
        self.recent_frames = collections.deque(maxlen=self.config['temporal_buffer_size'])
        self.frame_ready = threading.Event()
        self.lock = threading.Lock()
        max_workers = max(1, os.cpu_count() - 1 if os.cpu_count() else 2)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Calibration range
        self.min_yaw, self.max_yaw = self.config['calibration_range_yaw']
        self.min_pitch, self.max_pitch = self.config['calibration_range_pitch']

        # Performance tracking
        self.timing_stats = collections.defaultdict(list)

        # Setup camera frame callback
        self.camera.frame_callback = self._frame_callback

        # Initialize Kalman filter
        self._init_kalman_filter()

        print(f"GprTurretCalibrator initialized.")
        print(f"  Using {max_workers} worker threads.")
        print(f"  GPR Kernel: {self.config['gpr_kernel']}")
        print(f"  Feature flags: subpixel={self.use_subpixel}, temporal={self.use_temporal}, parallel={self.use_parallel}")
        print(f"  Calibration range: Yaw=[{self.min_yaw:.2f}, {self.max_yaw:.2f}], Pitch=[{self.min_pitch:.2f}, {self.max_pitch:.2f}]")

    # ============================================
    # Kalman Filter & Detection (Mostly Unchanged)
    # ============================================
    def _init_kalman_filter(self):
        """Initialize Kalman filter with configurable parameters"""
        self.kalman_filter = cv2.KalmanFilter(4, 2)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        p_noise = self.config['kalman_process_noise']
        self.kalman_filter.processNoiseCov = np.diag([p_noise, p_noise, p_noise * 10, p_noise * 10]).astype(np.float32) # Higher noise for velocity
        m_noise = self.config['kalman_measurement_noise']
        self.kalman_filter.measurementNoiseCov = np.diag([m_noise, m_noise]).astype(np.float32)
        e_cov = self.config['kalman_error_cov_post']
        self.kalman_filter.errorCovPost = np.eye(4, dtype=np.float32) * e_cov
        self.kalman_initialized = False

    def _frame_callback(self, frame):
        """Store the latest frame and signal it's ready"""
        with self.lock:
            frame_copy = frame.copy()
            self.current_frame = frame_copy
            self.recent_frames.append(frame_copy)
            self.frame_ready.set()

    def detect_laser_point(self, frame):
        """Enhanced laser point detection with better filtering and debug capability"""
        start_time = time.time()
        
        # Enable debug mode to see what's happening
        debug = False  # Set to True to save debug images
        debug_dir = "debug_frames"
        if debug and not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # Create a debug frame copy if needed
        if debug:
            debug_frame = frame.copy()
            timestamp = int(time.time() * 1000)
            cv2.imwrite(f"{debug_dir}/original_{timestamp}.jpg", frame)
        
        # Try HSV detection first
        point = self._detect_laser_hsv(frame, debug_mode=debug, debug_dir=debug_dir)
        
        # If HSV detection fails, try brightness detection
        if point is None:
            point = self._detect_laser_brightness(frame, debug_mode=debug, debug_dir=debug_dir)
        
        # Apply subpixel refinement if enabled and a point was found
        if point is not None and self.use_subpixel:
            refined_point = self._refine_to_subpixel(frame, point)
            if refined_point is not None:
                # Calculate distance between original and refined point
                dist = np.linalg.norm(np.array(point) - np.array(refined_point))
                # Only use refinement if it didn't move too far
                if dist < self.config['subpixel_roi_size']:
                    point = refined_point
        
        # Add debug visualization
        if debug and point is not None:
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(debug_frame, (x, y), 10, (0, 255, 0), 2)
            cv2.line(debug_frame, (x-15, y), (x+15, y), (0, 255, 0), 1)
            cv2.line(debug_frame, (x, y-15), (x, y+15), (0, 255, 0), 1)
            cv2.putText(debug_frame, f"({x}, {y})", (x+20, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(f"{debug_dir}/detection_{timestamp}.jpg", debug_frame)
        
        self.timing_stats['detect_laser_point'].append(time.time() - start_time)
        return point

    def _detect_laser_hsv(self, frame, debug_mode=False, debug_dir="debug_frames"):
        """Improved HSV-based laser detection with better filtering"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Use two ranges for red (wraps around hue value 0/180)
        lower1 = np.array(self.config['hsv_lower1'])
        upper1 = np.array(self.config['hsv_upper1'])
        lower2 = np.array(self.config['hsv_lower2'])
        upper2 = np.array(self.config['hsv_upper2'])
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Save debug images if requested
        if debug_mode:
            timestamp = int(time.time() * 1000)
            cv2.imwrite(f"{debug_dir}/hsv_mask_{timestamp}.jpg", mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by area
        min_area = self.config['min_contour_area']
        max_area = self.config['max_contour_area']
        valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
        
        if debug_mode:
            contour_image = frame.copy()
            cv2.drawContours(contour_image, valid_contours, -1, (0, 255, 0), 2)
            timestamp = int(time.time() * 1000)
            cv2.imwrite(f"{debug_dir}/hsv_contours_{timestamp}.jpg", contour_image)
        
        if not valid_contours:
            return None
        
        # Find the brightest contour
        best_contour = None
        best_brightness = -1
        
        for contour in valid_contours:
            # Create mask for this contour
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], 0, 255, -1)
            
            # Calculate mean brightness within contour (from V channel)
            mean_val = np.mean(hsv[contour_mask == 255, 2])
            
            # Area normalized brightness (favor small bright spots over large dim ones)
            area = cv2.contourArea(contour)
            normalized_brightness = mean_val * (1.0 + 10.0/max(area, 1.0))
            
            if normalized_brightness > best_brightness:
                best_brightness = normalized_brightness
                best_contour = contour
        
        if best_contour is None:
            return None
        
        # Calculate centroid using moments
        M = cv2.moments(best_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Additional check: ensure the contour is not at the exact same coordinates as previous detections
            # This helps avoid getting stuck on a "phantom" point
            if hasattr(self, 'last_detection_point') and self.last_detection_point is not None:
                last_x, last_y = self.last_detection_point
                if abs(cx - last_x) < 3 and abs(cy - last_y) < 3:
                    # Too similar to last point, check if this is a pattern
                    if hasattr(self, 'repeated_detection_count'):
                        self.repeated_detection_count += 1
                        if self.repeated_detection_count > 5:
                            # If we've detected the same point multiple times, try a different method
                            print("Warning: Same point detected multiple times, trying brightness detection")
                            return self._detect_laser_brightness(frame, debug_mode, debug_dir)
                    else:
                        self.repeated_detection_count = 1
                else:
                    self.repeated_detection_count = 0
            
            self.last_detection_point = (cx, cy)
            return (cx, cy)
        
        return None

    def _detect_laser_brightness(self, frame, debug_mode=False, debug_dir="debug_frames"):
        """Improved brightness-based laser detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to better handle varying lighting
        thresh_val = self.config['brightness_thresh']
        
        # First try simple thresholding
        _, bright_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        # If simple thresholding fails to find anything, try adaptive thresholding
        if cv2.countNonZero(bright_mask) == 0:
            # Use a gentler threshold
            _, bright_mask = cv2.threshold(gray, thresh_val - 30, 255, cv2.THRESH_BINARY)
            
            # If that still fails, try adaptive thresholding
            if cv2.countNonZero(bright_mask) == 0:
                bright_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                                cv2.THRESH_BINARY, 15, -5)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Save debug images if requested
        if debug_mode:
            timestamp = int(time.time() * 1000)
            cv2.imwrite(f"{debug_dir}/bright_mask_{timestamp}.jpg", bright_mask)
        
        # Find contours
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug_mode and contours:
            contour_image = frame.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            timestamp = int(time.time() * 1000)
            cv2.imwrite(f"{debug_dir}/bright_contours_{timestamp}.jpg", contour_image)
        
        if not contours:
            return None
        
        # First try filtering by area
        min_area = self.config['min_contour_area']
        max_area = self.config['max_contour_area']
        valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
        
        # If we don't have valid contours, try alternatives
        if not valid_contours:
            # Perhaps our area thresholds are too strict, try with looser constraints
            valid_contours = [c for c in contours if cv2.contourArea(c) > 0]
            
            # Sort by area, take top 5
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:5]
        
        if not valid_contours:
            return None
        
        # Find the brightest spot
        brightest_contour = None
        max_brightness = 0
        max_normalized_brightness = 0
        
        for contour in valid_contours:
            # Create mask for this contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Calculate mean brightness within contour
            mean_val = cv2.mean(gray, mask=mask)[0]
            area = cv2.contourArea(contour)
            
            # Calculate brightness normalized by area (favor small bright spots)
            area_factor = 1.0 + 10.0/max(area, 1.0)
            normalized_brightness = mean_val * area_factor
            
            if normalized_brightness > max_normalized_brightness:
                max_normalized_brightness = normalized_brightness
                max_brightness = mean_val
                brightest_contour = contour
        
        if brightest_contour is None:
            return None
        
        # Calculate centroid
        M = cv2.moments(brightest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Similar check as in HSV detection to avoid getting stuck
            if hasattr(self, 'last_detection_point') and self.last_detection_point is not None:
                last_x, last_y = self.last_detection_point
                if abs(cx - last_x) < 3 and abs(cy - last_y) < 3:
                    if hasattr(self, 'repeated_detection_count'):
                        self.repeated_detection_count += 1
                        if self.repeated_detection_count > 5:
                            print("Warning: Detecting the same point repeatedly, may need to adjust settings")
                            # Try with different thresholds
                            if hasattr(self, 'dynamic_brightness_thresh'):
                                self.dynamic_brightness_thresh += 10
                            else:
                                self.dynamic_brightness_thresh = self.config['brightness_thresh'] + 10
                            
                            # Return None to force fresh detection next time
                            return None
                    else:
                        self.repeated_detection_count = 1
                else:
                    self.repeated_detection_count = 0
            
            self.last_detection_point = (cx, cy)
            return (cx, cy)
        
        return None

    def _refine_to_subpixel(self, frame, point):
        """Refine point detection to subpixel accuracy with proper error handling"""
        if point is None: 
            return None
            
        # Convert to integers for indexing
        x_int, y_int = int(round(point[0])), int(round(point[1]))
        height, width = frame.shape[:2]
        
        # Get ROI size from config
        roi_half = self.config['subpixel_roi_size'] // 2
        
        # Ensure we're not too close to the edges
        if x_int < roi_half*2+5 or y_int < roi_half*2+5 or x_int >= width-roi_half*2-5 or y_int >= height-roi_half*2-5:
            return point  # Return original if too close to edge
        
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Extract ROI
            roi = gray[y_int-roi_half:y_int+roi_half+1, x_int-roi_half:x_int+roi_half+1]
            if roi.size == 0 or roi.shape[0] < roi_half*2 or roi.shape[1] < roi_half*2:
                return point
            
            # Find the brightest point in the ROI
            max_val = np.max(roi)
            max_loc = np.where(roi == max_val)
            if len(max_loc[0]) == 0:
                return point
            
            # Use the first brightest point (if multiple)
            cy_roi = float(max_loc[0][0])
            cx_roi = float(max_loc[1][0])
            
            # Format for cornerSubPix - must be shape (1,1,2)
            corners = np.array([[[cx_roi, cy_roi]]], dtype=np.float32)
            
            # Apply cornerSubPix
            term_criteria = self.config['subpixel_term_criteria']
            refined_corners = cv2.cornerSubPix(roi, corners, (roi_half//2, roi_half//2), 
                                            (-1, -1), term_criteria)
            
            # Extract the refined coordinates from the array
            if refined_corners is not None and refined_corners.shape[0] > 0:
                # Access the first corner's x,y coordinates properly
                cx_refined = refined_corners[0, 0, 0]  # First element x
                cy_refined = refined_corners[0, 0, 1]  # First element y
                
                # Convert back to full image coordinates
                refined_x = x_int - roi_half + cx_refined
                refined_y = y_int - roi_half + cy_refined
                
                # Only accept refinement if it didn't jump too far
                max_jump = roi_half * 0.8
                if (abs(refined_x - x_int) <= max_jump and 
                    abs(refined_y - y_int) <= max_jump):
                    return (refined_x, refined_y)
            
            # Fallback to moment-based refinement
            thresh_val = max(50, self.config['brightness_thresh'] - 30)
            _, roi_thresh = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
            M = cv2.moments(roi_thresh)
            if M["m00"] > 0:
                cx_roi = M["m10"] / M["m00"]
                cy_roi = M["m01"] / M["m00"]
                refined_x = x_int - roi_half + cx_roi
                refined_y = y_int - roi_half + cy_roi
                
                # Check if the refinement is reasonable
                if (abs(refined_x - x_int) <= max_jump and 
                    abs(refined_y - y_int) <= max_jump):
                    return (refined_x, refined_y)
        
        except cv2.error as e:
            print(f"Warning: CV2 error in subpixel refinement: {e}")
        except Exception as e:
            print(f"Warning: Unexpected error in subpixel refinement: {e}")
        
        # Return original point if all refinement methods fail
        return point

    def _detect_with_temporal_integration(self):
        """Use recent frames and Kalman filter for robust, smoothed detection"""
        # Reuse temporal integration logic from the previous version
        with self.lock:
            frames = list(self.recent_frames)
        if not frames:
            self.kalman_initialized = False
            return None

        # Detect points (parallel or sequential)
        if self.use_parallel and len(frames) > 1:
            futures = [self.thread_pool.submit(self.detect_laser_point, f) for f in frames]
            raw_points = [future.result() for future in futures if future.result() is not None]
        else:
            raw_points = [p for f in frames if (p := self.detect_laser_point(f)) is not None]

        if not raw_points:
            self.kalman_initialized = False
            return None

        points_array = np.array(raw_points, dtype=np.float32)

        # Optional Median Filtering before Kalman
        if self.config['temporal_median_filter'] and len(points_array) >= 3:
            median_x = np.median(points_array[:, 0])
            median_y = np.median(points_array[:, 1])
            mad_x = median_abs_deviation(points_array[:, 0], scale='normal')
            mad_y = median_abs_deviation(points_array[:, 1], scale='normal')
            thresh = 3.0
            mask = (np.abs(points_array[:, 0] - median_x) <= thresh * (mad_x + 1e-6)) & \
                   (np.abs(points_array[:, 1] - median_y) <= thresh * (mad_y + 1e-6))
            points_to_use = points_array[mask]
            if len(points_to_use) == 0:
                points_to_use = np.array([[median_x, median_y]], dtype=np.float32) # Fallback to median
        else:
            points_to_use = points_array

        # Kalman Update
        measurement = np.mean(points_to_use, axis=0).reshape((2, 1)).astype(np.float32)
        if not self.kalman_initialized:
            self.kalman_filter.statePost = np.array([[measurement[0, 0]], [measurement[1, 0]], [0], [0]], dtype=np.float32)
            self.kalman_initialized = True
            predicted_state = self.kalman_filter.statePost
        else:
            predicted_state = self.kalman_filter.predict()
            try:
                estimated_state = self.kalman_filter.correct(measurement)
                predicted_state = estimated_state
            except cv2.error as e:
                 print(f"Warning: Kalman correction error: {e}. Using prediction.")
                 self.kalman_initialized = False # Reset on error

        refined_point = (float(predicted_state[0, 0]), float(predicted_state[1, 0]))
        self.detection_history.append(refined_point)
        return refined_point

    # ============================================
    # Data Collection (Mostly Unchanged)
    # ============================================
    def _calculate_optimal_wait_time(self, current_pos, target_pos):
        dist = distance.euclidean(current_pos, target_pos)  # Renamed from 'distance' to 'dist'
        base_wait = self.config['turret_settle_time_base']
        distance_factor = self.config['turret_settle_time_factor']
        full_range_dist = distance.euclidean(
            (self.min_yaw, self.min_pitch), (self.max_yaw, self.max_pitch)
        )
        wait_time = base_wait + (dist / (full_range_dist + 1e-6)) * distance_factor
        min_wait = base_wait # Wait at least the base settle time
        # Add laser stabilization only *after* moving, handled in _collect_calibration_points
        return min(max(wait_time, min_wait), 1.5) # Clamp between base and 1.5s

    def _optimize_point_collection_order(self, start_pos, points):
        """Optimize point order using nearest neighbor heuristic"""
        # Reuse order optimization logic
        if not points: return []
        remaining_points = points.copy()
        ordered_points = []
        current_pos = start_pos
        while remaining_points:
            distances = [distance.euclidean(p, current_pos) for p in remaining_points]
            nearest_idx = np.argmin(distances)
            nearest_point = remaining_points.pop(nearest_idx)
            ordered_points.append(nearest_point)
            current_pos = nearest_point
        return ordered_points

    def _collect_calibration_points(self, positions_to_visit, display=True):
        """Collects calibration data, using temporal integration for detection"""
        # Reuse collection logic, ensuring it uses _detect_with_temporal_integration
        collected_turret_pos = []
        collected_image_points = []
        total_points_requested = len(positions_to_visit)
        points_collected_count = 0
        current_turret_pos = None
        if self.turret_positions: current_turret_pos = self.turret_positions[-1]
        elif self.turret_positions_raw: current_turret_pos = self.turret_positions_raw[-1]
        else: current_turret_pos = ((self.min_yaw + self.max_yaw)/2, (self.min_pitch+self.max_pitch)/2)

        for i, (target_yaw, target_pitch) in enumerate(positions_to_visit):
            target_yaw = np.clip(target_yaw, self.min_yaw, self.max_yaw)
            target_pitch = np.clip(target_pitch, self.min_pitch, self.max_pitch)
            target_pos = (target_yaw, target_pitch)

            print(f"  Collecting point {i+1}/{total_points_requested}: Target Yaw={target_yaw:.3f}, Pitch={target_pitch:.3f}")
            wait_time_settle = self._calculate_optimal_wait_time(current_turret_pos, target_pos)
            wait_time_laser = self.config['laser_stabilization_time']

            self.turret.rotate(target_yaw, target_pitch)
            time.sleep(wait_time_settle) # Wait for turret mechanics
            current_turret_pos = target_pos

            self.turret.laser_on()
            time.sleep(wait_time_laser) # Wait for laser power / camera exposure

            self.frame_ready.clear()
            if not self.frame_ready.wait(timeout=1.5):
                print("  Warning: Timeout waiting for new frame.")
                self.turret.laser_off()
                continue

            detected_point = None
            start_detect = time.time()
            if self.use_temporal and len(self.recent_frames) >= 3:
                 detected_point = self._detect_with_temporal_integration()
            else: # Fallback to single frame if needed
                 with self.lock:
                     if self.current_frame is None:
                         print("  Warning: Current frame is None.")
                         self.turret.laser_off(); continue
                     frame = self.current_frame.copy()
                 detected_point = self.detect_laser_point(frame)
            detect_time = time.time() - start_detect

            if detected_point:
                collected_turret_pos.append(target_pos)
                collected_image_points.append(detected_point)
                points_collected_count += 1
                print(f"    Point detected at ({detected_point[0]:.2f}, {detected_point[1]:.2f}) [Time: {detect_time*1000:.1f} ms]")

                if display:
                    with self.lock: display_frame = self.current_frame.copy()
                    pt_int = (int(round(detected_point[0])), int(round(detected_point[1])))
                    cv2.circle(display_frame, pt_int, 8, (0, 255, 0), 2)
                    cv2.line(display_frame, (pt_int[0]-10, pt_int[1]), (pt_int[0]+10, pt_int[1]), (0, 255, 0), 1)
                    cv2.line(display_frame, (pt_int[0], pt_int[1]-10), (pt_int[0], pt_int[1]+10), (0, 255, 0), 1)
                    text = f"Target:({target_yaw:.2f},{target_pitch:.2f})->Detect:({detected_point[0]:.2f},{detected_point[1]:.2f})"
                    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(display_frame, f"Collected: {points_collected_count}/{total_points_requested}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Calibration', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Collection interrupted.")
                        self.turret.laser_off()
                        return collected_turret_pos, collected_image_points
            else:
                print("    Warning: No laser point detected.")
                # Display frame without detection? (optional)

            self.turret.laser_off()
            time.sleep(0.05) # Small pause

        print(f"Data collection finished. Successfully collected {points_collected_count} / {total_points_requested} points.")
        return collected_turret_pos, collected_image_points


    # ============================================
    # Outlier Rejection (Adapted for GPR focus)
    # ============================================
    def _remove_outliers(self, turret_positions, image_points):
        """Applies RANSAC (preferred) or MAD outlier removal"""
        n_points = len(turret_positions)
        if n_points < 10: # Need a minimum number of points for robust outlier detection
            print("Warning: Too few points for reliable outlier removal. Skipping.")
            return np.array(turret_positions), np.array(image_points), 0

        turret_pos_np = np.array(turret_positions)
        image_points_np = np.array(image_points)

        if self.config['use_ransac'] and SKLEARN_AVAILABLE:
            print("Attempting RANSAC outlier removal...")
            order = self.config['ransac_base_poly_order']
            min_samples = max( (order+1)*(order+2)//2 + 1, int(n_points * self.config['ransac_min_samples_ratio']) )

            if min_samples >= n_points:
                 print("Warning: Not enough points for specified RANSAC min_samples ratio. Falling back to MAD.")
                 return self._remove_outliers_mad(turret_pos_np, image_points_np)

            try:
                 # Estimate residual scale first
                 pipe_temp = Pipeline([('poly', PolynomialFeatures(degree=order)), ('ridge', Ridge(alpha=1e-4))])
                 pipe_temp.fit(turret_pos_np, image_points_np)
                 preds_temp = pipe_temp.predict(turret_pos_np)
                 residuals = np.linalg.norm(image_points_np - preds_temp, axis=1)
                 mad = median_abs_deviation(residuals, scale='normal')
                 threshold = np.median(residuals) + self.config['ransac_residual_threshold_factor'] * (mad + 1e-6)
                 print(f"  RANSAC Residual Threshold: {threshold:.3f} pixels")

                 # Define RANSAC model (fit X and Y separately for potentially better outlier detection)
                 base_estimator = Pipeline([('poly', PolynomialFeatures(degree=order)), ('ridge', Ridge(alpha=1e-2))])
                 ransac_x = RANSACRegressor(base_estimator=base_estimator, min_samples=min_samples,
                                            residual_threshold=threshold, max_trials=100, random_state=42)
                 ransac_y = RANSACRegressor(base_estimator=base_estimator, min_samples=min_samples,
                                            residual_threshold=threshold, max_trials=100, random_state=43) # Different seed

                 ransac_x.fit(turret_pos_np, image_points_np[:, 0])
                 ransac_y.fit(turret_pos_np, image_points_np[:, 1])

                 inlier_mask = ransac_x.inlier_mask_ & ransac_y.inlier_mask_
                 num_outliers = n_points - np.sum(inlier_mask)
                 print(f"  RANSAC removed {num_outliers} outliers.")
                 return turret_pos_np[inlier_mask], image_points_np[inlier_mask], num_outliers

            except Exception as e:
                 print(f"  Warning: RANSAC failed ({e}). Falling back to MAD.")
                 return self._remove_outliers_mad(turret_pos_np, image_points_np)
        else:
            # Fallback to MAD
            return self._remove_outliers_mad(turret_pos_np, image_points_np)

    def _remove_outliers_mad(self, turret_pos_np, image_points_np):
        """Fallback outlier removal using Median Absolute Deviation"""
        print("Attempting MAD outlier removal...")
        n_points = len(turret_pos_np)
        order = min(2, max(1, int(np.sqrt(n_points) / 3))) # Very simple model order

        if n_points < (order+1)*(order+2)//2 + 2:
            print("  Not enough points for MAD removal.")
            return turret_pos_np, image_points_np, 0

        try:
            pipe_temp = Pipeline([('poly', PolynomialFeatures(degree=order)), ('ridge', Ridge(alpha=1e-4))])
            pipe_temp.fit(turret_pos_np, image_points_np)
            preds_temp = pipe_temp.predict(turret_pos_np)
            residuals = np.linalg.norm(image_points_np - preds_temp, axis=1)
            mad = median_abs_deviation(residuals, scale='normal')
            threshold = np.median(residuals) + self.config['ransac_residual_threshold_factor'] * (mad + 1e-6)

            inlier_mask = residuals <= threshold
            num_outliers = n_points - np.sum(inlier_mask)
            print(f"  MAD removed {num_outliers} outliers (threshold={threshold:.3f} pixels).")
            return turret_pos_np[inlier_mask], image_points_np[inlier_mask], num_outliers
        except Exception as e:
            print(f"  Warning: MAD removal failed ({e}). Returning original points.")
            return turret_pos_np, image_points_np, 0


    # ============================================
    # GPR Model Building & Prediction
    # ============================================
    def _build_gpr_model(self):
        """Builds the GPR calibration model after outlier removal"""
        if not self.turret_positions_raw:
            raise ValueError("No raw calibration points available to build model.")

        # --- Outlier Removal ---
        turret_pos_clean, image_points_clean, outliers_removed = self._remove_outliers(
            self.turret_positions_raw, self.image_points_raw
        )

        n_clean_points = len(turret_pos_clean)
        if n_clean_points < 10: # GPR generally needs at least this many
            raise ValueError(f"Insufficient number of points ({n_clean_points}) remaining after outlier removal for GPR.")

        # Store final points used for the model
        self.turret_positions = turret_pos_clean.tolist()
        self.image_points = image_points_clean.tolist()
        print(f"Building GPR model on {n_clean_points} points (removed {outliers_removed} outliers).")

        # --- Scaling ---
        turret_scaled = self.scaler_turret.fit_transform(turret_pos_clean)
        image_scaled = self.scaler_image.fit_transform(image_points_clean)

        # --- Kernel & Model Definition ---
        kernel = self.config['gpr_kernel']
        gpr_alpha = self.config['gpr_alpha']
        restarts = self.config['gpr_restarts']

        # Create separate GPR models for X and Y coordinates
        self.gp_model_x = GaussianProcessRegressor(
            kernel=kernel, # Kernel will be optimized
            n_restarts_optimizer=restarts,
            alpha=gpr_alpha, # Regularization / Noise term handling
            normalize_y=False, # We scale manually
            random_state=42
        )
        # Clone kernel structure but allow independent optimization
        kernel_y = kernel # Use same base kernel, will optimize independently
        # Note: If using clone(kernel), ensure it handles random_state correctly if kernel has one.
        # It's often fine to just reuse the kernel object instance here.
        self.gp_model_y = GaussianProcessRegressor(
            kernel=kernel_y,
            n_restarts_optimizer=restarts,
            alpha=gpr_alpha,
            normalize_y=False,
            random_state=43 # Different seed for Y optimization path
        )

        # --- Fit Models ---
        print("Fitting GPR models...")
        start_fit = time.time()
        try:
            self.gp_model_x.fit(turret_scaled, image_scaled[:, 0])
            self.gp_model_y.fit(turret_scaled, image_scaled[:, 1])
        except Exception as e:
             raise RuntimeError(f"Error during GPR fitting: {e}. Check data or kernel parameters.") from e
        fit_time = time.time() - start_fit
        print(f"GPR fitting completed in {fit_time:.2f}s.")
        print(f"  Optimized X Kernel: {self.gp_model_x.kernel_}")
        print(f"  Optimized Y Kernel: {self.gp_model_y.kernel_}")


        # --- Error Estimation (Cross-Validation) ---
        n_splits = min(5, n_clean_points)
        if n_splits < 2:
            print("Warning: Not enough points for CV. Reporting training error.")
            pred_x_scaled, std_x = self.gp_model_x.predict(turret_scaled, return_std=True)
            pred_y_scaled, std_y = self.gp_model_y.predict(turret_scaled, return_std=True)
            pred_scaled = np.column_stack((pred_x_scaled, pred_y_scaled))
            pred_orig = self.scaler_image.inverse_transform(pred_scaled)
            errors = np.linalg.norm(image_points_clean - pred_orig, axis=1)
            avg_error, max_error, std_error = np.mean(errors), np.max(errors), np.std(errors)
            uncertainty = np.mean(np.sqrt(std_x**2 + std_y**2) * np.sqrt(self.scaler_image.var_)) # Rough estimate
            print(f"  Training Avg Error: {avg_error:.3f} px")
            cv_results = None
        else:
            print(f"Performing {n_splits}-fold cross-validation...")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
            cv_errors = []
            cv_uncertainties = [] # Store predicted uncertainty on test sets
            start_cv = time.time()

            for fold, (train_idx, test_idx) in enumerate(kf.split(turret_scaled)):
                X_train, X_test = turret_scaled[train_idx], turret_scaled[test_idx]
                y_train_x, y_test_x = image_scaled[train_idx, 0], image_scaled[test_idx, 0]
                y_train_y, y_test_y = image_scaled[train_idx, 1], image_scaled[test_idx, 1]

                gp_x_cv = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=max(1, restarts//2), alpha=gpr_alpha, random_state=fold)
                gp_y_cv = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=max(1, restarts//2), alpha=gpr_alpha, random_state=fold + n_splits)

                try:
                    gp_x_cv.fit(X_train, y_train_x)
                    gp_y_cv.fit(X_train, y_train_y)

                    pred_x_test_scaled, std_x_test = gp_x_cv.predict(X_test, return_std=True)
                    pred_y_test_scaled, std_y_test = gp_y_cv.predict(X_test, return_std=True)

                    pred_test_scaled = np.column_stack((pred_x_test_scaled, pred_y_test_scaled))
                    pred_test_orig = self.scaler_image.inverse_transform(pred_test_scaled)
                    actual_test_orig = self.scaler_image.inverse_transform(np.column_stack((y_test_x, y_test_y)))

                    fold_errors = np.linalg.norm(actual_test_orig - pred_test_orig, axis=1)
                    cv_errors.extend(fold_errors)

                    # Convert std dev back to original scale (approximate)
                    std_orig_x = std_x_test * np.sqrt(self.scaler_image.var_[0])
                    std_orig_y = std_y_test * np.sqrt(self.scaler_image.var_[1])
                    fold_uncertainties = np.sqrt(std_orig_x**2 + std_orig_y**2) # Combine uncertainty
                    cv_uncertainties.extend(fold_uncertainties)

                except Exception as e:
                    print(f"  Warning: Error in CV fold {fold+1}: {e}. Skipping fold.")

            cv_time = time.time() - start_cv
            print(f"CV finished in {cv_time:.2f}s.")

            if cv_errors:
                errors = np.array(cv_errors)
                avg_error, max_error, std_error = np.mean(errors), np.max(errors), np.std(errors)
                uncertainty = np.mean(cv_uncertainties) if cv_uncertainties else float('nan')
                print(f"  Cross-Validation Avg Error: {avg_error:.3f} px")
                cv_results = {'errors': errors, 'uncertainties': cv_uncertainties}
            else:
                 print("Warning: CV failed. Reporting training error.")
                 # Fallback to training error calculation
                 pred_x_scaled, std_x = self.gp_model_x.predict(turret_scaled, return_std=True)
                 pred_y_scaled, std_y = self.gp_model_y.predict(turret_scaled, return_std=True)
                 pred_scaled = np.column_stack((pred_x_scaled, pred_y_scaled))
                 pred_orig = self.scaler_image.inverse_transform(pred_scaled)
                 errors = np.linalg.norm(image_points_clean - pred_orig, axis=1)
                 avg_error, max_error, std_error = np.mean(errors), np.max(errors), np.std(errors)
                 uncertainty = np.mean(np.sqrt(std_x**2 + std_y**2) * np.sqrt(self.scaler_image.var_))
                 cv_results = None

        # Store final results
        self.calibration_complete = True
        # Store key results for saving/reporting
        self.calibration_results_summary = {
            'avg_error': avg_error,
            'max_error': max_error,
            'std_error': std_error,
            'avg_uncertainty': uncertainty, # Average predicted std dev
            'num_points_final': n_clean_points,
            'num_outliers_removed': outliers_removed,
            'kernel_x': str(self.gp_model_x.kernel_),
            'kernel_y': str(self.gp_model_y.kernel_),
            'cv_results': cv_results # Contains detailed CV errors/uncertainties if successful
        }
        print(f"GPR Model build complete. Avg Error: {avg_error:.3f} px")

    def turret_to_image(self, yaw, pitch, return_std=False):
        """
        Convert turret coordinates (yaw, pitch) to image coordinates (x, y).

        Args:
            yaw (float): Turret yaw angle.
            pitch (float): Turret pitch angle.
            return_std (bool): If True, also return the predicted standard deviation.

        Returns:
            tuple: (x, y) image coordinates.
            or
            tuple: ( (x, y), (std_x, std_y) ) if return_std is True.

        Raises:
            ValueError: If calibration is not complete or models are missing.
        """
        if not self.calibration_complete or self.gp_model_x is None or self.gp_model_y is None:
            raise ValueError("Calibration not complete or GPR models not trained.")

        query_point = np.array([[yaw, pitch]])
        # Scale the input query point using the *fitted* scaler
        query_scaled = self.scaler_turret.transform(query_point)

        # Predict using the trained GPR models
        pred_x_scaled, std_x_scaled = self.gp_model_x.predict(query_scaled, return_std=True)
        pred_y_scaled, std_y_scaled = self.gp_model_y.predict(query_scaled, return_std=True)

        # Inverse transform the prediction to original image scale
        pred_scaled = np.column_stack((pred_x_scaled, pred_y_scaled))
        pred_orig = self.scaler_image.inverse_transform(pred_scaled)[0] # Get the first (only) row

        if return_std:
            # Scale the standard deviation back to the original image scale (approximate)
            # Assumes scaling is StandardScaler: std_orig = std_scaled * sqrt(scaler.var_)
            std_x_orig = std_x_scaled[0] * np.sqrt(self.scaler_image.var_[0])
            std_y_orig = std_y_scaled[0] * np.sqrt(self.scaler_image.var_[1])
            return tuple(pred_orig), (std_x_orig, std_y_orig)
        else:
            return tuple(pred_orig)

    # ============================================
    # Calibration Execution Flow
    # ============================================
    def run_calibration(self, num_points=50, display=True):
        """
        Executes the full GPR calibration process.

        Args:
            num_points (int): The target number of calibration points to collect.
            display (bool): Whether to show live calibration view and final plots.

        Returns:
            bool: True if calibration completed successfully, False otherwise.
        """
        if not self.turret.connected or not self.camera.streaming:
            print("Error: Turret or camera not connected.")
            return False

        print(f"--- Starting GPR Calibration ({num_points} points) ---")

        # Clear previous data
        self.turret_positions_raw = []
        self.image_points_raw = []
        self.turret_positions = []
        self.image_points = []
        self.gp_model_x = None
        self.gp_model_y = None
        self.calibration_complete = False
        self.calibration_results_summary = {}

        if display:
            cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Calibration', 800, 600)

        # --- Point Selection (Simple Grid + Random for GPR) ---
        # GPR benefits from good coverage. A grid ensures this, add random points for detail.
        num_grid = max(9, int(num_points * 0.6)) # ~60% grid
        num_random = num_points - num_grid

        n_side = int(np.ceil(np.sqrt(num_grid)))
        yaw_vals = np.linspace(self.min_yaw, self.max_yaw, n_side)
        pitch_vals = np.linspace(self.min_pitch, self.max_pitch, n_side)
        grid_points = list(set([(y, p) for y in yaw_vals for p in pitch_vals])) # Use set to avoid duplicates if n_side^2 > num_grid

        rng = np.random.default_rng(seed=42)
        random_yaws = rng.uniform(self.min_yaw, self.max_yaw, num_random)
        random_pitches = rng.uniform(self.min_pitch, self.max_pitch, num_random)
        random_points = list(zip(random_yaws, random_pitches))

        points_to_collect = grid_points + random_points
        random.shuffle(points_to_collect) # Shuffle combined list
        points_to_collect = points_to_collect[:num_points] # Ensure exact number

        print(f"Generated {len(points_to_collect)} points for collection (Grid + Random).")

        # --- Optimize Collection Order ---
        start_pos = ((self.min_yaw + self.max_yaw) / 2, (self.min_pitch + self.max_pitch) / 2)
        ordered_points = self._optimize_point_collection_order(start_pos, points_to_collect)

        # --- Collect Data ---
        print("Collecting calibration data...")
        new_turret_pos, new_image_points = self._collect_calibration_points(ordered_points, display=display)

        if not new_turret_pos:
            print("Error: No valid calibration points collected.")
            if display: cv2.destroyWindow('Calibration')
            return False

        self.turret_positions_raw = new_turret_pos
        self.image_points_raw = new_image_points

        # --- Build Model ---
        print("\nBuilding GPR model...")
        try:
            self._build_gpr_model() # This handles outlier removal internally
        except Exception as e:
            print(f"Error building GPR model: {e}")
            if display: cv2.destroyWindow('Calibration')
            return False

        print("\n--- Calibration Procedure Complete ---")
        if self.calibration_results_summary:
            print(f"  Model Type: GPR")
            print(f"  Points Used (Final): {self.calibration_results_summary['num_points_final']}")
            print(f"  Outliers Removed: {self.calibration_results_summary['num_outliers_removed']}")
            print(f"  Avg Error (CV or Train): {self.calibration_results_summary['avg_error']:.3f} pixels")
            print(f"  Avg Uncertainty (Predicted): {self.calibration_results_summary['avg_uncertainty']:.3f} pixels")

        # --- Visualize ---
        if display:
            cv2.destroyWindow('Calibration') # Close live view
            try:
                self.visualize_calibration()
            except Exception as e:
                print(f"Warning: Could not visualize calibration results: {e}")

        return True

    # ============================================
    # Visualization, Save/Load, Validation (Adapted)
    # ============================================
    def visualize_calibration(self):
        """Visualize GPR calibration results"""
        if not self.calibration_complete or not self.turret_positions:
            print("Calibration not complete or no data, cannot visualize.")
            return

        print("Generating GPR calibration visualization...")
        turret_pos = np.array(self.turret_positions)
        actual_points = np.array(self.image_points)

        # Predict points and uncertainty using the final model
        preds = []
        stds = []
        for y, p in turret_pos:
            pred, std_pair = self.turret_to_image(y, p, return_std=True)
            preds.append(pred)
            # Combine std_x, std_y into a single uncertainty metric (e.g., magnitude)
            stds.append(np.linalg.norm(std_pair))

        predicted_points = np.array(preds)
        uncertainties = np.array(stds)

        # Use errors from the stored summary (CV or training)
        errors = np.linalg.norm(actual_points - predicted_points, axis=1) # Recalculate training error for consistency check
        summary = self.calibration_results_summary
        avg_err_reported = summary.get('avg_error', np.mean(errors))
        max_err_reported = summary.get('max_error', np.max(errors))
        med_err_reported = np.median(errors) # Calculate median directly

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"GPR Calibration Results\nAvg Error: {avg_err_reported:.3f}px, Median: {med_err_reported:.3f}px, Max: {max_err_reported:.3f}px", fontsize=14)

        # Plot 1: Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(actual_points[:, 0], actual_points[:, 1], c='blue', label='Actual', alpha=0.6, s=20)
        ax.scatter(predicted_points[:, 0], predicted_points[:, 1], c='red', label='Predicted', alpha=0.6, s=20)
        for i in range(len(actual_points)):
            ax.plot([actual_points[i, 0], predicted_points[i, 0]], [actual_points[i, 1], predicted_points[i, 1]], 'k-', alpha=0.1)
        ax.set_title('Actual vs. Predicted Image Coordinates')
        ax.set_xlabel('Image X'); ax.set_ylabel('Image Y')
        ax.legend(); ax.grid(True, linestyle=':'); ax.axis('equal')

        # Plot 2: Error Heatmap
        ax = axes[0, 1]
        # Use reported errors if available (e.g., from CV), else use recalculated training errors
        error_source = summary['cv_results']['errors'] if summary.get('cv_results') else errors
        sc = ax.scatter(turret_pos[:, 0], turret_pos[:, 1], c=error_source, cmap='viridis', alpha=0.8, s=30, vmin=0, vmax=np.percentile(error_source, 95))
        plt.colorbar(sc, ax=ax, label='Error (pixels)')
        ax.set_title(f'Prediction Error by Turret Position ({"CV" if summary.get("cv_results") else "Train"})')
        ax.set_xlabel('Yaw'); ax.set_ylabel('Pitch')
        ax.grid(True, linestyle=':');
        ax.set_xlim(self.min_yaw - 0.05, self.max_yaw + 0.05)
        ax.set_ylim(self.min_pitch - 0.05, self.max_pitch + 0.05)

        # Plot 3: Error Histogram
        ax = axes[1, 0]
        ax.hist(error_source, bins=20, color='skyblue', edgecolor='black', alpha=0.8)
        ax.axvline(avg_err_reported, color='red', linestyle='--', linewidth=1.5, label=f'Avg: {avg_err_reported:.3f}px')
        ax.axvline(med_err_reported, color='purple', linestyle=':', linewidth=1.5, label=f'Median: {med_err_reported:.3f}px')
        ax.set_title(f'Error Distribution ({"CV" if summary.get("cv_results") else "Train"})')
        ax.set_xlabel('Error (pixels)'); ax.set_ylabel('Count')
        ax.legend(); ax.grid(True, linestyle=':')

        # Plot 4: Predicted Uncertainty Heatmap
        ax = axes[1, 1]
        sc = ax.scatter(turret_pos[:, 0], turret_pos[:, 1], c=uncertainties, cmap='magma', alpha=0.8, s=30, vmin=0, vmax=np.percentile(uncertainties, 95))
        plt.colorbar(sc, ax=ax, label='Predicted Std Dev (pixels)')
        ax.set_title('Model Uncertainty by Turret Position')
        ax.set_xlabel('Yaw'); ax.set_ylabel('Pitch')
        ax.grid(True, linestyle=':');
        ax.set_xlim(self.min_yaw - 0.05, self.max_yaw + 0.05)
        ax.set_ylim(self.min_pitch - 0.05, self.max_pitch + 0.05)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('gpr_calibration_results.png')
        print("Calibration visualization saved to 'gpr_calibration_results.png'")
        plt.show()

    def save_calibration(self, filename='gpr_turret_calibration.pkl'):
        """Save the GPR calibration model and parameters"""
        if not self.calibration_complete or self.gp_model_x is None or self.gp_model_y is None:
            print("Error: Calibration not complete or models invalid. Cannot save.")
            return False

        print(f"Saving GPR calibration to {filename}...")
        save_data = {
            'config': self.config,
            'model_type': self.model_type, # Should be 'gpr'
            'calibration_range_yaw': (self.min_yaw, self.max_yaw),
            'calibration_range_pitch': (self.min_pitch, self.max_pitch),
            # GPR specific data
            'gp_model_x': self.gp_model_x, # sklearn GPR models are picklable
            'gp_model_y': self.gp_model_y,
            'scaler_turret': self.scaler_turret, # Scalers are picklable
            'scaler_image': self.scaler_image,
            # Include final data points and summary for reference
            'turret_positions': self.turret_positions,
            'image_points': self.image_points,
            'calibration_results_summary': self.calibration_results_summary,
            # Raw points can be optionally saved for re-analysis
            # 'turret_positions_raw': self.turret_positions_raw,
            # 'image_points_raw': self.image_points_raw,
        }

        try:
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"GPR calibration successfully saved.")
            return True
        except Exception as e:
            print(f"Error saving GPR calibration data: {e}")
            return False

    def load_calibration(self, filename='gpr_turret_calibration.pkl'):
        """Load a GPR calibration model and parameters"""
        if not os.path.exists(filename):
            print(f"Error: Calibration file not found: {filename}")
            return False
        if not SKLEARN_AVAILABLE:
            print("Error: Cannot load GPR calibration - scikit-learn not installed.")
            return False

        print(f"Loading GPR calibration from {filename}...")
        try:
            with open(filename, 'rb') as f:
                load_data = pickle.load(f)

            # Basic check for GPR type
            if load_data.get('model_type') != 'gpr':
                print(f"Warning: File contains model type '{load_data.get('model_type')}', expected 'gpr'. Attempting to load anyway.")

            # Restore configuration and parameters
            self.config = load_data.get('config', self.config)
            self.model_type = 'gpr' # Set explicitly
            self.min_yaw, self.max_yaw = load_data.get('calibration_range_yaw', self.config['calibration_range_yaw'])
            self.min_pitch, self.max_pitch = load_data.get('calibration_range_pitch', self.config['calibration_range_pitch'])

            # Restore GPR models and scalers
            self.gp_model_x = load_data.get('gp_model_x')
            self.gp_model_y = load_data.get('gp_model_y')
            self.scaler_turret = load_data.get('scaler_turret')
            self.scaler_image = load_data.get('scaler_image')

            # Restore final data points and summary
            self.turret_positions = load_data.get('turret_positions', [])
            self.image_points = load_data.get('image_points', [])
            self.calibration_results_summary = load_data.get('calibration_results_summary', {})
            # Restore raw points if saved
            self.turret_positions_raw = load_data.get('turret_positions_raw', [])
            self.image_points_raw = load_data.get('image_points_raw', [])


            # Validate loaded components
            if not all([self.gp_model_x, self.gp_model_y, self.scaler_turret, self.scaler_image]):
                raise ValueError("Loaded data missing essential GPR model components (models or scalers).")

            self.calibration_complete = True
            print("GPR calibration loaded successfully.")
            print(f"  Points in Model: {len(self.turret_positions)}")
            if self.calibration_results_summary:
                 print(f"  Loaded Avg Error: {self.calibration_results_summary.get('avg_error', 'N/A'):.3f} pixels")
                 print(f"  Loaded Avg Uncertainty: {self.calibration_results_summary.get('avg_uncertainty', 'N/A'):.3f} pixels")

            return True

        except Exception as e:
            print(f"Error loading GPR calibration: {e}")
            # Reset state
            self.calibration_complete = False
            self.gp_model_x = None; self.gp_model_y = None
            self.scaler_turret = StandardScaler(); self.scaler_image = StandardScaler()
            self.turret_positions = []; self.image_points = []
            self.calibration_results_summary = {}
            return False

    def validate_calibration(self, num_points=15, strategy='random', display=True):
        """Validate GPR calibration accuracy"""
        # Reuse validation logic, it primarily depends on turret_to_image which is now GPR specific
        if not self.calibration_complete:
            print("Error: Calibration not complete. Cannot validate.")
            return False

        print(f"\n--- Starting GPR Validation ({num_points} points, Strategy: {strategy}) ---")
        model_turret_pts = np.array(self.turret_positions)
        if len(model_turret_pts) == 0:
             print("Error: No turret positions found in the loaded model.")
             return False
        min_yaw_data, max_yaw_data = np.min(model_turret_pts[:, 0]), np.max(model_turret_pts[:, 0])
        min_pitch_data, max_pitch_data = np.min(model_turret_pts[:, 1]), np.max(model_turret_pts[:, 1])
        print(f"Validating within data range: Yaw [{min_yaw_data:.3f}, {max_yaw_data:.3f}], Pitch [{min_pitch_data:.3f}, {max_pitch_data:.3f}]")

        # --- Select Validation Points (same logic as before) ---
        test_points = []
        rng = np.random.default_rng(seed=123)
        if strategy == 'random':
             yaws = rng.uniform(min_yaw_data, max_yaw_data, num_points)
             pitches = rng.uniform(min_pitch_data, max_pitch_data, num_points)
             test_points = list(zip(yaws, pitches))
        elif strategy == 'grid':
             n_side = int(np.ceil(np.sqrt(num_points)))
             yaw_vals = np.linspace(min_yaw_data, max_yaw_data, n_side)
             pitch_vals = np.linspace(min_pitch_data, max_pitch_data, n_side)
             test_points = [(y, p) for y in yaw_vals for p in pitch_vals]
             if len(test_points) > num_points: test_points = random.sample(test_points, num_points)
        elif strategy == 'edges':
             num_points = max(num_points, 9) # Corners + Edges + Center
             corners = [(min_yaw_data, min_pitch_data), (min_yaw_data, max_pitch_data), (max_yaw_data, min_pitch_data), (max_yaw_data, max_pitch_data)]
             mid_yaw, mid_pitch = (min_yaw_data + max_yaw_data)/2, (min_pitch_data + max_pitch_data)/2
             edges = [(min_yaw_data, mid_pitch), (max_yaw_data, mid_pitch), (mid_yaw, min_pitch_data), (mid_yaw, max_pitch_data)]
             center = [(mid_yaw, mid_pitch)]
             num_random = num_points - len(corners) - len(edges) - len(center)
             random_pts = []
             if num_random > 0:
                  yaws = rng.uniform(min_yaw_data, max_yaw_data, num_random)
                  pitches = rng.uniform(min_pitch_data, max_pitch_data, num_random)
                  random_pts = list(zip(yaws, pitches))
             test_points = corners + edges + center + random_pts
             rng.shuffle(test_points); test_points = test_points[:num_points]
        else: # Default random
             yaws = rng.uniform(min_yaw_data, max_yaw_data, num_points)
             pitches = rng.uniform(min_pitch_data, max_pitch_data, num_points)
             test_points = list(zip(yaws, pitches))

        # --- Perform Validation (same core loop) ---
        validation_results = {'turret_pos': [], 'predicted_img': [], 'actual_img': [], 'errors': [], 'pred_uncertainty': []}
        current_turret_pos = test_points[0]
        if display: cv2.namedWindow('Validation', cv2.WINDOW_NORMAL); cv2.resizeWindow('Validation', 800, 600)

        for i, (target_yaw, target_pitch) in enumerate(test_points):
             print(f"  Validating point {i+1}/{len(test_points)}: Target Yaw={target_yaw:.3f}, Pitch={target_pitch:.3f}")
             try:
                 pred_point, pred_std = self.turret_to_image(target_yaw, target_pitch, return_std=True)
                 pred_uncertainty = np.linalg.norm(pred_std) # Combine std_x, std_y
             except Exception as e: print(f"    Error predicting: {e}"); continue

             # Move, Detect (using temporal if enabled)
             wait_time_settle = self._calculate_optimal_wait_time(current_turret_pos, (target_yaw, target_pitch))
             wait_time_laser = self.config['laser_stabilization_time']
             current_turret_pos = (target_yaw, target_pitch)
             self.turret.rotate(target_yaw, target_pitch); time.sleep(wait_time_settle)
             self.turret.laser_on(); time.sleep(wait_time_laser)
             self.frame_ready.clear()
             actual_point = None
             if self.frame_ready.wait(timeout=1.5):
                 if self.use_temporal and len(self.recent_frames) >= 3: actual_point = self._detect_with_temporal_integration()
                 else:
                      with self.lock: frame = self.current_frame.copy() if self.current_frame is not None else None
                      if frame is not None: actual_point = self.detect_laser_point(frame)
             else: print("    Warning: Timeout waiting for validation frame.")
             self.turret.laser_off()

             # Record and Display
             if actual_point:
                 error = distance.euclidean(actual_point, pred_point)
                 print(f"    Pred:({pred_point[0]:.2f},{pred_point[1]:.2f}) Act:({actual_point[0]:.2f},{actual_point[1]:.2f}) Err:{error:.3f} Uncert:{pred_uncertainty:.3f}")
                 validation_results['turret_pos'].append((target_yaw, target_pitch))
                 validation_results['predicted_img'].append(pred_point)
                 validation_results['actual_img'].append(actual_point)
                 validation_results['errors'].append(error)
                 validation_results['pred_uncertainty'].append(pred_uncertainty)
                 if display: # Display logic (same as before, add uncertainty text?)
                     with self.lock: display_frame = self.current_frame.copy() if self.current_frame is not None else np.zeros((480,640,3), dtype=np.uint8)
                     pt_pred_int = (int(round(pred_point[0])), int(round(pred_point[1])))
                     pt_act_int = (int(round(actual_point[0])), int(round(actual_point[1])))
                     cv2.circle(display_frame, pt_pred_int, 12, (0, 255, 255), 2); cv2.putText(display_frame, "Pred", (pt_pred_int[0]+15,pt_pred_int[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                     cv2.circle(display_frame, pt_act_int, 10, (0, 255, 0), 2); cv2.line(display_frame, (pt_act_int[0]-10,pt_act_int[1]),(pt_act_int[0]+10,pt_act_int[1]),(0,255,0),1); cv2.line(display_frame, (pt_act_int[0],pt_act_int[1]-10),(pt_act_int[0],pt_act_int[1]+10),(0,255,0),1); cv2.putText(display_frame, "Actual", (pt_act_int[0]+15,pt_act_int[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                     cv2.line(display_frame, pt_pred_int, pt_act_int, (255, 0, 255), 1)
                     cv2.putText(display_frame, f"Err: {error:.2f}px Uncert: {pred_uncertainty:.2f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                     cv2.imshow('Validation', display_frame)
                     if cv2.waitKey(200) == ord('q'): print("Validation interrupted."); break
             else:
                 print(f"    Pred:({pred_point[0]:.2f},{pred_point[1]:.2f}) Actual: NOT DETECTED")
                 # Display logic for no detection (same as before)
                 if display:
                      with self.lock: display_frame = self.current_frame.copy() if self.current_frame is not None else np.zeros((480,640,3), dtype=np.uint8)
                      pt_pred_int = (int(round(pred_point[0])), int(round(pred_point[1])))
                      cv2.circle(display_frame, pt_pred_int, 12, (0, 0, 255), 2)
                      cv2.putText(display_frame, "Pred (No Detect)", (pt_pred_int[0]+15,pt_pred_int[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                      cv2.imshow('Validation', display_frame)
                      if cv2.waitKey(200) == ord('q'): print("Validation interrupted."); break


        # --- Validation Summary (same as before) ---
        if display: cv2.destroyWindow('Validation')
        num_validated = len(validation_results['errors'])
        if num_validated > 0:
             errors_np = np.array(validation_results['errors'])
             uncert_np = np.array(validation_results['pred_uncertainty'])
             avg_err, med_err, max_err, std_err = np.mean(errors_np), np.median(errors_np), np.max(errors_np), np.std(errors_np)
             avg_uncert = np.mean(uncert_np)
             print("\n--- GPR Validation Summary ---")
             print(f"  Points Validated: {num_validated} / {len(test_points)}")
             print(f"  Average Error:    {avg_err:.3f} pixels")
             print(f"  Median Error:     {med_err:.3f} pixels")
             print(f"  Maximum Error:    {max_err:.3f} pixels")
             print(f"  Std Dev Error:    {std_err:.3f} pixels")
             print(f"  Avg Pred Uncert:  {avg_uncert:.3f} pixels")

             # Plot error histogram
             plt.figure(figsize=(8, 6))
             plt.hist(errors_np, bins=15, edgecolor='black', alpha=0.7)
             plt.axvline(avg_err, color='red', ls='--', lw=1, label=f'Avg: {avg_err:.3f}')
             plt.axvline(med_err, color='purple', ls=':', lw=1, label=f'Median: {med_err:.3f}')
             plt.title(f'GPR Validation Error Distribution ({num_validated} points)'); plt.xlabel('Error (pixels)'); plt.ylabel('Count')
             plt.legend(); plt.grid(True, linestyle=':'); plt.tight_layout()
             plt.savefig('gpr_validation_error_distribution.png')
             print("Validation error plot saved to 'gpr_validation_error_distribution.png'")
             plt.show()
             return validation_results
        else:
             print("\n--- GPR Validation Summary ---")
             print("  No points were successfully validated.")
             return False

    def analyze_performance(self):
        """Analyze and print timing statistics"""
        # Reuse performance analysis logic
        if not self.timing_stats: print("No performance data collected."); return
        print("\n--- Performance Analysis ---")
        total_time, num_samples = 0, 0
        for op, times in self.timing_stats.items():
            if times:
                t = np.array(times) * 1000 # ms
                n = len(t); avg = np.mean(t); p50 = np.median(t); p95 = np.percentile(t, 95); max_t = np.max(t)
                print(f"  {op:<25}: Avg={avg:>7.2f} | Median={p50:>7.2f} | 95th={p95:>7.2f} | Max={max_t:>7.2f} ms ({n} samples)")
                if op == 'point_acquisition': total_time, num_samples = np.sum(t), n
        if num_samples > 0: avg_fps = 1000.0 / (total_time / num_samples) if total_time > 0 else float('inf'); print(f"  Estimated Avg FPS (Point Acquisition): {avg_fps:.1f}")
        print("--------------------------")

    def __del__(self):
        """Cleanup resources"""
        print("Shutting down GPR calibrator thread pool...")
        self.thread_pool.shutdown(wait=False) # Don't wait indefinitely on exit
        cv2.destroyAllWindows()
