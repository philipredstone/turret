import cv2
import numpy as np


class LaserSpotDetector:
    """
    Detects red laser spots in camera images using computer vision techniques.
    
    This class implements multiple detection strategies:
    1. HSV color space filtering for red colors
    2. BGR channel analysis for red dominance
    3. Brightness filtering for laser spots
    4. Contour analysis with area and circularity filtering
    
    The detector is designed to be robust against varying lighting conditions
    and can adjust its parameters dynamically.
    """
    
    def __init__(self):
        """Initialize laser detector with default parameters."""
        # HSV color filtering parameters
        self.hue_tolerance = 10      # Tolerance for red hue detection
        self.saturation_min = 50     # Minimum saturation for color detection
        self.value_min = 50          # Minimum brightness in HSV
        
        # BGR color analysis parameters
        self.red_multiplier = 1.3    # How much red should dominate green/blue
        self.red_threshold = 100     # Minimum red channel value
        
        # Brightness filtering
        self.min_brightness = 150    # Minimum brightness for laser spots
        
        # Contour filtering parameters
        self.min_area = 5                 # Minimum contour area in pixels
        self.max_area = 5000              # Maximum contour area in pixels
        self.circularity_threshold = 0.3  # Minimum circularity (0=line, 1=circle)
        
        # Update HSV ranges based on current parameters
        self.update_color_ranges()
        
        # Debug mode settings
        self.debug_mode = False
        self.debug_callback = None  # Callback for debug visualizations
        
    def update_color_ranges(self):
        """
        Update HSV color ranges based on current parameters.
        
        Red color in HSV spans across the hue boundary (0° and 180°),
        so we need two ranges to capture all red hues.
        """
        # Lower red range (around 0° hue)
        self.lower_red1 = np.array([0, self.saturation_min, self.value_min])
        self.upper_red1 = np.array([self.hue_tolerance, 255, 255])
        
        # Upper red range (around 180° hue)
        self.lower_red2 = np.array([180-self.hue_tolerance, self.saturation_min, self.value_min])
        self.upper_red2 = np.array([180, 255, 255])
    
    def set_debug_callback(self, callback):
        """
        Set a callback function for debug visualization.
        
        Args:
            callback: Function to call with debug visualization data
        """
        self.debug_callback = callback
    
    def detect_laser_spot(self, frame, debug=False):
        """
        Detect red laser spot in the frame using multiple detection methods.
        
        This is the main detection method that combines several approaches:
        1. HSV color space filtering
        2. BGR channel dominance analysis
        3. Brightness filtering
        4. Morphological operations for noise reduction
        5. Contour analysis with shape filtering
        
        Args:
            frame: Input BGR image from camera
            debug: Whether to generate debug visualization data
            
        Returns:
            tuple or None: (x, y) position of laser center, or None if not found
        """
        if frame is None:
            return None
        
        debug_info = {}  # Store debug information for visualization
        
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Method 1: HSV color space filtering
        # Create masks for both red hue ranges
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = mask1 + mask2  # Combine both red ranges
        
        # Method 2: BGR channel dominance analysis
        # Split BGR channels for individual analysis
        b, g, r = cv2.split(frame)
        
        # Detect pixels where red channel dominates significantly
        red_dominant = (r > g * self.red_multiplier) & \
                      (r > b * self.red_multiplier) & \
                      (r > self.red_threshold)
        red_dominant_mask = red_dominant.astype(np.uint8) * 255
        
        # Combine both detection methods
        combined_mask = cv2.bitwise_or(red_mask, red_dominant_mask)
        
        # Method 3: Brightness filtering (optional enhancement)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, self.min_brightness, 255, cv2.THRESH_BINARY)
        
        # Create two versions: with and without brightness filter
        final_mask_with_brightness = cv2.bitwise_and(combined_mask, bright_mask)
        final_mask_without_brightness = combined_mask
        
        # Morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        
        # Clean up mask with brightness filter
        final_mask1 = cv2.morphologyEx(final_mask_with_brightness, cv2.MORPH_OPEN, kernel)
        final_mask1 = cv2.morphologyEx(final_mask1, cv2.MORPH_CLOSE, kernel)
        
        # Clean up mask without brightness filter
        final_mask2 = cv2.morphologyEx(final_mask_without_brightness, cv2.MORPH_OPEN, kernel)
        final_mask2 = cv2.morphologyEx(final_mask2, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in both masks
        contours1, _ = cv2.findContours(final_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(final_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store debug information for visualization
        debug_info['red_mask'] = red_mask
        debug_info['red_dominant'] = red_dominant_mask
        debug_info['combined_mask'] = combined_mask
        debug_info['bright_mask'] = bright_mask
        debug_info['final_mask1'] = final_mask1
        debug_info['final_mask2'] = final_mask2
        debug_info['contours1'] = contours1
        debug_info['contours2'] = contours2
        
        # Find the best contour from both sets
        laser_pos = None
        best_contour = None
        used_mask = 1
        
        # Try mask with brightness filter first (usually more precise)
        contour, score = self._find_best_contour(contours1)
        if contour is not None:
            best_contour = contour
            used_mask = 1
        else:
            # Fall back to mask without brightness filter (more permissive)
            contour, score = self._find_best_contour(contours2)
            if contour is not None:
                best_contour = contour
                used_mask = 2
        
        # Calculate centroid of best contour
        if best_contour is not None:
            # Calculate moments for centroid computation
            M = cv2.moments(best_contour)
            if M["m00"] != 0:  # Avoid division by zero
                cx = int(M["m10"] / M["m00"])  # Centroid X
                cy = int(M["m01"] / M["m00"])  # Centroid Y
                laser_pos = (cx, cy)
        
        # Add final results to debug info
        debug_info['best_contour'] = best_contour
        debug_info['used_mask'] = used_mask
        debug_info['laser_pos'] = laser_pos
        
        # Call debug visualization if enabled
        if debug and self.debug_callback:
            self.debug_callback(frame, debug_info)
        
        return laser_pos
    
    def _find_best_contour(self, contours):
        """
        Find the best contour based on area and circularity criteria.
        
        This method evaluates each contour and selects the one most likely
        to be a laser spot based on size and shape criteria.
        
        Args:
            contours: List of contours from cv2.findContours
            
        Returns:
            tuple: (best_contour, best_score) or (None, 0) if no good contour found
        """
        if not contours:
            return None, 0
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by size - must be within reasonable range for laser spot
            if self.min_area < area < self.max_area:
                # Calculate circularity as a shape measure
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # Circularity = 4π*area / perimeter²
                    # Perfect circle = 1.0, straight line approaches 0
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Must meet minimum circularity threshold
                    if circularity > self.circularity_threshold:
                        # Score combines circularity and size
                        # Prefer larger, more circular shapes
                        score = circularity * np.sqrt(area)
                        
                        if score > best_score:
                            best_score = score
                            best_contour = contour
        
        return best_contour, best_score
    
    def detect_laser_spot_simple(self, frame):
        """
        Simpler detection method that finds the most red-dominant area.
        
        This is a fallback method that uses a different approach:
        it creates a "redness" score for each pixel and finds the largest
        concentrated red area.
        
        Args:
            frame: Input BGR image from camera
            
        Returns:
            tuple or None: (x, y) position of laser center, or None if not found
        """
        if frame is None:
            return None
        
        # Split color channels
        b, g, r = cv2.split(frame)
        
        # Create a "redness" score by subtracting other channels from red
        # This highlights areas where red dominates
        redness = r.astype(float) - np.maximum(g, b).astype(float)
        redness = np.clip(redness, 0, 255).astype(np.uint8)
        
        # Threshold to get the most red areas
        _, thresh = cv2.threshold(redness, 50, 255, cv2.THRESH_BINARY)
        
        # Clean up noise with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest valid contour within size constraints
        valid_contours = [c for c in contours if self.min_area < cv2.contourArea(c) < self.max_area]
        
        if not valid_contours:
            return None
        
        # Select the largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Calculate centroid of largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        
        return None