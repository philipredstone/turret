import cv2
import numpy as np


class LaserSpotDetector:
    """Detects laser spot in camera images"""
    def __init__(self):
        # Adjustable parameters
        self.hue_tolerance = 10
        self.saturation_min = 50
        self.value_min = 50
        self.red_multiplier = 1.3  # How much red should dominate
        self.red_threshold = 100   # Minimum red value
        self.min_brightness = 150
        self.min_area = 5
        self.max_area = 5000
        self.circularity_threshold = 0.3
        
        self.update_color_ranges()
        
        # Debug mode
        self.debug_mode = False
        self.debug_callback = None  # Callback for debug visualizations
        
    def update_color_ranges(self):
        """Update HSV ranges based on current parameters"""
        # Lower and upper red ranges in HSV
        self.lower_red1 = np.array([0, self.saturation_min, self.value_min])
        self.upper_red1 = np.array([self.hue_tolerance, 255, 255])
        
        self.lower_red2 = np.array([180-self.hue_tolerance, self.saturation_min, self.value_min])
        self.upper_red2 = np.array([180, 255, 255])
    
    def set_debug_callback(self, callback):
        """Set a callback function for debug visualization"""
        self.debug_callback = callback
    
    def detect_laser_spot(self, frame, debug=False):
        """Detect red laser spot in the frame"""
        if frame is None:
            return None
        
        debug_info = {}
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color (both ends of hue spectrum)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = mask1 + mask2
        
        # Alternative method: detect bright red in BGR space
        b, g, r = cv2.split(frame)
        
        # Red channel should be significantly higher than others
        red_dominant = (r > g * self.red_multiplier) & (r > b * self.red_multiplier) & (r > self.red_threshold)
        red_dominant_mask = red_dominant.astype(np.uint8) * 255
        
        # Combine both detection methods
        combined_mask = cv2.bitwise_or(red_mask, red_dominant_mask)
        
        # Optional: add brightness filter
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, self.min_brightness, 255, cv2.THRESH_BINARY)
        
        # Try both with and without brightness filter
        final_mask_with_brightness = cv2.bitwise_and(combined_mask, bright_mask)
        final_mask_without_brightness = combined_mask
        
        # Clean up the masks
        kernel = np.ones((3, 3), np.uint8)
        final_mask1 = cv2.morphologyEx(final_mask_with_brightness, cv2.MORPH_OPEN, kernel)
        final_mask1 = cv2.morphologyEx(final_mask1, cv2.MORPH_CLOSE, kernel)
        
        final_mask2 = cv2.morphologyEx(final_mask_without_brightness, cv2.MORPH_OPEN, kernel)
        final_mask2 = cv2.morphologyEx(final_mask2, cv2.MORPH_CLOSE, kernel)
        
        # Try both masks
        contours1, _ = cv2.findContours(final_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(final_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store debug info
        debug_info['red_mask'] = red_mask
        debug_info['red_dominant'] = red_dominant_mask
        debug_info['combined_mask'] = combined_mask
        debug_info['bright_mask'] = bright_mask
        debug_info['final_mask1'] = final_mask1
        debug_info['final_mask2'] = final_mask2
        debug_info['contours1'] = contours1
        debug_info['contours2'] = contours2
        
        # Filter and find best contour
        laser_pos = None
        best_contour = None
        used_mask = 1
        
        # Try mask with brightness filter first
        contour, score = self._find_best_contour(contours1)
        if contour is not None:
            best_contour = contour
            used_mask = 1
        else:
            # Try mask without brightness filter
            contour, score = self._find_best_contour(contours2)
            if contour is not None:
                best_contour = contour
                used_mask = 2
        
        if best_contour is not None:
            # Get the centroid
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                laser_pos = (cx, cy)
        
        debug_info['best_contour'] = best_contour
        debug_info['used_mask'] = used_mask
        debug_info['laser_pos'] = laser_pos
        
        if debug and self.debug_callback:
            self.debug_callback(frame, debug_info)
        
        return laser_pos
    
    def _find_best_contour(self, contours):
        """Find the best contour based on area and circularity"""
        if not contours:
            return None, 0
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_area < area < self.max_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > self.circularity_threshold:
                        # Score based on area and circularity
                        score = circularity * np.sqrt(area)
                        
                        if score > best_score:
                            best_score = score
                            best_contour = contour
        
        return best_contour, best_score
    
    def detect_laser_spot_simple(self, frame):
        """Simpler detection method - just find the reddest spot"""
        if frame is None:
            return None
        
        # Split channels
        b, g, r = cv2.split(frame)
        
        # Create a "redness" score
        redness = r.astype(float) - np.maximum(g, b).astype(float)
        redness = np.clip(redness, 0, 255).astype(np.uint8)
        
        # Threshold to get the reddest areas
        _, thresh = cv2.threshold(redness, 50, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest red area within size constraints
        valid_contours = [c for c in contours if self.min_area < cv2.contourArea(c) < self.max_area]
        
        if not valid_contours:
            return None
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Get centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        
        return None