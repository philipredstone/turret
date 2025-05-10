import numpy as np
import time
from threading import Thread
from laser_detector import LaserSpotDetector


class AutomaticCalibrator:
    """Handles automatic calibration using feedback control"""
    
    def __init__(self, camera_client, turret_client, calibration_system):
        self.camera = camera_client
        self.turret = turret_client
        self.calibration = calibration_system
        self.laser_detector = LaserSpotDetector()
        
        # Adaptive PID controller parameters
        self.kp_base = 0.0005
        self.ki_base = 0.00002
        self.kd_base = 0.0002
        
        # Adaptive movement constraints
        self.max_movement_coarse = 0.01  # Larger movements for far targets
        self.max_movement_fine = 0.001   # Smaller movements for close targets
        self.adaptive_threshold = 50      # Pixel distance to switch modes
        
        # Convergence parameters
        self.tolerance = 5  # pixels
        self.max_iterations = 200
        self.settle_time = 0.5
        
        # Recovery parameters
        self.recovery_step_size = 0.001
        self.max_recovery_attempts = 10
        
        # Oscillation detection
        self.oscillation_history = []
        self.oscillation_threshold = 3  # Number of direction changes to detect oscillation
        
        # Convergence rate tracking
        self.convergence_history = []
        self.stuck_threshold = 5  # Iterations without improvement to consider "stuck"
        
        # State
        self.is_calibrating = False
        self.cancel_calibration = False
        self.calibration_thread = None
        self.status_callback = None
        self.debug_visualization_callback = None
        self.start_corner_idx = 0
        self.current_target_idx = -1
    
    def set_debug_visualization_callback(self, callback):
        """Set callback for debug visualization"""
        self.debug_visualization_callback = callback
        self.laser_detector.set_debug_callback(callback)
    
    def start_automatic_calibration(self, corners, start_corner_idx=0, status_callback=None):
        """Start automatic calibration from a specific corner"""
        if self.is_calibrating:
            return False
        
        self.status_callback = status_callback
        self.cancel_calibration = False
        self.is_calibrating = True
        self.start_corner_idx = start_corner_idx
        
        self.calibration_thread = Thread(
            target=self._calibration_worker,
            args=(corners,)
        )
        self.calibration_thread.start()
        return True
    
    def stop_calibration(self):
        """Stop the automatic calibration process"""
        self.cancel_calibration = True
        if self.calibration_thread:
            self.calibration_thread.join()
    
    def _calculate_adaptive_gains(self, distance):
        """Calculate adaptive PID gains based on distance to target"""
        # Use higher gains when far from target, lower when close
        if distance > self.adaptive_threshold:
            # Coarse mode - higher gains for faster convergence
            gain_multiplier = min(distance / self.adaptive_threshold, 5.0)
            kp = self.kp_base * gain_multiplier
            ki = self.ki_base * gain_multiplier
            kd = self.kd_base * gain_multiplier
            max_movement = self.max_movement_coarse
        else:
            # Fine mode - lower gains for stability
            gain_multiplier = max(distance / self.adaptive_threshold, 0.1)
            kp = self.kp_base * gain_multiplier
            ki = self.ki_base * gain_multiplier * 0.5  # Reduce integral gain in fine mode
            kd = self.kd_base * gain_multiplier
            max_movement = self.max_movement_fine
        
        return kp, ki, kd, max_movement
    
    def _detect_oscillation(self, error_x, error_y):
        """Detect if the system is oscillating around the target"""
        # Add current error to history
        self.oscillation_history.append((error_x, error_y))
        
        # Keep only recent history
        if len(self.oscillation_history) > 10:
            self.oscillation_history.pop(0)
        
        if len(self.oscillation_history) < 4:
            return False
        
        # Count direction changes
        direction_changes_x = 0
        direction_changes_y = 0
        
        for i in range(1, len(self.oscillation_history)):
            prev_error_x, prev_error_y = self.oscillation_history[i-1]
            curr_error_x, curr_error_y = self.oscillation_history[i]
            
            # Check for sign changes (direction changes)
            if prev_error_x * curr_error_x < 0:
                direction_changes_x += 1
            if prev_error_y * curr_error_y < 0:
                direction_changes_y += 1
        
        # If we see too many direction changes, we're likely oscillating
        return (direction_changes_x >= self.oscillation_threshold or 
                direction_changes_y >= self.oscillation_threshold)
    
    def _is_stuck(self, distance):
        """Check if we're stuck (not making progress)"""
        self.convergence_history.append(distance)
        
        # Keep only recent history
        if len(self.convergence_history) > self.stuck_threshold:
            self.convergence_history.pop(0)
        
        if len(self.convergence_history) < self.stuck_threshold:
            return False
        
        # Check if distance hasn't improved significantly
        min_distance = min(self.convergence_history)
        max_distance = max(self.convergence_history)
        distance_variation = max_distance - min_distance
        
        # If variation is small, we might be stuck
        return distance_variation < 2  # pixels
    
    def _detect_laser_with_retries(self, frame, max_retries=5):
        """Attempt to detect laser with parameter adjustments if needed"""
        laser_pos = None
        
        for retry in range(max_retries):
            laser_pos = self.laser_detector.detect_laser_spot(frame, debug=(retry == 0))
            
            if laser_pos is not None:
                return laser_pos
            
            # Adjust parameters progressively
            if retry == 0:
                # Try simple detection
                laser_pos = self.laser_detector.detect_laser_spot_simple(frame)
                if laser_pos is not None:
                    return laser_pos
            elif retry == 1:
                # Relax constraints
                self.laser_detector.min_area = max(2, self.laser_detector.min_area - 2)
                self.laser_detector.circularity_threshold = max(0.1, self.laser_detector.circularity_threshold - 0.1)
            elif retry == 2:
                # Further relax color constraints
                self.laser_detector.red_multiplier = max(1.1, self.laser_detector.red_multiplier - 0.1)
                self.laser_detector.red_threshold = max(60, self.laser_detector.red_threshold - 10)
                self.laser_detector.update_color_ranges()
            
            time.sleep(0.1)
        
        return None
    
    def _center_laser_on_target(self, target):
        """Use adaptive feedback control to center laser on target position"""
        # Initialize PID control variables
        integral_x = 0
        integral_y = 0
        prev_error_x = 0
        prev_error_y = 0
        
        # Track last successful position and movement
        last_good_position = (self.turret.current_yaw, self.turret.current_pitch)
        last_movement = (0, 0)
        
        # Reset tracking histories
        self.oscillation_history = []
        self.convergence_history = []
        
        # Phase tracking
        current_phase = "COARSE"  # Start in coarse mode
        phase_switch_count = 0
        
        for iteration in range(self.max_iterations):
            if self.cancel_calibration:
                return False
            
            # Get current frame
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Detect laser position
            laser_pos = self._detect_laser_with_retries(frame)
            
            if laser_pos is None:
                self._update_status(f"Lost laser tracking at iteration {iteration}")
                
                # Attempt recovery by reversing the last movement
                recovered = self._recover_laser_tracking(last_movement, last_good_position)
                
                if not recovered:
                    self._update_status("Failed to recover laser tracking")
                    return False
                
                # Reset PID state after recovery
                integral_x = 0
                integral_y = 0
                continue
            
            # Update last good position when we have tracking
            last_good_position = (self.turret.current_yaw, self.turret.current_pitch)
            
            # Calculate error
            error_x = target[0] - laser_pos[0]
            error_y = target[1] - laser_pos[1]
            
            # Check if we're close enough
            distance = np.sqrt(error_x**2 + error_y**2)
            
            if distance < self.tolerance:
                self._update_status(f"Converged in {iteration} iterations (distance: {distance:.1f} pixels)")
                time.sleep(self.settle_time)
                return True
            
            # Calculate adaptive gains based on distance
            kp, ki, kd, max_movement = self._calculate_adaptive_gains(distance)
            
            # Detect oscillation
            if self._detect_oscillation(error_x, error_y):
                self._update_status(f"Oscillation detected at iteration {iteration}, reducing gains")
                # Reduce gains to dampen oscillation
                kp *= 0.5
                ki *= 0.2
                kd *= 0.7
                max_movement *= 0.5
                # Reset integral term to prevent windup
                integral_x *= 0.5
                integral_y *= 0.5
            
            # Check if we're stuck
            if self._is_stuck(distance):
                self._update_status(f"Progress stalled at iteration {iteration}, adjusting strategy")
                # Try a different approach - maybe increase gains temporarily
                kp *= 1.5
                max_movement *= 1.5
                # Clear convergence history to reset stuck detection
                self.convergence_history = []
            
            # Update phase based on distance
            if distance > self.adaptive_threshold and current_phase == "FINE":
                current_phase = "COARSE"
                phase_switch_count += 1
                self._update_status(f"Switching to COARSE mode (distance: {distance:.1f}px)")
            elif distance <= self.adaptive_threshold and current_phase == "COARSE":
                current_phase = "FINE"
                phase_switch_count += 1
                self._update_status(f"Switching to FINE mode (distance: {distance:.1f}px)")
            
            # PID control calculation
            integral_x += error_x
            integral_y += error_y
            
            # Limit integral term to prevent windup
            integral_limit = 1000
            integral_x = np.clip(integral_x, -integral_limit, integral_limit)
            integral_y = np.clip(integral_y, -integral_limit, integral_limit)
            
            derivative_x = error_x - prev_error_x
            derivative_y = error_y - prev_error_y
            
            # Calculate control signals
            control_x = (kp * error_x + 
                        ki * integral_x + 
                        kd * derivative_x)
            
            control_y = (kp * error_y + 
                        ki * integral_y + 
                        kd * derivative_y)
            
            # Apply adaptive movement limiting
            control_magnitude = np.sqrt(control_x**2 + control_y**2)
            if control_magnitude > max_movement:
                # Scale down to maximum allowed movement
                scale_factor = max_movement / control_magnitude
                control_x *= scale_factor
                control_y *= scale_factor
            
            # Apply control
            new_yaw = self.turret.current_yaw + control_x
            new_pitch = self.turret.current_pitch - control_y
            
            # Clamp to valid ranges
            new_yaw = max(-1.0, min(0.0, new_yaw))
            new_pitch = max(0.0, min(0.5, new_pitch))
            
            # Store the movement for potential recovery
            last_movement = (control_x, control_y)
            
            # Log movement with phase information
            if iteration % 10 == 0:
                self._update_status(
                    f"Iteration {iteration} [{current_phase}]: Distance {distance:.1f}px, "
                    f"Movement: yaw={control_x:.6f}, pitch={control_y:.6f}, "
                    f"Max: {max_movement:.6f}"
                )
            
            # Send movement command
            self.turret.rotate(new_yaw, new_pitch)
            
            # Update for next iteration
            prev_error_x = error_x
            prev_error_y = error_y
            
            # Adaptive delay based on movement size
            if control_magnitude > max_movement * 0.8:
                # Large movement - wait longer
                time.sleep(0.15)
            else:
                # Small movement - shorter wait
                time.sleep(0.1)
        
        self._update_status(f"Failed to converge within iteration limit (final distance: {distance:.1f}px)")
        return False
    
    def _recover_laser_tracking(self, last_movement, last_good_position):
        """Attempt to recover laser tracking by reversing movement"""
        self._update_status("Attempting to recover laser tracking...")
        
        # First, try to reverse the last movement
        reverse_yaw_change = -last_movement[0]
        reverse_pitch_change = -last_movement[1]
        
        current_yaw = self.turret.current_yaw
        current_pitch = self.turret.current_pitch
        
        for attempt in range(self.max_recovery_attempts):
            # Move in the reverse direction with small steps
            recovery_yaw = current_yaw + reverse_yaw_change * (attempt + 1) * 0.5
            recovery_pitch = current_pitch + reverse_pitch_change * (attempt + 1) * 0.5
            
            # Clamp to valid ranges
            recovery_yaw = max(-1.0, min(0.0, recovery_yaw))
            recovery_pitch = max(0.0, min(0.5, recovery_pitch))
            
            self._update_status(f"Recovery attempt {attempt + 1}: Moving to ({recovery_yaw:.4f}, {recovery_pitch:.4f})")
            self.turret.rotate(recovery_yaw, recovery_pitch)
            time.sleep(0.2)
            
            # Check if we can see the laser again
            frame = self.camera.get_frame()
            if frame is not None:
                laser_pos = self._detect_laser_with_retries(frame)
                if laser_pos is not None:
                    self._update_status(f"Laser tracking recovered at attempt {attempt + 1}")
                    return True
        
        # If reversing didn't work, try returning to last known good position
        self._update_status("Reversal failed, returning to last known good position")
        self.turret.rotate(last_good_position[0], last_good_position[1])
        time.sleep(0.5)
        
        # Check if we can see the laser at the last good position
        frame = self.camera.get_frame()
        if frame is not None:
            laser_pos = self._detect_laser_with_retries(frame)
            if laser_pos is not None:
                self._update_status("Laser tracking recovered at last good position")
                return True
        
        # Last resort: scan in a small area around the last good position
        self._update_status("Attempting grid search around last position")
        return self._grid_search_recovery(last_good_position)
    
    def _grid_search_recovery(self, center_position):
        """Perform a small grid search around a position to find the laser"""
        center_yaw, center_pitch = center_position
        search_range = 0.005
        steps = 3
        
        for yaw_offset in np.linspace(-search_range, search_range, steps):
            for pitch_offset in np.linspace(-search_range, search_range, steps):
                search_yaw = center_yaw + yaw_offset
                search_pitch = center_pitch + pitch_offset
                
                # Clamp to valid ranges
                search_yaw = max(-1.0, min(0.0, search_yaw))
                search_pitch = max(0.0, min(0.5, search_pitch))
                
                self.turret.rotate(search_yaw, search_pitch)
                time.sleep(0.1)
                
                frame = self.camera.get_frame()
                if frame is not None:
                    laser_pos = self._detect_laser_with_retries(frame)
                    if laser_pos is not None:
                        self._update_status(f"Laser found during grid search at ({search_yaw:.4f}, {search_pitch:.4f})")
                        return True
        
        return False
    
    def _calibration_worker(self, corners):
        """Worker thread for automatic calibration"""
        try:
            # Ensure laser is on
            self.turret.laser_on()
            time.sleep(1.0)
            
            self._update_status(f"Starting calibration from corner {self.start_corner_idx}")
            
            # Capture the current position as the first calibration point
            current_corner = corners[self.start_corner_idx]
            current_yaw = self.turret.current_yaw
            current_pitch = self.turret.current_pitch
            
            # Verify the laser is visible
            frame = self.camera.get_frame()
            laser_pos = self._detect_laser_with_retries(frame)
            
            if laser_pos is None:
                self._update_status(f"Cannot detect laser at starting position. Please ensure laser is visible on corner {self.start_corner_idx}")
                return
            
            # Add the first calibration point
            self.calibration.calibration_data.append(
                (current_corner[0], current_corner[1], current_yaw, current_pitch)
            )
            self._update_status(f"Corner {self.start_corner_idx} captured at current position")
            
            # Move to each subsequent corner
            for i in range(self.start_corner_idx + 1, len(corners)):
                if self.cancel_calibration:
                    break
                
                self._update_status(f"Moving to corner {i}/{len(corners)-1}")
                
                # Set current target for visualization
                self.current_target_idx = i
                
                # Use feedback control to center laser on next corner
                target_corner = corners[i]
                success = self._center_laser_on_target(target_corner)
                
                if success:
                    # Capture calibration point
                    current_yaw = self.turret.current_yaw
                    current_pitch = self.turret.current_pitch
                    self.calibration.calibration_data.append(
                        (target_corner[0], target_corner[1], current_yaw, current_pitch)
                    )
                    self._update_status(f"Corner {i} calibrated successfully")
                else:
                    self._update_status(f"Failed to reach corner {i}")
                
                time.sleep(0.5)
            
            # Handle wrap-around if starting from non-zero corner
            if self.start_corner_idx > 0:
                for i in range(0, self.start_corner_idx):
                    if self.cancel_calibration:
                        break
                    
                    self._update_status(f"Moving to corner {i}/{len(corners)-1}")
                    self.current_target_idx = i
                    
                    target_corner = corners[i]
                    success = self._center_laser_on_target(target_corner)
                    
                    if success:
                        current_yaw = self.turret.current_yaw
                        current_pitch = self.turret.current_pitch
                        self.calibration.calibration_data.append(
                            (target_corner[0], target_corner[1], current_yaw, current_pitch)
                        )
                        self._update_status(f"Corner {i} calibrated successfully")
                    else:
                        self._update_status(f"Failed to reach corner {i}")
                    
                    time.sleep(0.5)
            
            # Turn off laser
            self.turret.laser_off()
            
            # Build calibration model
            if len(self.calibration.calibration_data) >= 5:
                success = self.calibration.build_transformation_model()
                if success:
                    self._update_status("Calibration completed successfully!")
                else:
                    self._update_status("Failed to build calibration model")
            else:
                self._update_status("Not enough calibration points collected")
        
        except Exception as e:
            self._update_status(f"Calibration error: {str(e)}")
        
        finally:
            self.is_calibrating = False
            self.turret.laser_off()
    
    def _update_status(self, message):
        """Update status through callback"""
        if self.status_callback:
            self.status_callback(message)
        print(f"Auto-calibration: {message}")