import numpy as np
import time
from threading import Thread
from laser_detector import LaserSpotDetector


class AutomaticCalibrator:
    """Handles automatic calibration using feedback control - Stabilized row transitions"""
    
    def __init__(self, camera_client, turret_client, calibration_system):
        self.camera = camera_client
        self.turret = turret_client
        self.calibration = calibration_system
        self.laser_detector = LaserSpotDetector()
        
        # Fast base PID parameters for normal movements
        self.kp_base = 0.0008
        self.ki_base = 0.00003
        self.kd_base = 0.0003
        
        # Movement constraints
        self.max_movement_coarse = 0.015
        self.max_movement_fine = 0.002
        self.adaptive_threshold = 50
        
        # Row transition specific parameters
        self.row_transition_step_size = 0.003  # Small steps for row transitions
        self.stabilization_time = 0.3  # Time to wait after large movements
        self.final_approach_threshold = 80  # Switch to ultra-conservative mode
        
        # Convergence parameters
        self.tolerance = 5  # pixels
        self.max_iterations = 200
        self.settle_time = 0.3
        
        # State
        self.is_calibrating = False
        self.cancel_calibration = False
        self.calibration_thread = None
        self.status_callback = None
        self.debug_visualization_callback = None
        self.start_corner_idx = 0
        self.current_target_idx = -1
        
        # Track successfully calibrated positions
        self.calibrated_positions = []
        
        # Board layout
        self.board_width = 9
    
    def set_debug_visualization_callback(self, callback):
        """Set callback for debug visualization"""
        self.debug_visualization_callback = callback
        self.laser_detector.set_debug_callback(callback)
    
    def _detect_row_transition(self, current_corner_idx, target_corner_idx):
        """Detect if we're transitioning between rows"""
        current_row = current_corner_idx // self.board_width
        target_row = target_corner_idx // self.board_width
        
        if current_row != target_row:
            current_col = current_corner_idx % self.board_width
            target_col = target_corner_idx % self.board_width
            
            # End-to-end transition (e.g., 8 to 9)
            if (current_col == self.board_width - 1 and target_col == 0) or \
               (current_col == 0 and target_col == self.board_width - 1):
                return True
        
        return False
    
    def _calculate_adaptive_gains(self, distance, phase="normal"):
        """Calculate adaptive PID gains based on distance and phase"""
        if phase == "row_transition_final":
            # Ultra-conservative for final approach after row transition
            kp = 0.0002
            ki = 0.00001
            kd = 0.0001
            max_movement = 0.0008
        elif phase == "row_transition":
            # Conservative for row transitions
            kp = 0.0004
            ki = 0.00002
            kd = 0.0002
            max_movement = self.row_transition_step_size
        else:
            # Normal aggressive gains
            if distance > self.adaptive_threshold:
                gain_multiplier = min(distance / self.adaptive_threshold, 5.0)
                kp = self.kp_base * gain_multiplier
                ki = self.ki_base * gain_multiplier
                kd = self.kd_base * gain_multiplier
                max_movement = self.max_movement_coarse
            else:
                gain_multiplier = max(distance / self.adaptive_threshold, 0.3)
                kp = self.kp_base * gain_multiplier
                ki = self.ki_base * gain_multiplier * 0.7
                kd = self.kd_base * gain_multiplier
                max_movement = self.max_movement_fine
        
        return kp, ki, kd, max_movement
    
    def start_automatic_calibration(self, corners, start_corner_idx=0, status_callback=None):
        """Start automatic calibration from a specific corner"""
        if self.is_calibrating:
            return False
        
        self.status_callback = status_callback
        self.cancel_calibration = False
        self.is_calibrating = True
        self.start_corner_idx = start_corner_idx
        self.calibrated_positions = []
        
        # Detect board width
        if len(corners) > 0:
            first_y = corners[0][1]
            for i in range(1, len(corners)):
                if abs(corners[i][1] - first_y) > 20:
                    self.board_width = i
                    break
        
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
    
    def _detect_laser_with_retries(self, frame, max_retries=3):
        """Attempt to detect laser with parameter adjustments if needed"""
        laser_pos = None
        
        for retry in range(max_retries):
            laser_pos = self.laser_detector.detect_laser_spot(frame, debug=(retry == 0))
            
            if laser_pos is not None:
                return laser_pos
            
            if retry == 0:
                laser_pos = self.laser_detector.detect_laser_spot_simple(frame)
                if laser_pos is not None:
                    return laser_pos
            elif retry == 1:
                self.laser_detector.min_area = max(2, self.laser_detector.min_area - 2)
                self.laser_detector.circularity_threshold = max(0.1, self.laser_detector.circularity_threshold - 0.1)
            
            time.sleep(0.05)
        
        return None
    
    def _stepwise_approach(self, target_corner):
        """Approach target using step-wise movements for row transitions"""
        self._update_status("Using stepwise approach for row transition")
        
        max_steps = 30
        
        for step in range(max_steps):
            if self.cancel_calibration:
                return False
            
            # Get current position
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            laser_pos = self._detect_laser_with_retries(frame)
            if laser_pos is None:
                self._update_status("Lost laser during stepwise approach")
                return False
            
            # Calculate error
            error_x = target_corner[0] - laser_pos[0]
            error_y = target_corner[1] - laser_pos[1]
            distance = np.sqrt(error_x**2 + error_y**2)
            
            self._update_status(f"Step {step}: Distance {distance:.1f}px")
            
            # Check if close enough
            if distance < self.tolerance:
                self._update_status("Stepwise approach successful")
                return True
            
            # Calculate movement direction (normalized)
            if distance > 0:
                direction_x = error_x / distance
                direction_y = error_y / distance
            else:
                direction_x = 0
                direction_y = 0
            
            # Determine step size based on distance
            if distance > 100:
                step_size = self.row_transition_step_size * 2
            elif distance > 50:
                step_size = self.row_transition_step_size
            else:
                step_size = self.row_transition_step_size * 0.5
            
            # Calculate movement
            movement_x = direction_x * step_size
            movement_y = direction_y * step_size
            
            # Apply movement
            new_yaw = self.turret.current_yaw + movement_x
            new_pitch = self.turret.current_pitch - movement_y
            
            # Clamp to valid ranges
            new_yaw = max(-1.0, min(0.0, new_yaw))
            new_pitch = max(0.0, min(0.5, new_pitch))
            
            # Move
            self.turret.rotate(new_yaw, new_pitch)
            
            # Wait for stabilization
            time.sleep(0.1)
        
        self._update_status("Stepwise approach reached max steps")
        return False
    
    def _handle_row_transition(self, target_corner):
        """Special handling for row transitions"""
        self._update_status("Handling row transition")
        
        # First, move closer using stepwise approach
        success = self._stepwise_approach(target_corner)
        
        if success:
            # Wait for system to stabilize
            time.sleep(self.stabilization_time)
            
            # Do final fine-tuning with ultra-conservative PID
            return self._center_laser_on_target(target_corner, phase="row_transition_final")
        
        return False
    
    def _center_laser_on_target(self, target, phase="normal"):
        """Use adaptive feedback control to center laser on target position"""
        # Initialize PID control variables
        integral_x = 0
        integral_y = 0
        prev_error_x = 0
        prev_error_y = 0
        
        # Track last successful position
        last_good_position = (self.turret.current_yaw, self.turret.current_pitch)
        last_movement = (0, 0)
        
        # Reset histories
        oscillation_history = []
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            if self.cancel_calibration:
                return False
            
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            laser_pos = self._detect_laser_with_retries(frame)
            
            if laser_pos is None:
                self._update_status(f"Lost laser tracking at iteration {iteration}")
                if not self._recover_laser_tracking(last_movement, last_good_position):
                    return False
                integral_x *= 0.5
                integral_y *= 0.5
                continue
            
            last_good_position = (self.turret.current_yaw, self.turret.current_pitch)
            
            # Calculate error
            error_x = target[0] - laser_pos[0]
            error_y = target[1] - laser_pos[1]
            distance = np.sqrt(error_x**2 + error_y**2)
            
            if distance < self.tolerance:
                self._update_status(f"Converged in {iteration} iterations (distance: {distance:.1f} pixels)")
                time.sleep(self.settle_time)
                return True
            
            # Calculate adaptive gains
            kp, ki, kd, max_movement = self._calculate_adaptive_gains(distance, phase)
            
            # Simple oscillation detection
            oscillation_history.append((error_x, error_y))
            if len(oscillation_history) > 6:
                oscillation_history.pop(0)
                
                # Check for oscillation
                direction_changes = 0
                for i in range(1, len(oscillation_history)):
                    prev_x, prev_y = oscillation_history[i-1]
                    curr_x, curr_y = oscillation_history[i]
                    if prev_x * curr_x < 0 or prev_y * curr_y < 0:
                        direction_changes += 1
                
                if direction_changes >= 4:
                    self._update_status(f"Oscillation detected at iteration {iteration}")
                    kp *= 0.5
                    ki *= 0.2
                    kd *= 0.7
                    max_movement *= 0.5
                    integral_x *= 0.5
                    integral_y *= 0.5
            
            # PID control
            integral_x += error_x
            integral_y += error_y
            
            # Limit integral
            integral_limit = 500 if phase == "row_transition_final" else 1000
            integral_x = np.clip(integral_x, -integral_limit, integral_limit)
            integral_y = np.clip(integral_y, -integral_limit, integral_limit)
            
            derivative_x = error_x - prev_error_x
            derivative_y = error_y - prev_error_y
            
            control_x = kp * error_x + ki * integral_x + kd * derivative_x
            control_y = kp * error_y + ki * integral_y + kd * derivative_y
            
            # Apply movement limiting
            control_magnitude = np.sqrt(control_x**2 + control_y**2)
            if control_magnitude > max_movement:
                scale_factor = max_movement / control_magnitude
                control_x *= scale_factor
                control_y *= scale_factor
            
            # Apply control
            new_yaw = self.turret.current_yaw + control_x
            new_pitch = self.turret.current_pitch - control_y
            
            new_yaw = max(-1.0, min(0.0, new_yaw))
            new_pitch = max(0.0, min(0.5, new_pitch))
            
            last_movement = (control_x, control_y)
            
            # Log movement
            if iteration % 10 == 0:
                self._update_status(
                    f"Iteration {iteration} [{phase}]: Distance {distance:.1f}px, "
                    f"Movement: yaw={control_x:.6f}, pitch={control_y:.6f}"
                )
            
            self.turret.rotate(new_yaw, new_pitch)
            
            prev_error_x = error_x
            prev_error_y = error_y
            
            # Delays based on phase
            if phase == "row_transition_final":
                time.sleep(0.1)  # Slower for stability
            elif phase == "row_transition":
                time.sleep(0.08)
            else:
                time.sleep(0.05)  # Fast for normal movement
        
        self._update_status(f"Failed to converge within iteration limit")
        return False
    
    def _recover_laser_tracking(self, last_movement, last_good_position):
        """Quick recovery mechanism"""
        self._update_status("Attempting quick recovery...")
        
        # Try reversing last movement
        reverse_yaw = self.turret.current_yaw - last_movement[0]
        reverse_pitch = self.turret.current_pitch - last_movement[1]
        
        self.turret.rotate(reverse_yaw, reverse_pitch)
        time.sleep(0.1)
        
        frame = self.camera.get_frame()
        if frame is not None:
            if self._detect_laser_with_retries(frame) is not None:
                return True
        
        # Return to last good position
        self.turret.rotate(last_good_position[0], last_good_position[1])
        time.sleep(0.2)
        
        frame = self.camera.get_frame()
        if frame is not None:
            if self._detect_laser_with_retries(frame) is not None:
                return True
        
        return False
    
    def _calibration_worker(self, corners):
        """Worker thread for automatic calibration"""
        try:
            self.turret.laser_on()
            time.sleep(0.5)
            
            self._update_status(f"Starting calibration from corner {self.start_corner_idx}")
            
            # Capture first point
            current_corner = corners[self.start_corner_idx]
            current_yaw = self.turret.current_yaw
            current_pitch = self.turret.current_pitch
            
            frame = self.camera.get_frame()
            if self._detect_laser_with_retries(frame) is None:
                self._update_status("Cannot detect laser at starting position")
                return
            
            self.calibration.calibration_data.append(
                (current_corner[0], current_corner[1], current_yaw, current_pitch)
            )
            self.calibrated_positions.append((self.start_corner_idx, current_yaw, current_pitch))
            self._update_status(f"Corner {self.start_corner_idx} captured")
            
            current_corner_idx = self.start_corner_idx
            
            # Process all corners
            for i in range(self.start_corner_idx + 1, len(corners)):
                if self.cancel_calibration:
                    break
                
                self._update_status(f"Moving to corner {i}/{len(corners)-1}")
                self.current_target_idx = i
                
                target_corner = corners[i]
                
                # Check if this is a row transition
                is_row_transition = self._detect_row_transition(current_corner_idx, i)
                
                success = False
                if is_row_transition:
                    # Use special row transition handling
                    success = self._handle_row_transition(target_corner)
                else:
                    # Normal fast movement
                    success = self._center_laser_on_target(target_corner)
                
                if success:
                    current_yaw = self.turret.current_yaw
                    current_pitch = self.turret.current_pitch
                    self.calibration.calibration_data.append(
                        (target_corner[0], target_corner[1], current_yaw, current_pitch)
                    )
                    self.calibrated_positions.append((i, current_yaw, current_pitch))
                    self._update_status(f"Corner {i} calibrated successfully")
                    current_corner_idx = i
                else:
                    self._update_status(f"Failed to reach corner {i}")
                
                time.sleep(0.3)
            
            # Handle wrap-around
            if self.start_corner_idx > 0:
                for i in range(0, self.start_corner_idx):
                    if self.cancel_calibration:
                        break
                    
                    self._update_status(f"Moving to corner {i}/{len(corners)-1}")
                    self.current_target_idx = i
                    
                    target_corner = corners[i]
                    is_row_transition = self._detect_row_transition(current_corner_idx, i)
                    
                    success = False
                    if is_row_transition:
                        success = self._handle_row_transition(target_corner)
                    else:
                        success = self._center_laser_on_target(target_corner)
                    
                    if success:
                        current_yaw = self.turret.current_yaw
                        current_pitch = self.turret.current_pitch
                        self.calibration.calibration_data.append(
                            (target_corner[0], target_corner[1], current_yaw, current_pitch)
                        )
                        self.calibrated_positions.append((i, current_yaw, current_pitch))
                        self._update_status(f"Corner {i} calibrated successfully")
                        current_corner_idx = i
                    else:
                        self._update_status(f"Failed to reach corner {i}")
                    
                    time.sleep(0.3)
            
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