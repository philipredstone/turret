import numpy as np
import time
from threading import Thread
from laser_detector import LaserSpotDetector


class AutomaticCalibrator:
    """
    Handles automatic calibration using adaptive feedback control with specialized row transition handling.
    
    This class implements a sophisticated PID-based control system that can automatically move
    a laser turret to different checkerboard corners for calibration. It includes special handling
    for row transitions (end of one row to beginning of next row) which require more careful movement.
    """
    
    def __init__(self, camera_client, turret_client, calibration_system):
        """
        Initialize the automatic calibrator.
        
        Args:
            camera_client: Interface to the camera for getting video frames
            turret_client: Interface to the turret for movement commands
            calibration_system: The calibration system to store calibration points
        """
        self.camera = camera_client
        self.turret = turret_client
        self.calibration = calibration_system
        self.laser_detector = LaserSpotDetector()
        
        # Base PID parameters for normal movements - these provide aggressive but stable control
        self.kp_base = 0.0008    # Proportional gain - how much to react to current error
        self.ki_base = 0.00003   # Integral gain - how much to react to accumulated error
        self.kd_base = 0.0003    # Derivative gain - how much to react to rate of change
        
        # Movement constraints to prevent overshooting
        self.max_movement_coarse = 0.015  # Maximum movement for large distances
        self.max_movement_fine = 0.002    # Maximum movement for fine adjustments
        self.adaptive_threshold = 50      # Distance threshold for switching between coarse/fine
        
        # Row transition specific parameters - more conservative for stability
        self.row_transition_step_size = 0.003  # Small steps for row transitions
        self.stabilization_time = 0.3          # Time to wait after large movements
        self.final_approach_threshold = 80     # Switch to ultra-conservative mode
        
        # Convergence parameters
        self.tolerance = 5           # Success threshold in pixels
        self.max_iterations = 200    # Maximum attempts before giving up
        self.settle_time = 0.3       # Time to wait before confirming convergence
        
        # Thread control and state management
        self.is_calibrating = False
        self.cancel_calibration = False
        self.calibration_thread = None
        self.status_callback = None
        self.debug_visualization_callback = None
        self.start_corner_idx = 0
        self.current_target_idx = -1
        
        # Track successfully calibrated positions for recovery
        self.calibrated_positions = []
        
        # Board layout - assumes standard checkerboard (will be detected automatically)
        self.board_width = 9
    
    def set_debug_visualization_callback(self, callback):
        """
        Set callback for debug visualization of laser detection.
        
        Args:
            callback: Function to call with debug visualization data
        """
        self.debug_visualization_callback = callback
        self.laser_detector.set_debug_callback(callback)
    
    def _detect_row_transition(self, current_corner_idx, target_corner_idx):
        """
        Detect if we're transitioning between rows of the checkerboard.
        
        Row transitions are challenging because they often involve large movements
        that can cause the control system to overshoot.
        
        Args:
            current_corner_idx: Index of current corner position
            target_corner_idx: Index of target corner position
            
        Returns:
            bool: True if this is a row transition requiring special handling
        """
        current_row = current_corner_idx // self.board_width
        target_row = target_corner_idx // self.board_width
        
        if current_row != target_row:
            current_col = current_corner_idx % self.board_width
            target_col = target_corner_idx % self.board_width
            
            # End-to-end transition (e.g., corner 8 to corner 9 in a 9-wide board)
            # These are the most challenging transitions
            if (current_col == self.board_width - 1 and target_col == 0) or \
               (current_col == 0 and target_col == self.board_width - 1):
                return True
        
        return False
    
    def _calculate_adaptive_gains(self, distance, phase="normal"):
        """
        Calculate adaptive PID gains based on distance to target and movement phase.
        
        The system uses different gain sets for different situations:
        - Normal: Aggressive gains for fast movement
        - Row transition: Conservative gains for stability
        - Row transition final: Ultra-conservative for final approach
        
        Args:
            distance: Distance to target in pixels
            phase: Movement phase ("normal", "row_transition", "row_transition_final")
            
        Returns:
            tuple: (kp, ki, kd, max_movement) - PID gains and movement limit
        """
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
            # Normal aggressive gains with distance-based scaling
            if distance > self.adaptive_threshold:
                # Far from target - use aggressive gains
                gain_multiplier = min(distance / self.adaptive_threshold, 5.0)
                kp = self.kp_base * gain_multiplier
                ki = self.ki_base * gain_multiplier
                kd = self.kd_base * gain_multiplier
                max_movement = self.max_movement_coarse
            else:
                # Close to target - use conservative gains
                gain_multiplier = max(distance / self.adaptive_threshold, 0.3)
                kp = self.kp_base * gain_multiplier
                ki = self.ki_base * gain_multiplier * 0.7  # Reduce integral gain to prevent windup
                kd = self.kd_base * gain_multiplier
                max_movement = self.max_movement_fine
        
        return kp, ki, kd, max_movement
    
    def start_automatic_calibration(self, corners, start_corner_idx=0, status_callback=None):
        """
        Start automatic calibration from a specific corner.
        
        Args:
            corners: List of corner positions [(x, y), ...]
            start_corner_idx: Index of corner to start calibration from
            status_callback: Function to call with status updates
            
        Returns:
            bool: True if calibration started successfully, False if already running
        """
        if self.is_calibrating:
            return False
        
        # Initialize calibration state
        self.status_callback = status_callback
        self.cancel_calibration = False
        self.is_calibrating = True
        self.start_corner_idx = start_corner_idx
        self.calibrated_positions = []
        
        # Auto-detect board width from corner positions
        # Assumes corners are ordered row by row
        if len(corners) > 0:
            first_y = corners[0][1]
            for i in range(1, len(corners)):
                if abs(corners[i][1] - first_y) > 20:  # Found first corner in second row
                    self.board_width = i
                    break
        
        # Start calibration in separate thread to avoid blocking UI
        self.calibration_thread = Thread(
            target=self._calibration_worker,
            args=(corners,)
        )
        self.calibration_thread.start()
        return True
    
    def stop_calibration(self):
        """Stop the automatic calibration process gracefully."""
        self.cancel_calibration = True
        if self.calibration_thread:
            self.calibration_thread.join()
    
    def _detect_laser_with_retries(self, frame, max_retries=3):
        """
        Attempt to detect laser with parameter adjustments if initial detection fails.
        
        This implements a fallback strategy:
        1. Try normal detection with debug output
        2. Try simplified detection method
        3. Relax detection parameters and try again
        
        Args:
            frame: Camera frame to analyze
            max_retries: Maximum number of detection attempts
            
        Returns:
            tuple or None: (x, y) position of laser spot, or None if not found
        """
        laser_pos = None
        
        for retry in range(max_retries):
            # First attempt with debug visualization
            laser_pos = self.laser_detector.detect_laser_spot(frame, debug=(retry == 0))
            
            if laser_pos is not None:
                return laser_pos
            
            if retry == 0:
                # Try simplified detection method as fallback
                laser_pos = self.laser_detector.detect_laser_spot_simple(frame)
                if laser_pos is not None:
                    return laser_pos
            elif retry == 1:
                # Relax detection parameters for better detection
                self.laser_detector.min_area = max(2, self.laser_detector.min_area - 2)
                self.laser_detector.circularity_threshold = max(0.1, self.laser_detector.circularity_threshold - 0.1)
            
            time.sleep(0.05)  # Brief pause between attempts
        
        return None
    
    def _stepwise_approach(self, target_corner):
        """
        Approach target using small step-wise movements for row transitions.
        
        This method is used for challenging row transitions where normal PID control
        might overshoot. It moves in small, controlled steps toward the target.
        
        Args:
            target_corner: Target position (x, y) in image coordinates
            
        Returns:
            bool: True if target reached successfully, False otherwise
        """
        self._update_status("Using stepwise approach for row transition")
        
        max_steps = 30  # Prevent infinite loops
        
        for step in range(max_steps):
            if self.cancel_calibration:
                return False
            
            # Get current laser position
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            laser_pos = self._detect_laser_with_retries(frame)
            if laser_pos is None:
                self._update_status("Lost laser during stepwise approach")
                return False
            
            # Calculate error vector
            error_x = target_corner[0] - laser_pos[0]
            error_y = target_corner[1] - laser_pos[1]
            distance = np.sqrt(error_x**2 + error_y**2)
            
            self._update_status(f"Step {step}: Distance {distance:.1f}px")
            
            # Check if close enough to target
            if distance < self.tolerance:
                self._update_status("Stepwise approach successful")
                return True
            
            # Calculate normalized movement direction
            if distance > 0:
                direction_x = error_x / distance
                direction_y = error_y / distance
            else:
                direction_x = 0
                direction_y = 0
            
            # Determine step size based on remaining distance
            if distance > 100:
                step_size = self.row_transition_step_size * 2    # Larger steps when far
            elif distance > 50:
                step_size = self.row_transition_step_size        # Normal steps
            else:
                step_size = self.row_transition_step_size * 0.5  # Smaller steps when close
            
            # Calculate actual movement commands
            movement_x = direction_x * step_size
            movement_y = direction_y * step_size
            
            # Apply movement with bounds checking
            new_yaw = self.turret.current_yaw + movement_x
            new_pitch = self.turret.current_pitch - movement_y  # Note: Y is inverted
            
            # Clamp to valid turret ranges
            new_yaw = max(-1.0, min(0.0, new_yaw))      # Yaw range: -1.0 to 0.0
            new_pitch = max(0.0, min(0.5, new_pitch))   # Pitch range: 0.0 to 0.5
            
            # Execute movement
            self.turret.rotate(new_yaw, new_pitch)
            
            # Wait for mechanical settling
            time.sleep(0.1)
        
        self._update_status("Stepwise approach reached max steps")
        return False
    
    def _handle_row_transition(self, target_corner):
        """
        Special handling for row transitions using stepwise approach + fine tuning.
        
        Args:
            target_corner: Target corner position (x, y)
            
        Returns:
            bool: True if transition completed successfully
        """
        self._update_status("Handling row transition")
        
        # First phase: Move closer using stepwise approach
        success = self._stepwise_approach(target_corner)
        
        if success:
            # Wait for system to stabilize after movement
            time.sleep(self.stabilization_time)
            
            # Second phase: Final fine-tuning with ultra-conservative PID
            return self._center_laser_on_target(target_corner, phase="row_transition_final")
        
        return False
    
    def _center_laser_on_target(self, target, phase="normal"):
        """
        Use adaptive feedback control to center laser on target position.
        
        This implements a sophisticated PID controller with adaptive gains,
        oscillation detection, and automatic parameter adjustment.
        
        Args:
            target: Target position (x, y) in image coordinates
            phase: Control phase ("normal", "row_transition", "row_transition_final")
            
        Returns:
            bool: True if successfully centered on target, False otherwise
        """
        # Initialize PID control variables
        integral_x = 0          # Accumulated error for integral term
        integral_y = 0
        prev_error_x = 0        # Previous error for derivative term
        prev_error_y = 0
        
        # Track last successful position for recovery
        last_good_position = (self.turret.current_yaw, self.turret.current_pitch)
        last_movement = (0, 0)
        
        # Reset tracking histories
        oscillation_history = []   # Track error history for oscillation detection
        convergence_history = []   # Track convergence progress
        
        for iteration in range(self.max_iterations):
            if self.cancel_calibration:
                return False
            
            # Get current frame and detect laser position
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            laser_pos = self._detect_laser_with_retries(frame)
            
            if laser_pos is None:
                self._update_status(f"Lost laser tracking at iteration {iteration}")
                # Attempt recovery using last known good position
                if not self._recover_laser_tracking(last_movement, last_good_position):
                    return False
                # Reduce integral windup after recovery
                integral_x *= 0.5
                integral_y *= 0.5
                continue
            
            # Update last known good position
            last_good_position = (self.turret.current_yaw, self.turret.current_pitch)
            
            # Calculate current error
            error_x = target[0] - laser_pos[0]
            error_y = target[1] - laser_pos[1]
            distance = np.sqrt(error_x**2 + error_y**2)
            
            # Check for convergence
            if distance < self.tolerance:
                self._update_status(f"Converged in {iteration} iterations (distance: {distance:.1f} pixels)")
                time.sleep(self.settle_time)  # Allow system to settle
                return True
            
            # Calculate adaptive gains based on distance and phase
            kp, ki, kd, max_movement = self._calculate_adaptive_gains(distance, phase)
            
            # Oscillation detection and mitigation
            oscillation_history.append((error_x, error_y))
            if len(oscillation_history) > 6:
                oscillation_history.pop(0)
                
                # Check for rapid direction changes indicating oscillation
                direction_changes = 0
                for i in range(1, len(oscillation_history)):
                    prev_x, prev_y = oscillation_history[i-1]
                    curr_x, curr_y = oscillation_history[i]
                    # Count sign changes in error direction
                    if prev_x * curr_x < 0 or prev_y * curr_y < 0:
                        direction_changes += 1
                
                if direction_changes >= 4:  # Oscillation detected
                    self._update_status(f"Oscillation detected at iteration {iteration}")
                    # Reduce gains to dampen oscillation
                    kp *= 0.5
                    ki *= 0.2
                    kd *= 0.7
                    max_movement *= 0.5
                    # Reduce accumulated integral to prevent windup
                    integral_x *= 0.5
                    integral_y *= 0.5
            
            # PID control calculation
            integral_x += error_x
            integral_y += error_y
            
            # Integral windup protection
            integral_limit = 500 if phase == "row_transition_final" else 1000
            integral_x = np.clip(integral_x, -integral_limit, integral_limit)
            integral_y = np.clip(integral_y, -integral_limit, integral_limit)
            
            # Derivative calculation
            derivative_x = error_x - prev_error_x
            derivative_y = error_y - prev_error_y
            
            # Combined PID output
            control_x = kp * error_x + ki * integral_x + kd * derivative_x
            control_y = kp * error_y + ki * integral_y + kd * derivative_y
            
            # Apply movement limiting to prevent large jumps
            control_magnitude = np.sqrt(control_x**2 + control_y**2)
            if control_magnitude > max_movement:
                scale_factor = max_movement / control_magnitude
                control_x *= scale_factor
                control_y *= scale_factor
            
            # Apply control to turret with bounds checking
            new_yaw = self.turret.current_yaw + control_x
            new_pitch = self.turret.current_pitch - control_y  # Y coordinate is inverted
            
            # Enforce turret movement limits
            new_yaw = max(-1.0, min(0.0, new_yaw))
            new_pitch = max(0.0, min(0.5, new_pitch))
            
            last_movement = (control_x, control_y)
            
            # Periodic status logging
            if iteration % 10 == 0:
                self._update_status(
                    f"Iteration {iteration} [{phase}]: Distance {distance:.1f}px, "
                    f"Movement: yaw={control_x:.6f}, pitch={control_y:.6f}"
                )
            
            # Execute movement
            self.turret.rotate(new_yaw, new_pitch)
            
            # Update previous error for next iteration
            prev_error_x = error_x
            prev_error_y = error_y
            
            # Phase-appropriate delays for system stability
            if phase == "row_transition_final":
                time.sleep(0.1)  # Slower for maximum stability
            elif phase == "row_transition":
                time.sleep(0.08)
            else:
                time.sleep(0.05)  # Fast for normal movement
        
        self._update_status(f"Failed to converge within iteration limit")
        return False
    
    def _recover_laser_tracking(self, last_movement, last_good_position):
        """
        Quick recovery mechanism when laser tracking is lost.
        
        Args:
            last_movement: Last movement commands applied (yaw_delta, pitch_delta)
            last_good_position: Last known good turret position (yaw, pitch)
            
        Returns:
            bool: True if recovery successful, False otherwise
        """
        self._update_status("Attempting quick recovery...")
        
        # Strategy 1: Reverse the last movement
        reverse_yaw = self.turret.current_yaw - last_movement[0]
        reverse_pitch = self.turret.current_pitch - last_movement[1]
        
        self.turret.rotate(reverse_yaw, reverse_pitch)
        time.sleep(0.1)
        
        # Check if laser is visible again
        frame = self.camera.get_frame()
        if frame is not None:
            if self._detect_laser_with_retries(frame) is not None:
                return True
        
        # Strategy 2: Return to last known good position
        self.turret.rotate(last_good_position[0], last_good_position[1])
        time.sleep(0.2)
        
        # Final check
        frame = self.camera.get_frame()
        if frame is not None:
            if self._detect_laser_with_retries(frame) is not None:
                return True
        
        return False
    
    def _calibration_worker(self, corners):
        """
        Worker thread for automatic calibration process.
        
        This method runs in a separate thread to avoid blocking the UI.
        It sequentially visits all corners, starting from start_corner_idx.
        
        Args:
            corners: List of corner positions [(x, y), ...]
        """
        try:
            # Ensure laser is on for calibration
            self.turret.laser_on()
            time.sleep(0.5)
            
            self._update_status(f"Starting calibration from corner {self.start_corner_idx}")
            
            # Capture first point at starting position
            current_corner = corners[self.start_corner_idx]
            current_yaw = self.turret.current_yaw
            current_pitch = self.turret.current_pitch
            
            # Verify laser is visible at starting position
            frame = self.camera.get_frame()
            if self._detect_laser_with_retries(frame) is None:
                self._update_status("Cannot detect laser at starting position")
                return
            
            # Record first calibration point
            self.calibration.calibration_data.append(
                (current_corner[0], current_corner[1], current_yaw, current_pitch)
            )
            self.calibrated_positions.append((self.start_corner_idx, current_yaw, current_pitch))
            self._update_status(f"Corner {self.start_corner_idx} captured")
            
            current_corner_idx = self.start_corner_idx
            
            # Process all corners sequentially from start_corner_idx + 1 to end
            for i in range(self.start_corner_idx + 1, len(corners)):
                if self.cancel_calibration:
                    break
                
                self._update_status(f"Moving to corner {i}/{len(corners)-1}")
                self.current_target_idx = i
                
                target_corner = corners[i]
                
                # Check if this move requires special row transition handling
                is_row_transition = self._detect_row_transition(current_corner_idx, i)
                
                success = False
                if is_row_transition:
                    # Use special row transition handling
                    success = self._handle_row_transition(target_corner)
                else:
                    # Normal fast movement using PID control
                    success = self._center_laser_on_target(target_corner)
                
                if success:
                    # Record successful calibration point
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
                
                time.sleep(0.3)  # Brief pause between corners
            
            # Handle wrap-around: process corners 0 to start_corner_idx-1 if we started mid-sequence
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
            
            # Turn off laser when calibration is complete
            self.turret.laser_off()
            
            # Attempt to build calibration model if we have enough points
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
            # Cleanup regardless of success/failure
            self.is_calibrating = False
            self.turret.laser_off()
    
    def _update_status(self, message):
        """
        Update status through callback and console output.
        
        Args:
            message: Status message to display
        """
        if self.status_callback:
            self.status_callback(message)
        print(f"Auto-calibration: {message}")