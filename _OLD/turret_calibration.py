import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class TurretCalibration:
    def __init__(self, camera_client, turret_client):
        self.camera = camera_client
        self.turret = turret_client
        self.calibration_data = []
        self.board_size = (7, 6)  # Adjust to match your checkerboard
        self.pan_model = None
        self.tilt_model = None
        self.is_calibrated = False
        
    def detect_checkerboard(self, frame):
        """Detect checkerboard corners in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Return corners as a list of points
            return [(corner[0][0], corner[0][1]) for corner in corners], frame
        return None, frame
    
    def detect_laser(self, frame):
        """Detect the laser dot in the frame"""
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adjust these ranges based on your laser color (this is for red)
        lower_red = np.array([0, 100, 200])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([170, 100, 200])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        mask = mask1 + mask2
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (which should be the laser dot)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Filter by area to avoid noise
            if cv2.contourArea(largest_contour) > 5:  # Adjust threshold as needed
                M = cv2.moments(largest_contour)
                
                if M["m00"] > 0:
                    # Calculate centroid
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        
        return None
    
    def manual_calibration(self, num_points=10):
        """Perform manual calibration with user input"""
        print("Starting manual calibration...")
        print("Place the checkerboard in view of the camera and turret.")
        
        self.calibration_data = []
        
        while len(self.calibration_data) < num_points:
            # Get the latest frame
            frame = self.camera.get_frame()
            if frame is None:
                print("No frame available. Retrying...")
                time.sleep(0.5)
                continue
            
            # Detect the checkerboard
            corners, annotated_frame = self.detect_checkerboard(frame)
            if corners is None:
                print("Checkerboard not detected. Adjust position and try again.")
                # Display the frame
                cv2.imshow("Camera View", frame)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # Draw the corners
            for i, (x, y) in enumerate(corners):
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(annotated_frame, str(i), (int(x)+10, int(y)+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show the frame with corners
            cv2.imshow("Checkerboard Corners", annotated_frame)
            print("\nDetected", len(corners), "corners.")
            print("Select a corner (0-" + str(len(corners)-1) + ") to aim at, or 'q' to finish:")
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            
            # Get the corner index from key press
            try:
                corner_idx = int(chr(key))
                if corner_idx < 0 or corner_idx >= len(corners):
                    print("Invalid corner index.")
                    continue
            except:
                print("Invalid input. Please enter a number.")
                continue
            
            # Get the corner coordinates
            corner_x, corner_y = corners[corner_idx]
            print(f"Selected corner {corner_idx} at position ({corner_x:.1f}, {corner_y:.1f})")
            
            # Manual control of turret to aim at the corner
            print("Use keyboard to aim the turret at the selected corner:")
            print("w/s: up/down, a/d: left/right, space: turn on/off laser, enter: confirm position")
            
            # Turn on laser
            self.turret.laser_on()
            
            # Initial values
            yaw_angle = 90.0  # Starting position (adjust as needed)
            pitch_angle = 45.0  # Starting position (adjust as needed)
            step_size = 0.5  # Degrees per step
            
            laser_on = True
            
            while True:
                # Get latest frame
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                
                # Draw the target corner
                target_frame = frame.copy()
                cv2.circle(target_frame, (int(corner_x), int(corner_y)), 10, (0, 0, 255), 2)
                
                # Detect laser dot if laser is on
                if laser_on:
                    laser_pos = self.detect_laser(frame)
                    if laser_pos:
                        cv2.circle(target_frame, laser_pos, 8, (255, 0, 0), -1)
                
                # Display instructions
                cv2.putText(target_frame, f"Yaw: {yaw_angle:.1f}, Pitch: {pitch_angle:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(target_frame, "w/s: pitch up/down, a/d: yaw left/right", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(target_frame, "space: laser on/off, enter: confirm", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow("Aim Turret", target_frame)
                
                # Handle key press
                key = cv2.waitKey(100) & 0xFF
                
                if key == ord('w'):  # Pitch up
                    pitch_angle += step_size
                    self.turret.rotate(yaw_angle, pitch_angle)
                elif key == ord('s'):  # Pitch down
                    pitch_angle -= step_size
                    self.turret.rotate(yaw_angle, pitch_angle)
                elif key == ord('a'):  # Yaw left
                    yaw_angle -= step_size
                    self.turret.rotate(yaw_angle, pitch_angle)
                elif key == ord('d'):  # Yaw right
                    yaw_angle += step_size
                    self.turret.rotate(yaw_angle, pitch_angle)
                elif key == ord(' '):  # Toggle laser
                    laser_on = not laser_on
                    if laser_on:
                        self.turret.laser_on()
                    else:
                        self.turret.laser_off()
                elif key == 13:  # Enter key - confirm position
                    # Store the calibration point
                    self.calibration_data.append([corner_x, corner_y, yaw_angle, pitch_angle])
                    print(f"Calibration point added: Camera ({corner_x:.1f}, {corner_y:.1f}) -> Turret ({yaw_angle:.1f}, {pitch_angle:.1f})")
                    # Turn off laser
                    self.turret.laser_off()
                    break
                elif key == ord('q'):  # Quit without adding point
                    break
        
        # Clean up
        cv2.destroyAllWindows()
        self.turret.laser_off()
        
        # Build model if we have enough data
        if len(self.calibration_data) >= 5:
            self.build_transformation_model()
            return True
        else:
            print("Not enough calibration points collected. Minimum 5 points required.")
            return False
    
    def build_transformation_model(self):
        """Build the transformation model from calibration data"""
        if len(self.calibration_data) < 5:
            print("Not enough calibration points. Need at least 5.")
            return False
        
        data = np.array(self.calibration_data)
        
        # Extract features and targets
        X = data[:, 0:2]  # Camera x,y coordinates
        Y_pan = data[:, 2]  # Pan/Yaw angles
        Y_tilt = data[:, 3]  # Tilt/Pitch angles
        
        # Try polynomial regression for better accuracy
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # Create and train the models
        self.pan_model = LinearRegression().fit(X_poly, Y_pan)
        self.tilt_model = LinearRegression().fit(X_poly, Y_tilt)
        
        # Check model accuracy
        pan_score = self.pan_model.score(X_poly, Y_pan)
        tilt_score = self.tilt_model.score(X_poly, Y_tilt)
        
        print(f"Calibration complete!")
        print(f"Pan/Yaw model accuracy: {pan_score:.4f}")
        print(f"Tilt/Pitch model accuracy: {tilt_score:.4f}")
        
        self.poly_features = poly  # Save for later use
        self.is_calibrated = True
        
        # Visualize the model
        self.visualize_calibration()
        
        return True
    
    def visualize_calibration(self):
        """Visualize the calibration data and model predictions"""
        if not self.is_calibrated:
            print("Calibration not yet performed")
            return
        
        data = np.array(self.calibration_data)
        
        # Create a grid of points covering the image
        h, w = 480, 640  # Assuming this is your camera resolution
        grid_step = 50
        x_grid = np.arange(0, w, grid_step)
        y_grid = np.arange(0, h, grid_step)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Transform to polynomial features
        grid_poly = self.poly_features.transform(grid_points)
        
        # Predict angles for grid points
        pred_pan = self.pan_model.predict(grid_poly)
        pred_tilt = self.tilt_model.predict(grid_poly)
        
        # Reshape for plotting
        pan_grid = pred_pan.reshape(xx.shape)
        tilt_grid = pred_tilt.reshape(xx.shape)
        
        # Create plots
        fig = plt.figure(figsize=(15, 10))
        
        # 3D plot for pan/yaw
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o', label='Calibration Points')
        ax1.plot_surface(xx, yy, pan_grid, alpha=0.5, cmap='viridis')
        ax1.set_xlabel('Camera X')
        ax1.set_ylabel('Camera Y')
        ax1.set_zlabel('Pan/Yaw Angle')
        ax1.set_title('Camera Coordinates to Pan/Yaw Angle')
        
        # 3D plot for tilt/pitch
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.scatter(data[:, 0], data[:, 1], data[:, 3], c='b', marker='^', label='Calibration Points')
        ax2.plot_surface(xx, yy, tilt_grid, alpha=0.5, cmap='plasma')
        ax2.set_xlabel('Camera X')
        ax2.set_ylabel('Camera Y')
        ax2.set_zlabel('Tilt/Pitch Angle')
        ax2.set_title('Camera Coordinates to Tilt/Pitch Angle')
        
        # 2D plot showing vector field of pan/yaw changes
        ax3 = fig.add_subplot(223)
        ax3.quiver(xx, yy, pan_grid, np.zeros_like(pan_grid), angles='xy', scale_units='xy', scale=0.1)
        ax3.scatter(data[:, 0], data[:, 1], c='r', marker='o')
        ax3.set_xlabel('Camera X')
        ax3.set_ylabel('Camera Y')
        ax3.set_title('Pan/Yaw Vector Field')
        
        # 2D plot showing vector field of tilt/pitch changes
        ax4 = fig.add_subplot(224)
        ax4.quiver(xx, yy, np.zeros_like(tilt_grid), tilt_grid, angles='xy', scale_units='xy', scale=0.1)
        ax4.scatter(data[:, 0], data[:, 1], c='b', marker='^')
        ax4.set_xlabel('Camera X')
        ax4.set_ylabel('Camera Y')
        ax4.set_title('Tilt/Pitch Vector Field')
        
        plt.tight_layout()
        plt.show()
    
    def aim_at_target(self, target_x, target_y):
        """Aim the turret at a target point in the camera view"""
        if not self.is_calibrated:
            print("Calibration required before aiming")
            return False
        
        # Transform the target point
        target_point = np.array([[target_x, target_y]])
        target_poly = self.poly_features.transform(target_point)
        
        # Predict pan and tilt angles
        pan_angle = self.pan_model.predict(target_poly)[0]
        tilt_angle = self.tilt_model.predict(target_poly)[0]
        
        # Rotate turret to the predicted position
        return self.turret.rotate(pan_angle, tilt_angle)
    
    def interactive_targeting(self):
        """Interactive targeting mode to test calibration"""
        if not self.is_calibrated:
            print("Calibration required before targeting")
            return
        
        print("Interactive targeting mode.")
        print("Click on the camera view to aim the turret.")
        print("Press 'l' to toggle the laser.")
        print("Press 'q' to quit.")
        
        laser_on = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal laser_on
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Aiming at: ({x}, {y})")
                self.aim_at_target(x, y)
        
        cv2.namedWindow("Targeting")
        cv2.setMouseCallback("Targeting", mouse_callback)
        
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Display the frame
            cv2.imshow("Targeting", frame)
            
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                laser_on = not laser_on
                if laser_on:
                    self.turret.laser_on()
                else:
                    self.turret.laser_off()
        
        cv2.destroyAllWindows()
        self.turret.laser_off()