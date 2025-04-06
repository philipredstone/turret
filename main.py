import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
import os
import json
from PIL import Image, ImageTk

# Import the provided classes
from camera_stream_client import CameraStreamClient
from turret_client import TurretClient
from laser_calibration_helpers import LaserCalibrationHelpers

class TurretControlUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Laser Turret Control")
        self.root.geometry("1024x768")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize clients
        self.camera_client = CameraStreamClient()
        self.turret_client = TurretClient()
        self.calibration = LaserCalibrationHelpers()
        
        # Connection status
        self.camera_connected = False
        self.turret_connected = False
        
        # Current turret position
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        
        # Wall detection variables
        self.detection_running = False
        self.detection_thread = None
        
        # Wall overlay control
        self.show_wall_overlay = tk.BooleanVar(value=True)
        
        # Create UI elements
        self.create_ui()
        
        # Initialize update loop
        self.running = True
        self.update_thread = threading.Thread(target=self.update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def create_ui(self):
        # Main layout frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side: Camera view
        self.camera_frame = ttk.LabelFrame(self.main_frame, text="Camera View")
        self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera canvas for displaying the video feed
        self.camera_canvas = tk.Canvas(self.camera_frame, bg="black", width=640, height=480)
        self.camera_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.camera_canvas.bind("<Button-1>", self.on_camera_click)
        
        # Right side: Controls
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Connection controls
        self.connection_frame = ttk.LabelFrame(self.control_frame, text="Connection")
        self.connection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera connection
        ttk.Label(self.connection_frame, text="Camera IP:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.camera_ip = tk.StringVar(value="127.0.0.1")
        ttk.Entry(self.connection_frame, textvariable=self.camera_ip, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.connection_frame, text="Camera Port:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.camera_port = tk.StringVar(value="8080")
        ttk.Entry(self.connection_frame, textvariable=self.camera_port, width=15).grid(row=1, column=1, padx=5, pady=5)
        
        # Turret connection
        ttk.Label(self.connection_frame, text="Turret IP:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.turret_ip = tk.StringVar(value="127.0.0.1")
        ttk.Entry(self.connection_frame, textvariable=self.turret_ip, width=15).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(self.connection_frame, text="Turret Port:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.turret_port = tk.StringVar(value="8888")
        ttk.Entry(self.connection_frame, textvariable=self.turret_port, width=15).grid(row=3, column=1, padx=5, pady=5)
        
        # Connect button
        self.connect_btn = ttk.Button(self.connection_frame, text="Connect", command=self.connect)
        self.connect_btn.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Disconnect button
        self.disconnect_btn = ttk.Button(self.connection_frame, text="Disconnect", command=self.disconnect, state=tk.DISABLED)
        self.disconnect_btn.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Status indicators
        self.camera_status = tk.StringVar(value="Camera: Disconnected")
        self.turret_status = tk.StringVar(value="Turret: Disconnected")
        
        ttk.Label(self.connection_frame, textvariable=self.camera_status).grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.connection_frame, textvariable=self.turret_status).grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Turret controls
        self.turret_control_frame = ttk.LabelFrame(self.control_frame, text="Turret Control")
        self.turret_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Directional buttons in a grid
        self.up_btn = ttk.Button(self.turret_control_frame, text="Up", command=lambda: self.move_turret(0, 0.1))
        self.up_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.left_btn = ttk.Button(self.turret_control_frame, text="Left", command=lambda: self.move_turret(-0.1, 0))
        self.left_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.center_btn = ttk.Button(self.turret_control_frame, text="Center", command=self.center_turret)
        self.center_btn.grid(row=1, column=1, padx=5, pady=5)
        
        self.right_btn = ttk.Button(self.turret_control_frame, text="Right", command=lambda: self.move_turret(0.1, 0))
        self.right_btn.grid(row=1, column=2, padx=5, pady=5)
        
        self.down_btn = ttk.Button(self.turret_control_frame, text="Down", command=lambda: self.move_turret(0, -0.1))
        self.down_btn.grid(row=2, column=1, padx=5, pady=5)
        
        # Laser controls
        self.laser_frame = ttk.LabelFrame(self.control_frame, text="Laser")
        self.laser_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.laser_on_btn = ttk.Button(self.laser_frame, text="Laser ON", command=self.laser_on)
        self.laser_on_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.laser_off_btn = ttk.Button(self.laser_frame, text="Laser OFF", command=self.laser_off)
        self.laser_off_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Position display
        self.position_frame = ttk.LabelFrame(self.control_frame, text="Position")
        self.position_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.position_var = tk.StringVar(value="Yaw: 0.00, Pitch: 0.00")
        ttk.Label(self.position_frame, textvariable=self.position_var).pack(padx=5, pady=5)
        
        # Wall detection controls
        self.wall_detection_frame = ttk.LabelFrame(self.control_frame, text="Wall Detection")
        self.wall_detection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Detection parameters
        params_frame = ttk.Frame(self.wall_detection_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Grid Size:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.grid_size = tk.IntVar(value=6)
        ttk.Combobox(params_frame, textvariable=self.grid_size, values=[4, 5, 6, 7, 8, 9, 10], width=5).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Speed:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.scan_speed = tk.DoubleVar(value=0.5)
        ttk.Scale(params_frame, from_=0.1, to=0.9, variable=self.scan_speed, orient=tk.HORIZONTAL, length=100).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Detection buttons
        self.start_detection_btn = ttk.Button(self.wall_detection_frame, text="Start Wall Detection", command=self.start_wall_detection)
        self.start_detection_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.cancel_detection_btn = ttk.Button(self.wall_detection_frame, text="Cancel Detection", command=self.cancel_wall_detection, state=tk.DISABLED)
        self.cancel_detection_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Detection status
        self.detection_status = tk.StringVar(value="Ready")
        ttk.Label(self.wall_detection_frame, textvariable=self.detection_status).pack(padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(self.wall_detection_frame, variable=self.progress_var, length=200).pack(padx=5, pady=5)
        
        # Wall overlay checkbox
        overlay_check = ttk.Checkbutton(self.wall_detection_frame, text="Show Wall Overlay on Camera", 
                                      variable=self.show_wall_overlay)
        overlay_check.pack(fill=tk.X, padx=5, pady=5)
        
        # Set all control buttons to disabled initially
        self.set_controls_state(tk.DISABLED)
    
    def set_controls_state(self, state):
        """Enable or disable control buttons based on connection status"""
        self.up_btn.config(state=state)
        self.left_btn.config(state=state)
        self.center_btn.config(state=state)
        self.right_btn.config(state=state)
        self.down_btn.config(state=state)
        self.laser_on_btn.config(state=state)
        self.laser_off_btn.config(state=state)
        self.start_detection_btn.config(state=state)
    
    def connect(self):
        """Connect to the camera and turret"""
        # Get connection parameters
        camera_ip = self.camera_ip.get()
        camera_port = int(self.camera_port.get())
        turret_ip = self.turret_ip.get()
        turret_port = int(self.turret_port.get())
        
        # Try to connect to camera
        self.camera_client.error_callback = self.on_camera_error
        self.camera_client.frame_callback = self.on_frame_received
        
        if self.camera_client.connect(host=camera_ip, port=camera_port):
            self.camera_connected = True
            self.camera_status.set("Camera: Connected")
        else:
            messagebox.showerror("Connection Error", "Failed to connect to camera")
            return
        
        # Try to connect to turret
        self.turret_client.error_callback = self.on_turret_error
        self.turret_client.response_callback = self.on_turret_response
        
        self.turret_client.host = turret_ip
        self.turret_client.port = turret_port
        
        if self.turret_client.connect():
            self.turret_connected = True
            self.turret_status.set("Turret: Connected")
        else:
            messagebox.showerror("Connection Error", "Failed to connect to turret")
            self.camera_client.disconnect()
            self.camera_connected = False
            self.camera_status.set("Camera: Disconnected")
            return
        
        # Update UI
        self.connect_btn.config(state=tk.DISABLED)
        self.disconnect_btn.config(state=tk.NORMAL)
        self.set_controls_state(tk.NORMAL)
        
        # Center the turret on connect
        self.center_turret()
    
    def disconnect(self):
        """Disconnect from camera and turret"""
        if self.camera_connected:
            self.camera_client.disconnect()
            self.camera_connected = False
            self.camera_status.set("Camera: Disconnected")
        
        if self.turret_connected:
            self.turret_client.disconnect()
            self.turret_connected = False
            self.turret_status.set("Turret: Disconnected")
        
        # Update UI
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        self.set_controls_state(tk.DISABLED)
        
        # Clear the canvas
        self.camera_canvas.delete("all")
    
    def update_loop(self):
        """Main update loop for the UI"""
        last_frame_time = 0
        frame_interval = 1/30  # Limit to 30 FPS to reduce flicker
        
        while self.running:
            current_time = time.time()
            
            # Limit frame rate to reduce flickering
            if current_time - last_frame_time >= frame_interval:
                last_frame_time = current_time
                
                if self.camera_connected:
                    frame = self.camera_client.get_frame()
                    if frame is not None:
                        # Process the frame
                        self.process_frame(frame)
            
            # Update position display without redrawing frame
            self.position_var.set(f"Yaw: {self.current_yaw:.2f}, Pitch: {self.current_pitch:.2f}")
            
            # Sleep to avoid excessive CPU usage
            time.sleep(0.01)  # Small sleep but let the main thread process events
    
    def process_frame(self, frame):
        """Process and display a video frame"""
        try:
            # Make a copy of the frame to work with
            display_frame = frame.copy()
            
            # Check for laser dot in the frame
            laser_pos = self.calibration.detect_laser_dot(frame)
            
            # If a laser dot is detected, draw a circle around it
            if laser_pos:
                x, y = laser_pos
                cv2.circle(display_frame, (x, y), 10, (0, 255, 0), 2)
            
            # Draw wall overlay if enabled and we have wall data
            if self.show_wall_overlay.get() and hasattr(self, 'wall_map') and self.wall_map is not None:
                self._draw_wall_overlay(display_frame)
            
            # Convert to tkinter-compatible image
            image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Resize to fit canvas while maintaining aspect ratio
            canvas_width = self.camera_canvas.winfo_width()
            canvas_height = self.camera_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Skip when canvas not properly sized yet
                # Calculate scale factor based on the canvas size
                # We want to fill as much of the canvas as possible
                scale_factor = min(canvas_width / w, canvas_height / h)
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                
                if new_w < 100 or new_h < 100:  # If scaling is too small, use minimum size
                    scale_factor = max(100 / w, 100 / h)
                    new_w = int(w * scale_factor)
                    new_h = int(h * scale_factor)
                
                image = cv2.resize(image, (new_w, new_h))
            
            # Convert to ImageTk format
            photo_image = ImageTk.PhotoImage(image=Image.fromarray(image))
            
            # Calculate x,y to center the image in the canvas
            x = (canvas_width - photo_image.width()) // 2
            y = (canvas_height - photo_image.height()) // 2
            
            # Update canvas using after() to ensure thread safety
            self.root.after(0, self._update_canvas, photo_image, x, y)
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
    
    def _draw_wall_overlay(self, frame):
        """Draw the exact hit points where the laser detected the wall"""
        if not hasattr(self, 'wall_map') or self.wall_map is None or 'hits' not in self.wall_map:
            return
            
        # Get the frame dimensions
        h, w = frame.shape[:2]
        
        # Get the hit points - actual turret coordinates where the wall was detected
        hits = self.wall_map.get('hits', [])
        
        if not hits:  # No hit points
            return
            
        # Draw each hit point directly - no polygon, no grid, just the exact hit locations
        for yaw, pitch in hits:
            # Convert from turret coordinates to camera pixel coordinates
            # Simple linear mapping:
            # yaw goes from -1 (left) to 0 (middle) -> map to 0 to w/2
            px = int(w/2 * (yaw + 1))
            # pitch goes from 0 (middle) to 1 (top) -> map to h/2 to 0
            py = int(h/2 * (1 - pitch))
            
            # Draw a circle at the exact hit location
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)  # Solid green circle
        
        # Add a label
        cv2.putText(frame, "Detected Wall Points", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _update_canvas(self, photo_image, x, y):
        """Update the canvas with new image (called from the main thread)"""
        self.camera_canvas.delete("all")
        self.camera_canvas.create_image(x, y, image=photo_image, anchor=tk.NW)
        self.camera_canvas.image = photo_image  # Keep a reference
    
    def on_camera_click(self, event):
        """Handle clicks on the camera view"""
        if not self.camera_connected or not self.turret_connected:
            return
        
        # Get normalized coordinates ([-1, 1] range)
        canvas_width = self.camera_canvas.winfo_width()
        canvas_height = self.camera_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Get normalized position
        x_norm = 2 * (event.x / canvas_width) - 1
        y_norm = 2 * (event.y / canvas_height) - 1
        
        # Use calibration if available, otherwise simple mapping
        if self.calibration.calibration is not None:
            # Use calibration to map pixel to turret position
            yaw, pitch = self.calibration.pixel_to_turret(event.x, event.y)
        else:
            # Simple linear mapping if no calibration is available
            # Map x from [0, canvas_width] to [-1, 0]
            yaw = max(-1.0, min(0.0, x_norm))
            # Map y from [0, canvas_height] to [0, 1] (invert and clamp)
            pitch = max(0.0, min(1.0, -y_norm))
        
        # Move the turret
        self.rotate_turret(yaw, pitch)
    
    def move_turret(self, yaw_delta, pitch_delta):
        """Move the turret by the specified amount"""
        if not self.turret_connected:
            return
        
        # Calculate new position
        new_yaw = max(-1.0, min(0.0, self.current_yaw + yaw_delta))
        new_pitch = max(0.0, min(1.0, self.current_pitch + pitch_delta))
        
        # Rotate to new position
        self.rotate_turret(new_yaw, new_pitch)
    
    def rotate_turret(self, yaw, pitch):
        """Rotate the turret to a specific position"""
        if not self.turret_connected:
            return
        
        # Clamp values to valid range
        yaw = max(-1.0, min(0.0, yaw))
        pitch = max(0.0, min(1.0, pitch))
        
        # Update current position
        self.current_yaw = yaw
        self.current_pitch = pitch
        
        # Send command to turret
        self.turret_client.rotate(yaw, pitch)
    
    def center_turret(self):
        """Center the turret within its valid range"""
        if not self.turret_connected:
            return
        
        # Center is now at (-0.5, 0.5) given the constraints
        self.rotate_turret(-0.5, 0.5)
    
    def laser_on(self):
        """Turn the laser on"""
        if not self.turret_connected:
            return
        
        self.turret_client.laser_on()
    
    def laser_off(self):
        """Turn the laser off"""
        if not self.turret_connected:
            return
        
        self.turret_client.laser_off()
    
    def on_camera_error(self, error_message):
        """Handle camera errors"""
        print(f"Camera error: {error_message}")
        self.camera_status.set(f"Camera: Error")
    
    def on_turret_error(self, error_message):
        """Handle turret errors"""
        print(f"Turret error: {error_message}")
        self.turret_status.set(f"Turret: Error")
    
    def on_frame_received(self, frame):
        """Handle new frames from the camera"""
        # This is called in the camera stream thread
        # We don't need to do anything here as the main loop already pulls frames
        pass
    
    def on_turret_response(self, response):
        """Handle responses from the turret"""
        # This is called in the turret response thread
        print(f"Turret response: {response}")
    
    def start_wall_detection(self):
        """Start the wall detection process"""
        if not self.camera_connected or not self.turret_connected:
            messagebox.showerror("Error", "Camera and turret must be connected first")
            return
        
        if self.detection_running:
            return
        
        # Update UI state
        self.detection_running = True
        self.start_detection_btn.config(state=tk.DISABLED)
        self.cancel_detection_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.detection_status.set("Initializing wall detection...")
        
        # Get detection parameters
        grid_size = self.grid_size.get()
        scan_speed = self.scan_speed.get()
        
        # Initialize wall map data
        self.wall_map = {
            'resolution': grid_size,
            'hit_matrix': np.zeros((grid_size, grid_size)),
            'hits': [],
            'misses': []
        }
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._wall_detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def _wall_detection_worker(self):
        """Worker thread for wall detection"""
        try:
            # Get detection parameters
            grid_size = self.grid_size.get()
            scan_speed = self.scan_speed.get()
            
            # Calculate delay between steps based on speed (0.1 to 0.9 seconds)
            step_delay = (1.0 - scan_speed) * 0.5 + 0.1  # Scale for reasonable timing
            
            # Create the scanning grid with limited range
            yaw_values = np.linspace(-1.0, 0.0, grid_size)  # Yaw from -1 to 0
            pitch_values = np.linspace(0.0, 1.0, grid_size)  # Pitch from 0 to 1
            
            # Total number of points to scan
            total_points = grid_size * grid_size
            current_point = 0
            
            # Begin scanning in a serpentine pattern (alternating direction for each row)
            for y_idx, pitch in enumerate(pitch_values):
                # Check if detection was canceled
                if not self.detection_running:
                    return
                
                # Alternate scanning direction for each row (serpentine pattern)
                if y_idx % 2 == 0:
                    x_range = range(grid_size)  # Left to right
                else:
                    x_range = range(grid_size-1, -1, -1)  # Right to left
                
                for x_idx in x_range:
                    # Check if detection was canceled
                    if not self.detection_running:
                        return
                    
                    # Get yaw value based on the scanning direction
                    yaw = yaw_values[x_idx if y_idx % 2 == 0 else grid_size - 1 - x_idx]
                    
                    # Update progress
                    current_point += 1
                    progress = (current_point / total_points) * 100
                    self._update_detection_ui(progress, f"Scanning {current_point}/{total_points}: ({yaw:.2f}, {pitch:.2f})")
                    
                    # Step 1: Move turret to position
                    self.rotate_turret(yaw, pitch)
                    time.sleep(step_delay)  # Wait for movement to complete
                    
                    # Step 2: Turn on laser
                    self.laser_on()
                    time.sleep(step_delay)  # Wait for laser to stabilize
                    
                    # Step 3: Check for laser dot (multiple attempts for reliability)
                    hit_detected = False
                    max_attempts = 3
                    
                    for _ in range(max_attempts):
                        frame = self.camera_client.get_frame()
                        if frame is not None:
                            laser_pos = self.calibration.detect_laser_dot(frame)
                            if laser_pos:
                                hit_detected = True
                                break
                        time.sleep(0.1)
                    
                    # Step 4: Turn off laser before moving
                    self.laser_off()
                    time.sleep(step_delay * 0.5)  # Shorter delay before next move
                    
                    # Record result in the wall map
                    x_map_idx = x_idx if y_idx % 2 == 0 else grid_size - 1 - x_idx
                    self.wall_map['hit_matrix'][y_idx, x_map_idx] = 1 if hit_detected else 0
                    
                    if hit_detected:
                        self.wall_map['hits'].append((yaw, pitch))
                    else:
                        self.wall_map['misses'].append((yaw, pitch))
            
            # Detection complete, save results
            self._finalize_detection()
            
        except Exception as e:
            print(f"Error in wall detection: {str(e)}")
            self._update_detection_ui(100, f"Error: {str(e)}")
            self.detection_running = False
    
    def _update_detection_ui(self, progress, status):
        """Update the detection UI elements (thread-safe)"""
        self.root.after(0, lambda: self._do_update_detection_ui(progress, status))
    
    def _do_update_detection_ui(self, progress, status):
        """Update UI from the main thread"""
        self.progress_var.set(progress)
        self.detection_status.set(status)
    
    def _finalize_detection(self):
        """Finalize the wall detection process"""
        self.root.after(0, self._do_finalize_detection)
    
    def _do_finalize_detection(self):
        """Finalize detection from the main thread"""
        # Update UI
        self.detection_running = False
        self.start_detection_btn.config(state=tk.NORMAL)
        self.cancel_detection_btn.config(state=tk.DISABLED)
        self.detection_status.set("Wall detection complete!")
        
        # Create directory for results if needed
        os.makedirs("detection_data", exist_ok=True)
        
        # Save wall map to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"detection_data/wall_map_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                save_data = {
                    'resolution': self.wall_map['resolution'],
                    'hit_matrix': self.wall_map['hit_matrix'].tolist(),
                    'hits': self.wall_map['hits'],
                    'misses': self.wall_map['misses'],
                    'timestamp': timestamp
                }
                json.dump(save_data, f, indent=2)
            
            messagebox.showinfo("Success", f"Wall map saved to {filename}")
            
            # Turn on wall overlay to show results
            self.show_wall_overlay.set(True)
            
            # Show a simple visualization of the results
            self._show_wall_map_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def _show_wall_map_results(self):
        """Show a simple visualization of the wall map"""
        try:
            # Create a new window for the results
            results_window = tk.Toplevel(self.root)
            results_window.title("Wall Detection Results")
            results_window.geometry("400x400")
            
            # Create a canvas to draw the wall map
            canvas = tk.Canvas(results_window, bg="white")
            canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Draw the wall map
            grid_size = self.wall_map['resolution']
            hit_matrix = self.wall_map['hit_matrix']
            
            # Calculate cell size
            canvas_width = canvas.winfo_width() or 400
            canvas_height = canvas.winfo_height() or 400
            margin = 40
            cell_width = (canvas_width - 2 * margin) / grid_size
            cell_height = (canvas_height - 2 * margin) / grid_size
            
            # Wait for the canvas to be drawn
            self.root.update()
            
            # Draw grid
            for y in range(grid_size):
                for x in range(grid_size):
                    # Calculate cell coordinates
                    x1 = margin + x * cell_width
                    y1 = margin + y * cell_height
                    x2 = x1 + cell_width
                    y2 = y1 + cell_height
                    
                    # Draw cell with color based on hit/miss
                    color = "lightgreen" if hit_matrix[y, x] == 1 else "pink"
                    canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
            
            # Draw axes labels
            canvas.create_line(margin, canvas_height - margin, canvas_width - margin, canvas_height - margin, arrow=tk.LAST)
            canvas.create_text(canvas_width - margin + 10, canvas_height - margin, text="Yaw (-1 to 0)", anchor=tk.W)
            
            canvas.create_line(margin, canvas_height - margin, margin, margin, arrow=tk.LAST)
            canvas.create_text(margin, margin - 10, text="Pitch (0 to 1)", anchor=tk.S)
            
            # Draw legend
            canvas.create_rectangle(canvas_width - 100, margin, canvas_width - 80, margin + 20, fill="lightgreen", outline="gray")
            canvas.create_text(canvas_width - 60, margin + 10, text="Wall", anchor=tk.W)
            
            canvas.create_rectangle(canvas_width - 100, margin + 30, canvas_width - 80, margin + 50, fill="pink", outline="gray")
            canvas.create_text(canvas_width - 60, margin + 40, text="No Wall", anchor=tk.W)
            
        except Exception as e:
            print(f"Error showing results: {str(e)}")
    
    def cancel_wall_detection(self):
        """Cancel the wall detection process"""
        if self.detection_running:
            self.detection_running = False
            self.detection_status.set("Cancelling detection...")
            self.cancel_detection_btn.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window closing"""
        if self.detection_running:
            if messagebox.askyesno("Quit", "Wall detection is in progress. Are you sure you want to quit?"):
                self.detection_running = False
                time.sleep(0.5)  # Give time for threads to clean up
                self.running = False
                self.disconnect()
                self.root.destroy()
        else:
            self.running = False
            self.disconnect()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TurretControlUI(root)
    root.mainloop()