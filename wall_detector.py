import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
import os
import json

class WallDetector:
    def __init__(self, turret_ui):
        """
        Initialize the wall detector with a reference to the turret UI
        
        Args:
            turret_ui: Instance of TurretControlUI
        """
        self.turret_ui = turret_ui
        self.detection_running = False
        self.detection_canceled = False
        
    def detect_walls(self):
        """Launch the wall detection UI and process"""
        if not self.turret_ui.camera_connected or not self.turret_ui.turret_connected:
            messagebox.showerror("Error", "Camera and turret must be connected first")
            return
            
        # Create detection window
        detection_window = tk.Toplevel(self.turret_ui.root)
        detection_window.title("Wall Boundary Detection")
        detection_window.geometry("700x600")
        detection_window.protocol("WM_DELETE_WINDOW", lambda: self._cancel_detection(detection_window))
        
        # Setup UI elements
        self._create_detection_ui(detection_window)
        
    def _create_detection_ui(self, window):
        """Create the UI for wall detection"""
        # Control panel frame
        control_frame = ttk.Frame(window)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Grid resolution setting
        ttk.Label(control_frame, text="Grid Resolution:").grid(row=0, column=0, padx=5, pady=5)
        resolution_var = tk.IntVar(value=6)
        resolution_combo = ttk.Combobox(control_frame, textvariable=resolution_var, 
                                       values=[4, 5, 6, 7, 8, 9, 10], width=5)
        resolution_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Scan speed setting
        ttk.Label(control_frame, text="Scan Speed:").grid(row=0, column=2, padx=5, pady=5)
        speed_var = tk.DoubleVar(value=0.5)
        speed_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, variable=speed_var, 
                              orient=tk.HORIZONTAL, length=100)
        speed_scale.grid(row=0, column=3, padx=5, pady=5)
        speed_label = ttk.Label(control_frame, text="0.5")
        speed_label.grid(row=0, column=4, padx=5, pady=5)
        
        # Update speed label when scale changes
        def update_speed_label(*args):
            speed_label.config(text=f"{speed_var.get():.1f}")
        speed_var.trace("w", update_speed_label)
        
        # Status frame
        status_frame = ttk.Frame(window)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        status_var = tk.StringVar(value="Ready to start scanning")
        status_label = ttk.Label(status_frame, textvariable=status_var)
        status_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(status_frame, variable=progress_var, 
                                     length=300, mode='determinate')
        progress_bar.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Main canvas for visualization
        canvas_frame = ttk.LabelFrame(window, text="Wall Detection Map")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(canvas_frame, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Start button
        start_button = ttk.Button(button_frame, text="Start Detection", 
                                command=lambda: self._start_detection(
                                    window, canvas, resolution_var, speed_var, 
                                    status_var, progress_var))
        start_button.pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        cancel_button = ttk.Button(button_frame, text="Cancel", 
                                 command=lambda: self._cancel_detection(window))
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Save button (initially disabled)
        self.save_button = ttk.Button(button_frame, text="Save Results", state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # Store UI elements for later access
        self.detection_window = window
        self.detection_canvas = canvas
        self.status_var = status_var
        self.progress_var = progress_var
        
    def _start_detection(self, window, canvas, resolution_var, speed_var, status_var, progress_var):
        """Start the wall detection process"""
        if self.detection_running:
            return
            
        self.detection_running = True
        self.detection_canceled = False
        
        # Clear canvas
        canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Use default dimensions if canvas not properly sized yet
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 500
            canvas_height = 400
            
        # Setup coordinate axes
        canvas.create_line(50, canvas_height-50, canvas_width-50, canvas_height-50, 
                         arrow=tk.LAST, width=2)  # X-axis (yaw)
        canvas.create_line(50, canvas_height-50, 50, 50, 
                         arrow=tk.LAST, width=2)  # Y-axis (pitch)
        canvas.create_text(canvas_width-40, canvas_height-30, text="Yaw")
        canvas.create_text(30, 60, text="Pitch")
        
        # Get detection parameters
        resolution = resolution_var.get()
        speed = speed_var.get()
        
        # Calculate grid cell size
        cell_width = (canvas_width - 100) / resolution
        cell_height = (canvas_height - 100) / resolution
        
        # Draw grid lines
        for i in range(resolution + 1):
            # Vertical grid lines
            canvas.create_line(
                50 + i * cell_width, canvas_height - 50,
                50 + i * cell_width, 50,
                fill="lightgray"
            )
            
            # Horizontal grid lines
            canvas.create_line(
                50, canvas_height - 50 - i * cell_height,
                canvas_width - 50, canvas_height - 50 - i * cell_height,
                fill="lightgray"
            )
        
        # Create the scanning grid
        yaw_values = np.linspace(-1.0, 1.0, resolution)
        pitch_values = np.linspace(-1.0, 1.0, resolution)
        
        # Initialize results matrix
        self.hit_matrix = np.zeros((resolution, resolution))
        self.hits = []
        self.misses = []
        
        # Store detection parameters
        self.detection_params = {
            'resolution': resolution,
            'yaw_values': yaw_values,
            'pitch_values': pitch_values,
            'canvas_width': canvas_width,
            'canvas_height': canvas_height,
            'cell_width': cell_width,
            'cell_height': cell_height,
            'speed': speed
        }
        
        # Launch detection thread
        detection_thread = threading.Thread(target=self._detection_worker)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Enable save button and configure its command
        self.save_button.config(
            command=lambda: self._save_detection_results(),
            state=tk.DISABLED  # Will be enabled when detection completes
        )
        
    def _detection_worker(self):
        """Worker thread to perform the wall detection"""
        try:
            # Get detection parameters
            resolution = self.detection_params['resolution']
            yaw_values = self.detection_params['yaw_values']
            pitch_values = self.detection_params['pitch_values']
            speed = self.detection_params['speed']
            
            # Calculate delay between steps based on speed (0.1 to 0.9 seconds)
            step_delay = 1.0 - speed
            
            # Total number of points to scan
            total_points = resolution * resolution
            current_point = 0
            
            # Begin scanning
            for y_idx, pitch in enumerate(pitch_values):
                # Alternate scanning direction for each row (serpentine pattern)
                # This minimizes turret movement distance
                if y_idx % 2 == 0:
                    x_range = range(resolution)  # Left to right
                else:
                    x_range = range(resolution-1, -1, -1)  # Right to left
                
                for x_idx in x_range:
                    # Check if detection was canceled
                    if self.detection_canceled:
                        self.detection_running = False
                        self._update_ui(100, "Detection canceled")
                        return
                    
                    # Get the yaw value (adjusting for scan direction)
                    x_val = x_idx if y_idx % 2 == 0 else resolution - 1 - x_idx
                    yaw = yaw_values[x_val]
                    
                    # Update progress
                    current_point += 1
                    progress = (current_point / total_points) * 100
                    self._update_ui(progress, f"Scanning position {current_point}/{total_points}: ({yaw:.2f}, {pitch:.2f})")
                    
                    # Step 1: Move turret to position
                    self.turret_ui.rotate_turret(yaw, pitch)
                    time.sleep(step_delay)  # Wait for movement to complete
                    
                    # Step 2: Turn on laser
                    self.turret_ui.turret_client.laser_on()
                    time.sleep(step_delay)  # Wait for laser to stabilize
                    
                    # Step 3: Check for laser dot (multiple attempts for reliability)
                    hit_detected = False
                    max_attempts = 3
                    
                    for _ in range(max_attempts):
                        frame = self.turret_ui.camera_client.get_frame()
                        if frame is not None:
                            laser_pos = self.turret_ui.calibration.detect_laser_dot(frame)
                            if laser_pos:
                                hit_detected = True
                                break
                        time.sleep(0.1)
                    
                    # Step 4: Turn off laser before moving
                    self.turret_ui.turret_client.laser_off()
                    time.sleep(0.2)  # Short delay before next move
                    
                    # Record result
                    self.hit_matrix[y_idx, x_val] = 1 if hit_detected else 0
                    
                    if hit_detected:
                        self.hits.append((yaw, pitch))
                    else:
                        self.misses.append((yaw, pitch))
                    
                    # Draw result on canvas
                    self._draw_detection_point(x_val, y_idx, hit_detected)
            
            # Detection complete
            self._finalize_detection()
            
        except Exception as e:
            print(f"Error in wall detection: {str(e)}")
            self._update_ui(100, f"Error: {str(e)}")
            self.detection_running = False
    
    def _update_ui(self, progress, status):
        """Update the UI with progress and status (thread-safe)"""
        self.detection_window.after(0, lambda: self._do_update_ui(progress, status))
    
    def _do_update_ui(self, progress, status):
        """Update UI elements (called from main thread)"""
        self.progress_var.set(progress)
        self.status_var.set(status)
    
    def _draw_detection_point(self, x_idx, y_idx, is_hit):
        """Draw a detection point on the canvas (thread-safe)"""
        self.detection_window.after(0, lambda: self._do_draw_point(x_idx, y_idx, is_hit))
    
    def _do_draw_point(self, x_idx, y_idx, is_hit):
        """Draw detection point (called from main thread)"""
        cell_width = self.detection_params['cell_width']
        cell_height = self.detection_params['cell_height']
        canvas_height = self.detection_params['canvas_height']
        
        # Calculate position
        x1 = 50 + x_idx * cell_width
        y1 = canvas_height - 50 - (y_idx + 1) * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        
        # Draw rectangle with color based on hit/miss
        color = "lightgreen" if is_hit else "pink"
        self.detection_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
    
    def _finalize_detection(self):
        """Finalize the detection process"""
        self.detection_window.after(0, self._do_finalize_detection)
    
    def _do_finalize_detection(self):
        """Finalize detection (called from main thread)"""
        self.detection_running = False
        self.status_var.set("Detection complete!")
        
        # Enable save button
        self.save_button.config(state=tk.NORMAL)
        
        # Draw boundary
        self._draw_boundary()
    
    def _draw_boundary(self):
        """Draw the wall boundary on the canvas"""
        resolution = self.detection_params['resolution']
        cell_width = self.detection_params['cell_width']
        cell_height = self.detection_params['cell_height']
        canvas_height = self.detection_params['canvas_height']
        canvas_width = self.detection_params['canvas_width']
        
        # Find boundary cells (where state changes from hit to miss)
        boundary_cells = []
        for y in range(resolution):
            for x in range(resolution):
                # Check if this cell has neighboring cells with different values
                is_boundary = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < resolution and 0 <= ny < resolution and 
                            self.hit_matrix[y, x] != self.hit_matrix[ny, nx]):
                            is_boundary = True
                            break
                    if is_boundary:
                        break
                
                if is_boundary:
                    boundary_cells.append((x, y))
        
        # Draw boundary cells with blue outline
        for x, y in boundary_cells:
            x1 = 50 + x * cell_width
            y1 = canvas_height - 50 - (y + 1) * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            
            # Draw blue border around boundary cells
            self.detection_canvas.create_rectangle(x1, y1, x2, y2, fill="", outline="blue", width=2)
        
        # Add legend
        legend_x = canvas_width - 150
        legend_y = 20
        
        self.detection_canvas.create_rectangle(legend_x, legend_y, legend_x+20, legend_y+20, 
                                            fill="lightgreen", outline="gray")
        self.detection_canvas.create_text(legend_x+70, legend_y+10, text="Hit (Wall)")
        
        self.detection_canvas.create_rectangle(legend_x, legend_y+30, legend_x+20, legend_y+50, 
                                            fill="pink", outline="gray")
        self.detection_canvas.create_text(legend_x+70, legend_y+40, text="Miss (No Wall)")
        
        self.detection_canvas.create_rectangle(legend_x, legend_y+60, legend_x+20, legend_y+80, 
                                            fill="", outline="blue", width=2)
        self.detection_canvas.create_text(legend_x+70, legend_y+70, text="Boundary")
    
    def _save_detection_results(self):
        """Save the detection results to a file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs("detection_data", exist_ok=True)
            
            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"detection_data/wall_map_{timestamp}.json"
            
            # Prepare data for saving
            save_data = {
                'resolution': self.detection_params['resolution'],
                'yaw_values': self.detection_params['yaw_values'].tolist(),
                'pitch_values': self.detection_params['pitch_values'].tolist(),
                'hit_matrix': self.hit_matrix.tolist(),
                'hits': self.hits,
                'misses': self.misses,
                'timestamp': timestamp
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            messagebox.showinfo("Success", f"Wall map saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def _cancel_detection(self, window):
        """Cancel the detection process"""
        if self.detection_running:
            self.detection_canceled = True
            self.status_var.set("Cancelling...")
        else:
            window.destroy()


# Add this method to your TurretControlUI class
def detect_wall_boundaries(self):
    """Launch the wall boundary detection tool"""
    detector = WallDetector(self)
    detector.detect_walls()

# Add a button for wall detection to your TurretControlUI's __init__ method (after creating other UI elements)
# wall_detection_button = ttk.Button(self.control_frame, text="Detect Wall Boundaries", command=self.detect_wall_boundaries)
# wall_detection_button.pack(fill=tk.X, padx=5, pady=5)