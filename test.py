import cv2
import numpy as np
import sys
import time
import socket
import threading
from enum import Enum


class TurretClient:
    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.receive_thread = None
        self.lock = threading.Lock()
        self.response_callback = None
        self.error_callback = None
        # Track current position
        self.current_yaw = 0
        self.current_pitch = 0
       
    def connect(self):
        """Establish connection to the turret controller"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
           
            # Start receive thread
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
           
            return True
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Connection error: {str(e)}")
            return False
   
    def disconnect(self):
        """Disconnect from the turret controller"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
   
    def _receive_loop(self):
        """Thread function to receive responses from the server"""
        buffer = ""
       
        while self.connected:
            try:
                data = self.socket.recv(1024).decode('ascii')
                if not data:
                    # Connection closed
                    self.connected = False
                    if self.error_callback:
                        self.error_callback("Connection closed by server")
                    break
               
                buffer += data
               
                # Process complete messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if self.response_callback:
                        self.response_callback(line)
            except Exception as e:
                if self.connected and self.error_callback:
                    self.error_callback(f"Receive error: {str(e)}")
                self.connected = False
                break
   
    def send_command(self, command):
        """Send a command to the turret controller"""
        if not self.connected:
            if self.error_callback:
                self.error_callback("Not connected")
            return False
       
        try:
            with self.lock:
                self.socket.sendall(f"{command}\n".encode('ascii'))
            return True
        except Exception as e:
            self.connected = False
            if self.error_callback:
                self.error_callback(f"Send error: {str(e)}")
            return False
   
    def laser_on(self):
        """Turn on the laser"""
        return self.send_command("LASER:ON")
   
    def laser_off(self):
        """Turn off the laser"""
        return self.send_command("LASER:OFF")
   
    def rotate(self, yaw, pitch):
        """Rotate the turret"""
        success = self.send_command(f"ROTATE:{yaw},{pitch}")
        if success:
            self.current_yaw = yaw
            self.current_pitch = pitch
        return success
   
    def ping(self):
        """Ping the server to check if it's alive"""
        return self.send_command("PING")


class ViewMode(Enum):
    ORIGINAL = "Camera Feed"
    DETECTION = "Detection View"
    ANALYSIS = "Analysis View"
    SETTINGS = "Settings"


class LaserDetectionDebugger:
    def __init__(self, camera_url="http://127.0.0.1:8080/camera", turret_host="127.0.0.1", turret_port=8888):
        self.camera_url = camera_url
        self.cap = None
        
        # Turret control
        self.turret = TurretClient(turret_host, turret_port)
        self.laser_on = False
        self.turret_step = 0.005
        
        # Detection parameters
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])
        self.min_brightness = 150
        self.min_area = 10
        self.max_area = 5000
        
        # UI state
        self.current_view = ViewMode.ORIGINAL
        self.selected_parameter = None
        self.show_help = False
        
        # Colors for modern UI
        self.colors = {
            'bg': (20, 20, 20),
            'panel': (35, 35, 35),
            'accent': (0, 120, 215),
            'success': (0, 200, 100),
            'danger': (200, 50, 50),
            'text': (255, 255, 255),
            'text_dim': (150, 150, 150),
            'border': (60, 60, 60)
        }
        
        # Window properties
        self.window_name = 'Laser Detection Debugger'
        self.window_width = 1280
        self.window_height = 720
    
    def connect_camera(self):
        """Connect to the camera"""
        print(f"Connecting to camera at {self.camera_url}")
        self.cap = cv2.VideoCapture(self.camera_url)
        
        if not self.cap.isOpened():
            print("Failed to open camera")
            return False
        
        print("Camera connected successfully")
        return True
    
    def connect_turret(self):
        """Connect to the turret controller"""
        print("Connecting to turret controller...")
        # Set up callbacks
        self.turret.response_callback = self.handle_turret_response
        self.turret.error_callback = self.handle_turret_error
        
        if self.turret.connect():
            print("Turret connected successfully")
            return True
        else:
            print("Failed to connect to turret controller")
            return False
    
    def handle_turret_response(self, response):
        """Handle responses from turret"""
        print(f"Turret response: {response}")
    
    def handle_turret_error(self, error):
        """Handle turret errors"""
        print(f"Turret error: {error}")
    
    def toggle_laser(self):
        """Toggle laser on/off"""
        if not self.turret.connected:
            return
        
        if self.laser_on:
            if self.turret.laser_off():
                self.laser_on = False
        else:
            if self.turret.laser_on():
                self.laser_on = True
    
    def move_turret(self, delta_yaw, delta_pitch):
        """Move turret relative to current position"""
        if not self.turret.connected:
            return
        
        new_yaw = self.turret.current_yaw + delta_yaw
        new_pitch = self.turret.current_pitch + delta_pitch
        
        # Clamp values
        new_yaw = max(-1.0, min(0.0, new_yaw))
        new_pitch = max(0.0, min(0.5, new_pitch))
        
        self.turret.rotate(new_yaw, new_pitch)
    
    def detect_laser(self, frame):
        """Detect laser spot in frame"""
        if frame is None:
            return None, {}
        
        debug_info = {}
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for red
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = mask1 + mask2
        
        # Red dominance in BGR
        b, g, r = cv2.split(frame)
        red_dominant = (r > g * 1.5) & (r > b * 1.5) & (r > 100)
        red_dominant_mask = red_dominant.astype(np.uint8) * 255
        
        # Combine masks
        combined_mask = cv2.bitwise_or(red_mask, red_dominant_mask)
        
        # Apply morphology
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                valid_contours.append(contour)
        
        # Find laser position
        laser_pos = None
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                laser_pos = (cx, cy)
        
        # Store debug info
        debug_info['red_mask'] = red_mask
        debug_info['red_dominant'] = red_dominant_mask
        debug_info['final_mask'] = final_mask
        debug_info['contours'] = contours
        debug_info['valid_contours'] = valid_contours
        
        return laser_pos, debug_info
    
    def draw_modern_button(self, img, x, y, w, h, text, active=False, enabled=True):
        """Draw a modern-style button"""
        color = self.colors['accent'] if active else self.colors['panel']
        if not enabled:
            color = self.colors['border']
        
        # Button background
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        
        # Button border
        cv2.rectangle(img, (x, y), (x+w, y+h), self.colors['border'], 1)
        
        # Button text
        text_color = self.colors['text'] if enabled else self.colors['text_dim']
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    def draw_modern_panel(self, img, x, y, w, h, title=""):
        """Draw a modern panel with title"""
        # Panel background
        cv2.rectangle(img, (x, y), (x+w, y+h), self.colors['panel'], -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), self.colors['border'], 1)
        
        if title:
            # Title bar
            cv2.rectangle(img, (x, y), (x+w, y+30), self.colors['bg'], -1)
            cv2.putText(img, title, (x+10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            cv2.line(img, (x, y+30), (x+w, y+30), self.colors['border'], 1)
    
    def draw_status_bar(self, img, height):
        """Draw status bar at bottom"""
        h, w = img.shape[:2]
        y = h - height
        
        # Background
        cv2.rectangle(img, (0, y), (w, h), self.colors['bg'], -1)
        cv2.line(img, (0, y), (w, y), self.colors['border'], 1)
        
        # Status items
        x = 10
        y_text = y + 20
        
        # Connection status
        camera_color = self.colors['success'] if self.cap and self.cap.isOpened() else self.colors['danger']
        turret_color = self.colors['success'] if self.turret.connected else self.colors['danger']
        
        cv2.putText(img, "Camera:", (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.circle(img, (x+70, y_text-5), 5, camera_color, -1)
        
        cv2.putText(img, "Turret:", (x+100, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.circle(img, (x+160, y_text-5), 5, turret_color, -1)
        
        # Turret position
        cv2.putText(img, f"Yaw: {self.turret.current_yaw:.3f}", (x+200, y_text), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.putText(img, f"Pitch: {self.turret.current_pitch:.3f}", (x+320, y_text), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Current mode
        cv2.putText(img, f"Mode: {self.current_view.value}", (x+450, y_text), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def draw_main_view(self, frame, laser_pos):
        """Draw the main camera view"""
        h, w = frame.shape[:2]
        display = np.zeros((h, w, 3), dtype=np.uint8)
        display[:] = self.colors['bg']
        
        # Main panel
        panel_margin = 20
        panel_x, panel_y = panel_margin, panel_margin
        panel_w, panel_h = w - panel_margin*2, h - panel_margin*2 - 50  # Leave room for status bar
        
        self.draw_modern_panel(display, panel_x, panel_y, panel_w, panel_h, "Live Camera Feed")
        
        # Camera feed
        feed_margin = 40
        feed_x, feed_y = panel_x + feed_margin, panel_y + feed_margin
        feed_w, feed_h = panel_w - feed_margin*2, panel_h - feed_margin*2
        
        if frame is not None:
            # Resize frame to fit
            scale = min(feed_w/frame.shape[1], feed_h/frame.shape[0])
            new_w, new_h = int(frame.shape[1]*scale), int(frame.shape[0]*scale)
            resized = cv2.resize(frame, (new_w, new_h))
            
            # Center in panel
            x_offset = feed_x + (feed_w - new_w) // 2
            y_offset = feed_y + (feed_h - new_h) // 2
            
            display[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # Draw laser detection
            if laser_pos:
                scaled_pos = (int(x_offset + laser_pos[0]*scale), int(y_offset + laser_pos[1]*scale))
                cv2.circle(display, scaled_pos, 10, self.colors['success'], 2)
                cv2.putText(display, "Laser Detected", (scaled_pos[0]+15, scaled_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['success'], 1)
        
        # Control panel on right
        ctrl_w = 250
        ctrl_x = w - ctrl_w - panel_margin
        ctrl_y = panel_margin
        ctrl_h = 300
        
        self.draw_modern_panel(display, ctrl_x, ctrl_y, ctrl_w, ctrl_h, "Controls")
        
        # Control buttons
        btn_w, btn_h = 100, 35
        btn_x = ctrl_x + (ctrl_w - btn_w) // 2
        btn_y = ctrl_y + 50
        btn_spacing = 45
        
        # Laser toggle
        self.draw_modern_button(display, btn_x, btn_y, btn_w, btn_h, 
                               "Laser ON" if self.laser_on else "Laser OFF",
                               active=self.laser_on, enabled=self.turret.connected)
        
        # View buttons
        btn_y += btn_spacing
        self.draw_modern_button(display, btn_x, btn_y, btn_w, btn_h, "Camera", 
                               active=self.current_view==ViewMode.ORIGINAL)
        
        btn_y += btn_spacing
        self.draw_modern_button(display, btn_x, btn_y, btn_w, btn_h, "Detection", 
                               active=self.current_view==ViewMode.DETECTION)
        
        btn_y += btn_spacing
        self.draw_modern_button(display, btn_x, btn_y, btn_w, btn_h, "Analysis", 
                               active=self.current_view==ViewMode.ANALYSIS)
        
        btn_y += btn_spacing
        self.draw_modern_button(display, btn_x, btn_y, btn_w, btn_h, "Settings", 
                               active=self.current_view==ViewMode.SETTINGS)
        
        # Help text
        help_y = h - 80
        help_text = "WASD: Move | L: Laser | TAB: Switch Views | F1: Help"
        cv2.putText(display, help_text, (20, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_dim'], 1)
        
        return display
    
    def draw_detection_view(self, frame, debug_info):
        """Draw detection analysis view"""
        h, w = frame.shape[:2]
        display = np.zeros((h, w, 3), dtype=np.uint8)
        display[:] = self.colors['bg']
        
        # Title
        cv2.putText(display, "Detection Analysis", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        
        # Grid layout
        grid_x, grid_y = 20, 80
        cell_w, cell_h = (w - 40) // 3, (h - 180) // 2
        
        views = [
            ("Original", frame),
            ("Red Mask", debug_info.get('red_mask')),
            ("Red Dominant", debug_info.get('red_dominant')),
            ("Combined", debug_info.get('final_mask')),
            ("Final", debug_info.get('final_mask')),
            ("Result", frame.copy() if frame is not None else None)
        ]
        
        for i, (title, img) in enumerate(views):
            if img is None:
                continue
            
            row, col = i // 3, i % 3
            x = grid_x + col * cell_w
            y = grid_y + row * cell_h
            
            # Panel
            self.draw_modern_panel(display, x+5, y+5, cell_w-10, cell_h-10, title)
            
            # Image
            img_display = img
            if len(img.shape) == 2:
                img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            scale = min((cell_w-20)/img_display.shape[1], (cell_h-50)/img_display.shape[0])
            new_w, new_h = int(img_display.shape[1]*scale), int(img_display.shape[0]*scale)
            resized = cv2.resize(img_display, (new_w, new_h))
            
            img_x = x + (cell_w - new_w) // 2
            img_y = y + 35 + (cell_h - 35 - new_h) // 2
            
            display[img_y:img_y+new_h, img_x:img_x+new_w] = resized
        
        return display
    
    def draw_settings_view(self):
        """Draw settings panel"""
        h, w = self.window_height, self.window_width
        display = np.zeros((h, w, 3), dtype=np.uint8)
        display[:] = self.colors['bg']
        
        # Title
        cv2.putText(display, "Settings", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        
        # Parameters panel
        panel_x, panel_y = 20, 80
        panel_w, panel_h = w - 40, h - 180
        self.draw_modern_panel(display, panel_x, panel_y, panel_w, panel_h, "Detection Parameters")
        
        # Parameter list
        params = [
            ("Hue Range", f"[{self.lower_red1[0]}-{self.upper_red1[0]}]"),
            ("Saturation Range", f"[{self.lower_red1[1]}-{self.upper_red1[1]}]"),
            ("Value Range", f"[{self.lower_red1[2]}-{self.upper_red1[2]}]"),
            ("Brightness", str(self.min_brightness)),
            ("Min Area", str(self.min_area)),
            ("Max Area", str(self.max_area)),
            ("Turret Step", f"{self.turret_step:.3f}")
        ]
        
        y = panel_y + 50
        for i, (name, value) in enumerate(params):
            x = panel_x + 30
            
            # Highlight selected parameter
            if i == self.selected_parameter:
                cv2.rectangle(display, (x-5, y-20), (x+panel_w-50, y+10), self.colors['accent'], -1)
            
            cv2.putText(display, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            cv2.putText(display, value, (x+200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            
            y += 40
        
        # Instructions
        inst_y = h - 100
        cv2.putText(display, "↑↓: Select | ←→: Adjust | Enter: Apply | ESC: Cancel", 
                   (20, inst_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_dim'], 1)
        
        return display
    
    def run(self):
        """Main loop"""
        if not self.connect_camera():
            return
        
        self.connect_turret()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Detect laser
            laser_pos, debug_info = self.detect_laser(frame)
            
            # Draw current view
            if self.current_view == ViewMode.ORIGINAL:
                display = self.draw_main_view(frame, laser_pos)
            elif self.current_view == ViewMode.DETECTION:
                display = self.draw_detection_view(frame, debug_info)
            elif self.current_view == ViewMode.SETTINGS:
                display = self.draw_settings_view()
            else:
                display = self.draw_main_view(frame, laser_pos)
            
            # Draw status bar
            self.draw_status_bar(display, 50)
            
            cv2.imshow(self.window_name, display)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('l'):
                self.toggle_laser()
            elif key == ord('w'):
                self.move_turret(0, self.turret_step)
            elif key == ord('s'):
                self.move_turret(0, -self.turret_step)
            elif key == ord('a'):
                self.move_turret(-self.turret_step, 0)
            elif key == ord('d'):
                self.move_turret(self.turret_step, 0)
            elif key == 9:  # TAB
                # Cycle through views
                modes = list(ViewMode)
                current_idx = modes.index(self.current_view)
                self.current_view = modes[(current_idx + 1) % len(modes)]
            elif key == 27:  # ESC
                if self.current_view == ViewMode.SETTINGS:
                    self.current_view = ViewMode.ORIGINAL
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        if self.turret.connected:
            if self.laser_on:
                self.turret.laser_off()
            self.turret.disconnect()


def main():
    camera_url = "http://127.0.0.1:8080/camera"
    turret_host = "127.0.0.1"
    turret_port = 8888
    
    if len(sys.argv) > 1:
        camera_url = sys.argv[1]
    if len(sys.argv) > 2:
        turret_host = sys.argv[2]
    if len(sys.argv) > 3:
        turret_port = int(sys.argv[3])
    
    debugger = LaserDetectionDebugger(camera_url, turret_host, turret_port)
    debugger.run()


if __name__ == "__main__":
    main()