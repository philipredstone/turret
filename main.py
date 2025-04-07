import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import urllib.request
import io
import numpy as np
import queue
import socket
from http.client import HTTPConnection
import re
import struct

class DirectStreamingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("High-Speed Camera Preview")
        self.root.geometry("1000x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize components
        self.stream_thread = None
        self.camera_connected = False
        self.stream_active = False
        self.preview_frame = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.frames_queue = queue.Queue(maxsize=2)  # Small queue to avoid memory buildup
        
        # Debug settings
        self.debug_level = 2  # 0=minimal, 1=normal, 2=verbose
        
        # Add a performance timer (to monitor actual app performance)
        self.ui_fps = 0
        self.ui_frame_count = 0
        self.ui_last_fps_time = time.time()
        
        # Setup UI
        self.init_ui()
        
        # Log initial message
        self.log("High-Speed Camera Application started v2.0 (Enhanced MJPEG Parser)")

    def init_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top section
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Connection controls
        left_panel = ttk.Frame(top_frame, padding=5, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)  # Prevent shrinking
        
        # Camera connection group
        camera_group = ttk.LabelFrame(left_panel, text="Camera Connection", padding=10)
        camera_group.pack(fill=tk.X, pady=5)
        
        ttk.Label(camera_group, text="Host:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.camera_host = ttk.Entry(camera_group)
        self.camera_host.insert(0, "127.0.0.1")
        self.camera_host.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(camera_group, text="Port:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.camera_port = ttk.Spinbox(camera_group, from_=1, to=65535, width=5)
        self.camera_port.insert(0, "8080")
        self.camera_port.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(camera_group, text="Path:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.camera_path = ttk.Entry(camera_group)
        self.camera_path.insert(0, "/video")
        self.camera_path.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Stream type selection
        ttk.Label(camera_group, text="Stream Type:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.stream_type = ttk.Combobox(camera_group, values=["MJPEG", "JPEG HTTP"])
        self.stream_type.current(0)  # Default to MJPEG
        self.stream_type.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Connection button
        self.connect_btn = ttk.Button(camera_group, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Debug options
        debug_frame = ttk.Frame(camera_group)
        debug_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.show_fps = tk.BooleanVar(value=True)
        ttk.Checkbutton(debug_frame, text="Show FPS", variable=self.show_fps).pack(anchor=tk.W)
        
        # Right panel - Camera preview
        right_panel = ttk.Frame(top_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.preview_canvas = tk.Canvas(right_panel, bg="#222222", width=640, height=480)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.create_text(320, 240, text="No Camera Feed", fill="white", font=("Arial", 14))
        
        # Bottom section - Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_area.pack(fill=tk.BOTH, expand=True)
        self.log_area.config(state=tk.DISABLED)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Start UI updater
        self.update_ui()

    def log(self, message):
        """Add message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_area.see(tk.END)
        self.log_area.config(state=tk.DISABLED)
    
    def update_ui(self):
        """Ultra-optimized UI update loop"""
        # Avoid heavy status checks too frequently
        now = time.time()
        if not hasattr(self, 'last_status_update'):
            self.last_status_update = now - 1  # Force initial update
            
        # Check if we're still connected (do this infrequently)
        if now - self.last_status_update > 0.5:  # Only check every half second
            if self.camera_connected and self.stream_thread and not self.stream_thread.is_alive():
                self.camera_connected = False
                self.status_var.set("Disconnected - Stream thread stopped")
                self.connect_btn.config(text="Connect")
                
            # Update status bar (expensive, do infrequently)
            if self.camera_connected:
                self.status_var.set(f"Connected | Camera: {self.fps:.1f} FPS | UI: {self.ui_fps:.1f} FPS")
            else:
                self.status_var.set("Disconnected")
                
            self.last_status_update = now
            
        # Get current frame from queue if available - this is the main task
        try:
            if not self.frames_queue.empty():
                frame = self.frames_queue.get_nowait()
                self._update_canvas(frame)
                self.frames_queue.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            if self.debug_level > 0 and self.ui_frame_count % 100 == 0:  # Very infrequent logging
                self.log(f"Error in UI update: {str(e)}")
        
        # Schedule next update at optimal rate
        # Use adaptive timing to control frame rate
        target_interval = 0.01  # Aim for ~100 FPS UI updates (10ms)
        
        # Schedule next update with adaptive timing
        self.root.after(int(target_interval * 1000), self.update_ui)
    
    def toggle_connection(self):
        """Connect to or disconnect from the camera"""
        if self.camera_connected:
            # Disconnect
            self.stream_active = False
            if self.stream_thread:
                # Attempt to join thread with timeout
                self.stream_thread.join(timeout=1.0)
                self.stream_thread = None
                
            self.camera_connected = False
            self.log("Disconnected from camera")
            
            # Clear preview
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(320, 240, text="No Camera Feed", fill="white", font=("Arial", 14))
            
            # Clear frame queue
            while not self.frames_queue.empty():
                try:
                    self.frames_queue.get_nowait()
                    self.frames_queue.task_done()
                except:
                    break
        else:
            # Connect
            host = self.camera_host.get()
            port = int(self.camera_port.get())
            path = self.camera_path.get()
            stream_type = self.stream_type.get()
            
            # Clear any existing frames
            while not self.frames_queue.empty():
                try:
                    self.frames_queue.get_nowait()
                    self.frames_queue.task_done()
                except:
                    break
            
            self.log(f"Connecting to {stream_type} stream at http://{host}:{port}{path}")
            
            try:
                # Set stream active flag
                self.stream_active = True
                
                # Increase queue size for higher throughput
                self.frames_queue = queue.Queue(maxsize=4)  # Slightly larger queue
                
                # Create and start stream thread
                if stream_type == "MJPEG":
                    self.stream_thread = threading.Thread(
                        target=self.mjpeg_stream_thread,
                        args=(host, port, path),
                        daemon=True
                    )
                else:  # JPEG HTTP
                    self.stream_thread = threading.Thread(
                        target=self.jpeg_http_stream_thread,
                        args=(host, port, path),
                        daemon=True
                    )
                
                # Set thread priority if possible
                try:
                    self.stream_thread.daemon = True
                except:
                    pass
                    
                # Start the thread
                self.stream_thread.start()
                self.camera_connected = True
                
                # Reset FPS counters
                self.fps = 0
                self.frame_count = 0
                self.last_fps_time = time.time()
                self.ui_fps = 0
                self.ui_frame_count = 0
                self.ui_last_fps_time = time.time()
                
            except Exception as e:
                self.log(f"Error connecting to camera: {str(e)}")
                messagebox.showerror("Connection Error", str(e))
                self.stream_active = False
    
    def mjpeg_stream_thread(self, host, port, path):
        """Thread for streaming MJPEG from HTTP camera - highly optimized for 30 FPS"""
        self.log("Starting optimized 30 FPS MJPEG stream...")
        
        try:
            # Construct URL
            url = f"http://{host}:{port}{path}"
            self.log(f"Opening stream from {url}")
            
            # Create custom opener with increased buffer sizes
            opener = urllib.request.build_opener()
            opener.addheaders = [('Accept', 'image/jpeg')]
            
            # Open the stream with a larger timeout for initial connection
            stream = opener.open(url, timeout=10)
            
            # Get content type to check for MJPEG
            content_type = stream.getheader('Content-Type')
            self.log(f"Stream content type: {content_type}")
            
            # Initialize variables
            self.frame_count = 0
            self.last_fps_time = time.time()
            buffer = b''
            
            # Read initial data - can help prime the buffer
            initial_chunk = stream.read(16384)  # Larger initial read
            buffer += initial_chunk
            self.log(f"Read initial {len(initial_chunk)} bytes")
            
            # Process frames as quickly as possible
            while self.stream_active:
                try:
                    # Read large chunks of data for efficiency
                    chunk = stream.read(32768)  # Larger chunks for better throughput
                    if not chunk:
                        self.log("End of stream reached")
                        break
                    
                    # Add to buffer
                    buffer += chunk
                    
                    # Process all complete frames in the buffer
                    while True:
                        # Look for JPEG start marker
                        jpeg_start = buffer.find(b'\xff\xd8')
                        if jpeg_start == -1:
                            break  # No start marker found
                        
                        # Look for JPEG end marker
                        jpeg_end = buffer.find(b'\xff\xd9', jpeg_start)
                        if jpeg_end == -1:
                            break  # No end marker found
                        
                        # Extract the JPEG data
                        jpeg_data = buffer[jpeg_start:jpeg_end+2]
                        
                        # Process the image if our queue isn't full
                        # This prevents building up a backlog if UI can't keep up
                        if not self.frames_queue.full():
                            try:
                                # Create PIL image from JPEG data
                                image = Image.open(io.BytesIO(jpeg_data))
                                
                                # Convert to PhotoImage directly
                                photo = ImageTk.PhotoImage(image=image)
                                
                                # Add to queue for UI thread
                                self.frames_queue.put(photo)
                                
                                # Update FPS counter
                                self.frame_count += 1
                                current_time = time.time()
                                time_diff = current_time - self.last_fps_time
                                
                                if time_diff >= 0.5:  # Update every half second
                                    self.fps = self.frame_count / time_diff
                                    self.frame_count = 0
                                    self.last_fps_time = current_time
                            except Exception as e:
                                if self.frame_count % 100 == 0:  # Very infrequent logging
                                    self.log(f"Error processing image: {str(e)}")
                        
                        # Remove processed data from buffer
                        buffer = buffer[jpeg_end+2:]
                    
                    # Memory safety: if buffer gets too large without finding frames,
                    # truncate it to prevent memory issues
                    if len(buffer) > 1000000:  # 1MB limit
                        # Keep a reasonable tail that might contain a partial frame
                        buffer = buffer[-100000:]
                    
                except Exception as e:
                    self.log(f"Error reading from stream: {str(e)}")
                    # Short pause on error to avoid tight loop, but keep it minimal
                    time.sleep(0.01)
            
            # Clean up
            stream.close()
            self.log("MJPEG stream stopped")
            
        except Exception as e:
            self.log(f"Stream error: {str(e)}")
            self.camera_connected = False
    
    def jpeg_http_stream_thread(self, host, port, path):
        """Thread for streaming individual JPEG images from HTTP server"""
        self.log("Starting JPEG HTTP stream...")
        
        try:
            # Get initial response to check content type
            url = f"http://{host}:{port}{path}"
            request = urllib.request.Request(url)
            
            with urllib.request.urlopen(request, timeout=5) as response:
                content_type = response.getheader('Content-Type')
                self.log(f"Content type from server: {content_type}")
                
                # If this is actually an MJPEG stream, switch to that
                if 'multipart/x-mixed-replace' in content_type:
                    self.log("Detected MJPEG stream - switching to MJPEG handler")
                    self.mjpeg_stream_thread(host, port, path)
                    return
            
            # Create a persistent HTTP connection
            conn = HTTPConnection(host, port, timeout=5.0)
            
            # Frame processing loop
            self.frame_count = 0
            self.last_fps_time = time.time()
            
            while self.stream_active:
                try:
                    # Request a single image
                    start_time = time.time()
                    conn.request("GET", path)
                    response = conn.getresponse()
                    
                    if response.status == 200:
                        # Read all data
                        data = response.read()
                        
                        # Look for JPEG markers in data
                        jpeg_start = data.find(b'\xff\xd8')
                        jpeg_end = data.find(b'\xff\xd9')
                        
                        if jpeg_start >= 0 and jpeg_end > jpeg_start:
                            # Extract JPEG data
                            jpeg_data = data[jpeg_start:jpeg_end+2]
                            
                            if len(jpeg_data) > 100:  # Make sure we have enough data for an image
                                # Create image
                                image = Image.open(io.BytesIO(jpeg_data))
                                
                                # Update FPS counter
                                self.frame_count += 1
                                current_time = time.time()
                                time_diff = current_time - self.last_fps_time
                                
                                if time_diff >= 0.5:
                                    self.fps = self.frame_count / time_diff
                                    self.frame_count = 0
                                    self.last_fps_time = current_time
                                
                                # Create photo image
                                photo = ImageTk.PhotoImage(image=image)
                                
                                # Add to queue for UI thread to process
                                if not self.frames_queue.full():
                                    self.frames_queue.put(photo)
                        else:
                            self.log(f"No JPEG markers found in data (length: {len(data)})")
                            # Debug first bytes
                            if len(data) > 10:
                                self.log(f"First bytes: {data[:20]}")
                    else:
                        self.log(f"HTTP error: {response.status} {response.reason}")
                        time.sleep(1.0)  # Wait before retry
                    
                    # Calculate time to wait to maintain target FPS
                    process_time = time.time() - start_time
                    wait_time = max(0.001, (1.0/30) - process_time)  # Target 30fps max
                    time.sleep(wait_time)
                
                except (ConnectionResetError, ConnectionAbortedError) as e:
                    self.log(f"Connection error: {str(e)}. Reconnecting...")
                    time.sleep(0.5)
                    # Recreate connection
                    try:
                        conn.close()
                    except:
                        pass
                    conn = HTTPConnection(host, port, timeout=5.0)
                    
                except Exception as e:
                    if self.frame_count % 10 == 0:  # Limit log messages
                        self.log(f"Error getting frame: {str(e)}")
                    time.sleep(0.5)  # Wait before retry
            
            # Clean up
            conn.close()
            self.log("JPEG HTTP stream stopped")
            
        except Exception as e:
            self.log(f"Stream error: {str(e)}")
            self.camera_connected = False
    
    def _update_canvas(self, photo):
        """Highly optimized canvas update for maximum performance"""
        try:
            # Store reference to prevent garbage collection
            self.preview_frame = photo
            
            # Get dimensions only once per session unless changed
            if not hasattr(self, 'canvas_dims'):
                self.canvas_dims = (self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height())
            
            # Check if canvas size has changed
            current_dims = (self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height())
            if current_dims != self.canvas_dims and current_dims[0] > 1 and current_dims[1] > 1:
                self.canvas_dims = current_dims
                # Clear canvas completely when dimensions change
                self.preview_canvas.delete("all")
                # Create image item once and store its ID
                self.image_id = self.preview_canvas.create_image(
                    self.canvas_dims[0] // 2,
                    self.canvas_dims[1] // 2,
                    image=self.preview_frame
                )
                
                # Create text items once and store their IDs
                if self.show_fps.get():
                    self.fps_id = self.preview_canvas.create_text(
                        10, 10, text="", fill="yellow",
                        font=("Arial", 10, "bold"), anchor=tk.NW
                    )
                    self.ui_fps_id = self.preview_canvas.create_text(
                        10, 30, text="", fill="yellow",
                        font=("Arial", 10, "bold"), anchor=tk.NW
                    )
                    self.img_dims_id = self.preview_canvas.create_text(
                        10, 50, text="", fill="yellow",
                        font=("Arial", 10, "bold"), anchor=tk.NW
                    )
            else:
                # Just update the existing image item instead of recreating
                if hasattr(self, 'image_id'):
                    self.preview_canvas.itemconfig(self.image_id, image=self.preview_frame)
                else:
                    # Create image item if it doesn't exist yet
                    self.image_id = self.preview_canvas.create_image(
                        current_dims[0] // 2,
                        current_dims[1] // 2,
                        image=self.preview_frame
                    )
            
            # Update FPS counter if enabled (updating text is cheaper than recreating)
            if self.show_fps.get() and hasattr(self, 'fps_id'):
                # Only update text every few frames to reduce overhead
                if self.ui_frame_count % 5 == 0:  # Update every 5 frames
                    self.preview_canvas.itemconfig(
                        self.fps_id, 
                        text=f"Camera FPS: {self.fps:.1f}"
                    )
                    self.preview_canvas.itemconfig(
                        self.ui_fps_id, 
                        text=f"UI FPS: {self.ui_fps:.1f}"
                    )
                    
                    # Update image dimensions less frequently
                    if self.ui_frame_count % 30 == 0:  # Update every 30 frames
                        img_width = photo.width()
                        img_height = photo.height()
                        self.preview_canvas.itemconfig(
                            self.img_dims_id,
                            text=f"Image: {img_width}x{img_height}"
                        )
            
            # Update UI frame counter
            self.ui_frame_count += 1
            current_time = time.time()
            time_diff = current_time - self.ui_last_fps_time
            
            if time_diff >= 0.5:
                self.ui_fps = self.ui_frame_count / time_diff
                self.ui_frame_count = 0
                self.ui_last_fps_time = current_time
                
        except Exception as e:
            if self.debug_level > 0:
                self.log(f"Error updating canvas: {str(e)}")
    
    def on_closing(self):
        """Handle application close"""
        self.stream_active = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1.0)
        self.root.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DirectStreamingApp(root)
    root.mainloop()