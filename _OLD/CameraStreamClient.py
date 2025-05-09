import cv2
import threading
import time
import numpy as np
import urllib.request
import io
from PIL import Image

class CameraStreamClient:
    def __init__(self):
        self.stream_url = None
        self.video_capture = None
        self.streaming = False
        self.stream_thread = None
        self.frame_callback = None
        self.error_callback = None
        self.frame = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0
        self.last_frame_update = 0
        self.debug = True  # Enable debug output
        self.frame_hash = None  # Track frame content changes
        self.connection_attempt = 0
        self.max_connection_attempts = 3
        self.use_direct_http = False  # Use direct HTTP requests instead of OpenCV
        self.stream_boundary = None  # For multipart MJPEG stream boundary
       
    def connect(self, host='127.0.0.1', port=8080):
        """Connect to the camera stream"""
        try:
            self.stream_url = f"http://{host}:{port}/camera"
            
            if self.debug:
                print(f"Connecting to camera at {self.stream_url}")
            
            # Reset connection attempt counter
            self.connection_attempt = 0
            
            # Try to connect with OpenCV first
            if not self.use_direct_http:
                self._connect_with_opencv()
            
            # If OpenCV connection failed or use_direct_http is True, try direct HTTP
            if self.video_capture is None or not self.video_capture.isOpened() or self.use_direct_http:
                if self.debug:
                    print("Using direct HTTP connection instead of OpenCV")
                self.use_direct_http = True
                
                # Release OpenCV resources if they exist
                if self.video_capture:
                    self.video_capture.release()
                    self.video_capture = None
                
                # Test the HTTP connection
                try:
                    req = urllib.request.Request(self.stream_url)
                    with urllib.request.urlopen(req, timeout=5) as response:
                        content_type = response.getheader('Content-Type')
                        if 'multipart/x-mixed-replace' in content_type:
                            # Get boundary for MJPEG stream
                            self.stream_boundary = content_type.split('boundary=')[1]
                            if self.debug:
                                print(f"Found MJPEG stream with boundary: {self.stream_boundary}")
                        else:
                            if self.debug:
                                print(f"Warning: Content type is not multipart/x-mixed-replace: {content_type}")
                except Exception as e:
                    if self.debug:
                        print(f"HTTP connection test failed: {str(e)}")
                    if self.error_callback:
                        self.error_callback(f"HTTP connection test failed: {str(e)}")
                    return False
            
            # Start streaming thread
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._stream_loop, name="CameraStreamThread")
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            if self.debug:
                print("Stream thread started")
           
            return True
        except Exception as e:
            if self.debug:
                print(f"Camera connection error: {str(e)}")
            if self.error_callback:
                self.error_callback(f"Camera stream connection error: {str(e)}")
            return False
    
    def _connect_with_opencv(self):
        """Attempt to connect using OpenCV"""
        try:
            # Create video capture
            self.video_capture = cv2.VideoCapture(self.stream_url)
            
            # Force buffer size to be small to avoid old frames
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
           
            if not self.video_capture.isOpened():
                if self.debug:
                    print(f"Failed to open video stream with OpenCV at {self.stream_url}")
                return False
            
            if self.debug:
                print("Video capture opened successfully with OpenCV")
            return True
        except Exception as e:
            if self.debug:
                print(f"OpenCV connection error: {str(e)}")
            return False

    def disconnect(self):
        """Disconnect from the camera stream"""
        if self.debug:
            print("Disconnecting camera stream")
        self.streaming = False
        if self.stream_thread and self.stream_thread.is_alive():
            try:
                self.stream_thread.join(timeout=1.0)  # Wait for thread to finish
            except:
                pass  # Ignore errors on join
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass  # Ignore errors on release
            self.video_capture = None
        if self.debug:
            print("Camera stream disconnected")
   
    def _stream_loop(self):
        """Thread function to read frames from the stream"""
        consecutive_errors = 0
        
        while self.streaming:
            try:
                if self.use_direct_http:
                    # Process stream using direct HTTP
                    frame = self._get_frame_http()
                else:
                    # Process stream using OpenCV
                    frame = self._get_frame_opencv()
                
                if frame is not None:
                    # Store the frame and update timestamp
                    with self.lock:
                        self.frame = frame
                        self.frame_count += 1
                        self.last_frame_update = time.time()
                   
                    # Calculate FPS
                    current_time = time.time()
                    if current_time - self.last_frame_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_frame_time = current_time
                        if self.debug:
                            print(f"Camera FPS: {self.fps}")
                   
                    # Notify callback
                    if self.frame_callback:
                        self.frame_callback(frame)
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if self.debug:
                        print(f"No frame received (error count: {consecutive_errors})")
                    
                    # If we keep failing, try to reconnect or switch methods
                    if consecutive_errors > 20:
                        if not self.use_direct_http:
                            if self.debug:
                                print("Switching to direct HTTP method")
                            self.use_direct_http = True
                            consecutive_errors = 0
                        else:
                            if self.debug:
                                print("Attempting to reconnect")
                            # Try reconnect
                            self._attempt_reconnect()
                            consecutive_errors = 0
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
            except Exception as e:
                if self.debug:
                    print(f"Stream error: {str(e)}")
                if self.streaming and self.error_callback:
                    self.error_callback(f"Stream error: {str(e)}")
                consecutive_errors += 1
                time.sleep(0.5)  # Longer delay on error
                
                # Try to reconnect if persistent errors
                if consecutive_errors > 5:
                    self._attempt_reconnect()
                    consecutive_errors = 0
    
    def _get_frame_opencv(self):
        """Get a frame using OpenCV"""
        if not self.video_capture or not self.video_capture.isOpened():
            return None
        
        try:
            # Flush the buffer by reading multiple frames
            for _ in range(2):  # Discard old frames
                self.video_capture.grab()
            
            # Now get the latest frame
            ret, frame = self.video_capture.read()
            
            if not ret or frame is None:
                if self.debug:
                    print("Failed to read frame with OpenCV")
                return None
            
            return frame
        except Exception as e:
            if self.debug:
                print(f"Error getting frame with OpenCV: {str(e)}")
            return None
    
    def _get_frame_http(self):
        """Get a frame using direct HTTP requests"""
        try:
            req = urllib.request.Request(self.stream_url)
            with urllib.request.urlopen(req, timeout=1.0) as response:
                content_type = response.getheader('Content-Type')
                
                # Check if this is a single image or a stream
                if 'multipart/x-mixed-replace' in content_type:
                    # This is a MJPEG stream, read one frame
                    return self._read_mjpeg_frame(response)
                elif 'image/jpeg' in content_type or 'image/jpg' in content_type:
                    # Single JPEG image
                    img_data = response.read()
                    img = Image.open(io.BytesIO(img_data))
                    # Convert PIL image to OpenCV format
                    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                else:
                    if self.debug:
                        print(f"Unsupported content type: {content_type}")
                    return None
        except Exception as e:
            if self.debug:
                print(f"HTTP frame fetch error: {str(e)}")
            return None
    
    def _read_mjpeg_frame(self, response):
        """Read a single frame from an MJPEG stream"""
        try:
            # If boundary isn't known, try to get it
            if not self.stream_boundary and 'multipart/x-mixed-replace' in response.getheader('Content-Type'):
                self.stream_boundary = response.getheader('Content-Type').split('boundary=')[1]
            
            if not self.stream_boundary:
                if self.debug:
                    print("No MJPEG boundary found")
                return None
            
            boundary = f'--{self.stream_boundary}'
            boundary_bytes = boundary.encode()
            
            # Read until boundary
            while True:
                line = response.readline().strip()
                if boundary_bytes in line:
                    break
            
            # Read headers
            content_length = None
            while True:
                line = response.readline().strip()
                if not line:
                    break
                
                if b'Content-Length' in line:
                    content_length = int(line.split(b':')[1])
            
            if not content_length:
                if self.debug:
                    print("No Content-Length in MJPEG frame")
                return None
            
            # Read image data
            image_data = response.read(content_length)
            
            # Convert to OpenCV image
            img = Image.open(io.BytesIO(image_data))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            if self.debug:
                print(f"Error reading MJPEG frame: {str(e)}")
            return None
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to the camera stream"""
        if self.connection_attempt >= self.max_connection_attempts:
            if self.debug:
                print(f"Max connection attempts ({self.max_connection_attempts}) reached, giving up")
            return
        
        self.connection_attempt += 1
        
        if self.debug:
            print(f"Attempting to reconnect (attempt {self.connection_attempt}/{self.max_connection_attempts})")
        
        # Release current resources
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass
            self.video_capture = None
        
        # Try to reconnect
        try:
            if not self.use_direct_http:
                self._connect_with_opencv()
            else:
                # Test HTTP connection
                req = urllib.request.Request(self.stream_url)
                with urllib.request.urlopen(req, timeout=5):
                    pass  # Just checking if connection works
            
            if self.debug:
                print("Reconnection successful")
            # Reset connection attempt counter on success
            self.connection_attempt = 0
        except Exception as e:
            if self.debug:
                print(f"Reconnection failed: {str(e)}")
   
    def get_frame(self):
        """Get the current frame with forced refresh"""
        # Try to get a fresh frame
        if self.use_direct_http:
            try:
                direct_frame = self._get_frame_http()
                if direct_frame is not None:
                    # Update our stored frame
                    with self.lock:
                        self.frame = direct_frame
                        self.last_frame_update = time.time()
                        self.frame_count += 1
                    
                    if self.debug:
                        print("Directly grabbed new HTTP frame")
                    return direct_frame
            except Exception as e:
                if self.debug:
                    print(f"Error in direct HTTP frame grab: {str(e)}")
        elif self.video_capture and self.video_capture.isOpened():
            try:
                # Try OpenCV capture
                direct_frame = self._get_frame_opencv()
                if direct_frame is not None:
                    # Update our stored frame
                    with self.lock:
                        self.frame = direct_frame
                        self.last_frame_update = time.time()
                        self.frame_count += 1
                    
                    if self.debug:
                        print("Directly grabbed new OpenCV frame")
                    return direct_frame
            except Exception as e:
                if self.debug:
                    print(f"Error in direct OpenCV frame grab: {str(e)}")
        
        # Fallback to stored frame if direct grab failed
        with self.lock:
            if self.frame is not None:
                # Check freshness
                freshness = time.time() - self.last_frame_update
                if freshness > 2.0 and self.debug:
                    print(f"Warning: Returning stale frame ({freshness:.1f}s old)")
                return self.frame.copy()
        
        if self.debug:
            print("No frame available")
        return None
    
    def get_test_pattern(self):
        """Generate a test pattern frame for testing UI updates"""
        # Create a test pattern with the current time to ensure it changes
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add colored areas
        frame[0:height//3, :] = (0, 0, 255)  # Red top
        frame[height//3:2*height//3, :] = (0, 255, 0)  # Green middle
        frame[2*height//3:, :] = (255, 0, 0)  # Blue bottom
        
        # Add current time
        text = time.strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(frame, text, (width//2-100, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Add indicator for test pattern
        cv2.putText(frame, "TEST PATTERN", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame