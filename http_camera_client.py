import cv2
import time
import threading
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

class HttpCameraClient(QObject):
    """A simple client for connecting to a remote HTTP camera stream with improved buffer management"""
    # Signal emitted when a new frame is available
    frame_updated = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.stream_url = None
        self.video_capture = None
        self.streaming = False
        self.stream_thread = None
        self.frame_callback = None
        self.frame = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0
        self.debug = False  # Set to True for verbose logging
    
    def connect(self, host='127.0.0.1', port=8080, path='/camera'):
        """Connect to the HTTP camera stream"""
        try:
            # Construct the URL
            self.stream_url = f"http://{host}:{port}{path}"
            if self.debug:
                print(f"Connecting to camera at {self.stream_url}")
            
            # Create video capture
            self.video_capture = cv2.VideoCapture(self.stream_url)
            
            # Set properties to minimize buffering
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer size
            
            # Check if connection was successful
            if not self.video_capture.isOpened():
                if self.debug:
                    print(f"Failed to open video stream at {self.stream_url}")
                return False
            
            if self.debug:
                print("Video capture opened successfully")
            
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
            return False
    
    def disconnect(self):
        """Disconnect from the camera stream"""
        if self.debug:
            print("Disconnecting from camera")
        self.streaming = False
        
        # Wait for thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            try:
                self.stream_thread.join(timeout=1.0)
            except:
                pass
        
        # Release video capture resources
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass
            self.video_capture = None
        
        if self.debug:
            print("Camera disconnected")
    
    def _stream_loop(self):
        """Thread function to continuously read frames from the HTTP stream"""
        consecutive_errors = 0
        
        while self.streaming and self.video_capture:
            try:
                # Flush any old frames by grabbing multiple frames
                for _ in range(3):  # Drop several frames to minimize latency
                    self.video_capture.grab()
                
                # Read the most recent frame
                ret, frame = self.video_capture.read()
                
                if not ret or frame is None:
                    # If frame grab failed, try again after a short delay
                    if self.debug:
                        print("Failed to read frame")
                    consecutive_errors += 1
                    
                    # After too many errors, try to reconnect
                    if consecutive_errors > 5:
                        if self.debug:
                            print("Too many consecutive errors, attempting to reconnect...")
                        try:
                            # Reset the connection
                            self.video_capture.release()
                            time.sleep(0.5)
                            self.video_capture = cv2.VideoCapture(self.stream_url)
                            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            consecutive_errors = 0
                        except Exception as e:
                            if self.debug:
                                print(f"Error reconnecting: {str(e)}")
                    
                    time.sleep(0.1)
                    continue
                
                # Reset error counter on successful frame
                consecutive_errors = 0
                
                # Store the frame and update counters
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_frame_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_frame_time = current_time
                    if self.debug:
                        print(f"Camera FPS: {self.fps}")
                
                # Emit signal for frame update
                self.frame_updated.emit()
                
                # Call the callback if registered
                if self.frame_callback:
                    self.frame_callback(frame)
                
                # Small delay to reduce CPU usage (make adjustable)
                time.sleep(0.01)
                
            except Exception as e:
                if self.debug:
                    print(f"Stream error: {str(e)}")
                consecutive_errors += 1
                time.sleep(0.5)  # Longer delay on error
    
    def get_frame(self):
        """Get the current frame with flush option"""
        # First try to grab fresh frames directly
        if self.video_capture and self.video_capture.isOpened():
            try:
                # Drop any buffered frames to get the most recent one
                for _ in range(2):  # Try to clear buffer
                    self.video_capture.grab()
                
                # Read the latest frame
                ret, direct_frame = self.video_capture.read()
                if ret and direct_frame is not None:
                    # Update our stored frame
                    with self.lock:
                        self.frame = direct_frame
                        self.frame_count += 1
                    
                    return direct_frame
            except Exception as e:
                if self.debug:
                    print(f"Error in direct frame grab: {str(e)}")
        
        # Fallback to the latest stored frame
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        
        return None
    
    def flush_buffer(self):
        """Explicitly flush the camera buffer to get fresh frames"""
        if self.video_capture and self.video_capture.isOpened():
            try:
                for _ in range(5):  # Discard several frames
                    self.video_capture.grab()
                return True
            except Exception as e:
                if self.debug:
                    print(f"Error flushing buffer: {str(e)}")
                return False
        return False
    
class RobustMjpegClient(QObject):
    """A robust client for connecting to MJPEG HTTP camera streams"""
    # Signal emitted when a new frame is available
    frame_updated = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.stream_url = None
        self.streaming = False
        self.stream_thread = None
        self.frame_callback = None
        self.frame = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0
        self.debug = False  # Set to True for verbose logging
        self.connection = None
        self.boundary = None
        
    def connect(self, host='127.0.0.1', port=8080, path='/camera'):
        """Connect to the HTTP camera stream"""
        try:
            # Disconnect previous connection if any
            self.disconnect()
            
            # Construct the URL
            self.stream_url = f"http://{host}:{port}{path}"
            if self.debug:
                print(f"Connecting to camera at {self.stream_url}")
            
            # Start streaming thread
            self.streaming = True
            self.stream_thread = threading.Thread(
                target=self._stream_loop, 
                name="MjpegStreamThread"
            )
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            if self.debug:
                print("Stream thread started")
            return True
            
        except Exception as e:
            if self.debug:
                print(f"Camera connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from the camera stream"""
        if self.debug:
            print("Disconnecting from camera")
        self.streaming = False
        
        # Close connection if open
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            self.connection = None
        
        # Wait for thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            try:
                self.stream_thread.join(timeout=1.0)
            except:
                pass
            
        if self.debug:
            print("Camera disconnected")
    
    def _extract_boundary(self, headers):
        """Extract boundary string from HTTP headers"""
        for line in headers.split('\r\n'):
            if line.startswith('Content-Type:'):
                match = re.search(r'boundary=(.*)', line)
                if match:
                    return match.group(1)
        return None
    
    def _connect_to_stream(self):
        """Establish HTTP connection to the MJPEG stream"""
        try:
            # Create request with long timeout
            request = urllib.request.Request(self.stream_url)
            self.connection = urllib.request.urlopen(request, timeout=10)
            
            # Get and process headers
            headers = self.connection.info().as_string()
            self.boundary = self._extract_boundary(headers)
            
            if self.boundary:
                if self.debug:
                    print(f"Found boundary: {self.boundary}")
            else:
                # If no boundary, try a default one
                self.boundary = "--boundarydonotcross"
                if self.debug:
                    print("No boundary found, using default")
            
            return True
        except Exception as e:
            if self.debug:
                print(f"Error connecting to stream: {str(e)}")
            return False
    
    def _read_jpeg_frame(self):
        """Read a single JPEG frame from the MJPEG stream"""
        if not self.connection:
            return None
        
        try:
            # Read until we find the boundary
            data = b''
            jpeg_started = False
            jpeg_data = b''
            
            while self.streaming:
                # Read some data
                chunk = self.connection.read(1024)
                if not chunk:
                    break
                
                data += chunk
                
                # Look for JPEG start marker (0xFF 0xD8)
                if not jpeg_started and b'\xff\xd8' in data:
                    jpeg_started = True
                    start_index = data.find(b'\xff\xd8')
                    jpeg_data = data[start_index:]
                    data = data[:start_index]
                
                # If we've started a JPEG, look for end marker (0xFF 0xD9)
                if jpeg_started and b'\xff\xd9' in jpeg_data:
                    end_index = jpeg_data.find(b'\xff\xd9')
                    # Include the end marker itself (2 bytes)
                    jpeg_data = jpeg_data[:end_index + 2]
                    
                    # Decode JPEG data
                    try:
                        # Use OpenCV to decode the JPEG data
                        nparr = np.frombuffer(jpeg_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            return img
                    except Exception as e:
                        if self.debug:
                            print(f"Error decoding JPEG: {str(e)}")
                    
                    # Start over for next frame
                    jpeg_started = False
                    jpeg_data = b''
            
            return None
            
        except Exception as e:
            if self.debug:
                print(f"Error reading frame: {str(e)}")
            return None
    
    def _stream_loop(self):
        """Thread function to continuously read frames from the MJPEG stream"""
        connection_attempts = 0
        max_attempts = 5
        
        while self.streaming:
            # Try to connect if we're not already connected
            if not self.connection:
                connection_attempts += 1
                if connection_attempts > max_attempts:
                    if self.debug:
                        print(f"Failed to connect after {max_attempts} attempts, giving up")
                    break
                
                if not self._connect_to_stream():
                    # If connection failed, wait and retry
                    time.sleep(2)
                    continue
                
                connection_attempts = 0
            
            # Try to read a frame
            frame = self._read_jpeg_frame()
            
            if frame is not None:
                # We successfully got a frame
                
                # Store the frame and update counters
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_frame_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_frame_time = current_time
                    if self.debug:
                        print(f"Camera FPS: {self.fps}")
                
                # Emit signal for frame update
                self.frame_updated.emit()
                
                # Call the callback if registered
                if self.frame_callback:
                    self.frame_callback(frame)
            else:
                # If we failed to read a frame, the connection might be dead
                if self.debug:
                    print("Failed to read frame, resetting connection")
                
                # Close the connection so we'll reconnect on the next iteration
                if self.connection:
                    try:
                        self.connection.close()
                    except:
                        pass
                    self.connection = None
                
                # Wait before retrying
                time.sleep(1)
    
    def get_frame(self):
        """Get the current frame"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None