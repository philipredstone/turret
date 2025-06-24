import cv2
import time
from PyQt5.QtCore import QMutex


class CameraStreamClient:
    """
    Camera client that fetches frames on demand from an HTTP video stream.
    
    This class provides a simple interface to connect to a camera stream
    (typically from a camera server) and retrieve individual frames.
    It uses OpenCV's VideoCapture for HTTP stream handling.
    """
    
    def __init__(self):
        """Initialize camera client with default settings."""
        self.stream_url = None          # HTTP URL for camera stream
        self.video_capture = None       # OpenCV VideoCapture object
        self.frame = None               # Last captured frame
        self.last_frame_time = 0        # Timestamp of last frame for FPS calculation
        self.fps = 0                    # Current frames per second
        self.debug = True               # Enable debug output
        self.mutex = QMutex()           # Thread safety for frame access
        self.frame_count = 0            # Frame counter for FPS calculation
        self.connected = False          # Connection status flag
    
    def connect(self, host='127.0.0.1', port=8080):
        """
        Connect to the camera stream at the specified host and port.
        
        Args:
            host: Camera server hostname or IP address
            port: Camera server port number
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Construct HTTP stream URL
            self.stream_url = f"http://{host}:{port}/camera"
            
            if self.debug:
                print(f"Connecting to camera at {self.stream_url}")
            
            # Initialize OpenCV VideoCapture for HTTP stream
            self.video_capture = cv2.VideoCapture(self.stream_url)
            
            # Force small buffer to reduce latency
            # This ensures we get the most recent frame, not buffered old frames
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify connection was successful
            if not self.video_capture.isOpened():
                if self.debug:
                    print(f"Failed to open video stream at {self.stream_url}")
                return False
            
            if self.debug:
                print("Video capture opened successfully")
            
            self.connected = True
            return True
            
        except Exception as e:
            if self.debug:
                print(f"Camera connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from the camera stream and cleanup resources."""
        if self.debug:
            print("Disconnecting camera")
        
        self.connected = False
        
        # Release OpenCV VideoCapture resources
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass  # Ignore errors during cleanup
            self.video_capture = None
    
    def get_frame(self):
        """
        Get a single frame from the camera on demand.
        
        This method fetches the most recent frame from the camera stream.
        It includes frame rate calculation and thread safety.
        
        Returns:
            numpy.ndarray or None: Camera frame as BGR image, or None if failed
        """
        if not self.connected or self.video_capture is None:
            return None
        
        try:
            # Lock to prevent concurrent access from multiple threads
            self.mutex.lock()
            
            # Grab and retrieve a new frame from the stream
            ret, frame = self.video_capture.read()
            
            if not ret or frame is None:
                if self.debug:
                    print("Failed to read frame")
                self.mutex.unlock()
                return None
            
            # Update frame rate statistics
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            
            # Calculate FPS every second
            if time_diff >= 1.0:
                self.fps = self.frame_count / time_diff
                self.frame_count = 0
                self.last_frame_time = current_time
            
            self.frame_count += 1
            self.frame = frame
            
            self.mutex.unlock()
            
            # Return a copy to prevent external modification of internal frame
            return frame.copy()
        
        except Exception as e:
            if self.debug:
                print(f"Error getting frame: {str(e)}")
            
            # Ensure mutex is unlocked even if exception occurs
            if self.mutex.tryLock():
                self.mutex.unlock()
            
            return None