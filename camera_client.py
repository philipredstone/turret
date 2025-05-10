import cv2
import time
from PyQt5.QtCore import QMutex


class CameraStreamClient:
    """Camera client that fetches frames on demand"""
    def __init__(self):
        self.stream_url = None
        self.video_capture = None
        self.frame = None
        self.last_frame_time = 0
        self.fps = 0
        self.debug = True
        self.mutex = QMutex()
        self.frame_count = 0
        self.connected = False
    
    def connect(self, host='127.0.0.1', port=8080):
        """Connect to the camera stream"""
        try:
            self.stream_url = f"http://{host}:{port}/camera"
            
            if self.debug:
                print(f"Connecting to camera at {self.stream_url}")
            
            # Only initialize the capture, don't start streaming
            self.video_capture = cv2.VideoCapture(self.stream_url)
            
            # Force buffer size to be small
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
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
        """Disconnect from the camera stream"""
        if self.debug:
            print("Disconnecting camera")
        
        self.connected = False
        
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass
            self.video_capture = None
    
    def get_frame(self):
        """Get a single frame on demand"""
        if not self.connected or self.video_capture is None:
            return None
        
        try:
            # Lock to prevent concurrent access
            self.mutex.lock()
            
            # Grab and retrieve a new frame
            ret, frame = self.video_capture.read()
            
            if not ret or frame is None:
                if self.debug:
                    print("Failed to read frame")
                self.mutex.unlock()
                return None
            
            # Update stats
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            
            if time_diff >= 1.0:
                self.fps = self.frame_count / time_diff
                self.frame_count = 0
                self.last_frame_time = current_time
            
            self.frame_count += 1
            self.frame = frame
            
            self.mutex.unlock()
            return frame.copy()
        
        except Exception as e:
            if self.debug:
                print(f"Error getting frame: {str(e)}")
            
            if self.mutex.tryLock():
                self.mutex.unlock()
            
            return None