import cv2
import threading
import time

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
        
    def connect(self, host='127.0.0.1', port=8080):
        """Connect to the camera stream"""
        try:
            self.stream_url = f"http://{host}:{port}/camera"
            
            # Create video capture
            self.video_capture = cv2.VideoCapture(self.stream_url)
            
            if not self.video_capture.isOpened():
                if self.error_callback:
                    self.error_callback(f"Failed to open video stream at {self.stream_url}")
                return False
                
            # Start streaming thread
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._stream_loop)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            return True
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Camera stream connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from the camera stream"""
        self.streaming = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
    
    def _stream_loop(self):
        """Thread function to read frames from the stream"""
        while self.streaming and self.video_capture:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    # If frame grab failed, try again after a short delay
                    time.sleep(0.05)
                    continue
                
                # Store the frame
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                
                # Notify callback
                if self.frame_callback:
                    self.frame_callback(frame)
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
            except Exception as e:
                if self.streaming and self.error_callback:
                    self.error_callback(f"Stream error: {str(e)}")
                time.sleep(0.5)  # Longer delay on error
    
    def get_frame(self):
        """Get the current frame"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None
