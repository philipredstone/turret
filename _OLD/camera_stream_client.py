import cv2
import threading
import time
import numpy as np

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
        self.debug = False  # Enable debug output
        self.frame_hash = None  # Track frame content changes
       
    def connect(self, host='127.0.0.1', port=8080):
        """Connect to the camera stream"""
        try:
            self.stream_url = f"http://{host}:{port}/camera"
            
            if self.debug:
                print(f"Connecting to camera at {self.stream_url}")
           
            # Create video capture
            self.video_capture = cv2.VideoCapture(self.stream_url)
            
            # Force buffer size to be small to avoid old frames
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
           
            if not self.video_capture.isOpened():
                if self.debug:
                    print(f"Failed to open video stream at {self.stream_url}")
                if self.error_callback:
                    self.error_callback(f"Failed to open video stream at {self.stream_url}")
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
            if self.error_callback:
                self.error_callback(f"Camera stream connection error: {str(e)}")
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
        
        while self.streaming and self.video_capture:
            try:
                # Flush the buffer by reading multiple frames
                for _ in range(2):  # Discard old frames
                    self.video_capture.grab()
                
                # Now get the latest frame
                ret, frame = self.video_capture.read()
                
                if not ret or frame is None:
                    # If frame grab failed, try again after a short delay
                    if self.debug:
                        print("Failed to read frame")
                    consecutive_errors += 1
                    if consecutive_errors > 10:
                        if self.debug:
                            print("Too many consecutive frame read errors, reconnecting...")
                        # Try to reconnect
                        try:
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
                
                consecutive_errors = 0
                
                # Check if this is a new frame (simple hash to detect changes)
                frame_bytes = frame.tobytes()[:1000]  # Use first portion for faster hashing
                new_hash = hash(frame_bytes)
                is_new_frame = new_hash != self.frame_hash
                
                if is_new_frame:
                    if self.debug and self.frame_hash is not None:
                        print(f"New frame detected, hash changed: {self.frame_hash} -> {new_hash}")
                    self.frame_hash = new_hash
                
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
                else:
                    if self.debug:
                        print("Frame unchanged, skipping")
               
                # Small delay to reduce CPU usage
                time.sleep(0.01)
            except Exception as e:
                if self.debug:
                    print(f"Stream error: {str(e)}")
                if self.streaming and self.error_callback:
                    self.error_callback(f"Stream error: {str(e)}")
                consecutive_errors += 1
                time.sleep(0.5)  # Longer delay on error
   
    def get_frame(self):
        """Get the current frame with forced refresh"""
        # Force a new grab if we have a capture
        if self.video_capture and self.video_capture.isOpened():
            try:
                # Flush the buffer
                self.video_capture.grab()
                
                # Read directly from the camera
                ret, direct_frame = self.video_capture.read()
                if ret and direct_frame is not None:
                    # Update our stored frame
                    with self.lock:
                        self.frame = direct_frame
                        self.last_frame_update = time.time()
                        self.frame_count += 1
                    
                    if self.debug:
                        print("Directly grabbed new frame")
                    return direct_frame
            except Exception as e:
                if self.debug:
                    print(f"Error in direct frame grab: {str(e)}")
        
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
        
        return frame