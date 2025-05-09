import numpy as np
import cv2
import time
from PyQt5.QtCore import QMutex  # Or from PySide2.QtCore import QMutex

class CameraStreamClientFixed:
    """Modified camera client that fetches frames on demand only"""
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

def detect_checkerboard(image, pattern_sizes=None):
    """Detect checkerboard pattern in the image with multiple size attempts.
    
    Args:
        image: Input image (numpy array)
        pattern_sizes: List of pattern sizes to try (inner corners)
        
    Returns:
        found: Boolean indicating if pattern was found
        corners: Array of detected corner coordinates
        image: Image with corners drawn (if found)
        pattern_size: Size of pattern that was successfully detected
    """
    if pattern_sizes is None:
        # Try multiple common pattern sizes
        pattern_sizes = [(9, 6), (8, 6), (7, 7), (7, 6), (6, 9), (6, 8), (6, 7), (6, 6), (5, 8), (5, 7), (5, 5)]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to enhance the checkerboard
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Try different pattern sizes
    for pattern_size in pattern_sizes:
        # Try with regular image
        found, corners = cv2.findChessboardCorners(gray, pattern_size, 
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                                 cv2.CALIB_CB_FAST_CHECK)
        
        # If not found, try with thresholded image
        if not found:
            found, corners = cv2.findChessboardCorners(thresh, pattern_size,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                                     cv2.CALIB_CB_FAST_CHECK)
        
        if found:
            print(f"Checkerboard found with pattern size {pattern_size}")
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners on the image
            cv2.drawChessboardCorners(image, pattern_size, corners, found)
            return found, corners, image, pattern_size
    
    return False, None, image, None

def main():
    # Create and connect the camera client
    camera = CameraStreamClientFixed()
    if not camera.connect():
        print("Failed to connect to camera")
        return
    
    try:
        print("Camera connected. Getting frame...")
        # Get a frame from the camera
        frame = camera.get_frame()
        
        if frame is None:
            print("Failed to get frame from camera")
            camera.disconnect()
            return
        
        print("Frame received. Detecting checkerboard pattern...")
        # Save the original frame for debugging
        cv2.imwrite("original_frame.jpg", frame)
        
        # Detect checkerboard with multiple pattern sizes
        found, corners, marked_image, pattern_size = detect_checkerboard(frame)
        
        if found:
            print(f"Checkerboard pattern found! Detected {len(corners)} corners with pattern size {pattern_size}.")
            
            # Save the image with marked corners
            cv2.imwrite("checkerboard_detected.jpg", marked_image)
            
            # Display the image
            cv2.imshow("Detected Checkerboard", marked_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No checkerboard pattern detected after trying multiple sizes.")
            
            # Try alternative detection approach
            print("Attempting alternative detection method...")
            # Convert to grayscale and apply Gaussian blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            cv2.imwrite("edges.jpg", edges)
            
            # Save the failed attempt for debugging
            cv2.imwrite("no_checkerboard.jpg", frame)
            
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Always disconnect from the camera
        camera.disconnect()
        print("Camera disconnected")

if __name__ == "__main__":
    main()