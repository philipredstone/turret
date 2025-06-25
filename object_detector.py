import cv2
import numpy as np
import time
import os
import sys
import shutil
from threading import Thread, Lock
import traceback

class YoloObjectDetector:
    """YOLO-based object detector with targeting capabilities"""
    
    def __init__(self, model_name="yolov8n", confidence_threshold=0.5):
        # Initialize callbacks first since _log_status might get called early
        self.detection_callback = None
        self.status_callback = None
        
        self.model = None
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.class_names = {}
        self.target_classes = []  # Filter detections to these classes only
        
        # Set up models directory
        try:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback for interactive environments where __file__ isn't defined
            current_file_path = os.getcwd()
            self._log_status(f"Warning: __file__ not defined, using current working directory for models: {current_file_path}")

        self.models_dir = os.path.join(current_file_path, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self._log_status(f"Using models directory: {self.models_dir}")
        
        # Detection state
        self.detections = []
        self.detection_frame = None
        self.detection_lock = Lock()
        
        # Threading control
        self.detection_thread = None
        self.is_detecting = False
        self._stop_detection_flag = False
        
        # Targeting config
        self.targeting_mode = "largest"  # largest, center, closest
        self.min_detection_area = 100    # Skip tiny detections
        
        # Load model on init
        self.load_model(model_name)
    
    def load_model(self, model_name="yolov8n"):
        """Load YOLO model, downloading if needed"""
        self._log_status(f"Loading YOLO model {model_name} using Ultralytics...")
        
        # Install ultralytics if missing
        try:
            import ultralytics
            from ultralytics.utils.downloads import download
            from ultralytics import YOLO
        except ImportError:
            self._log_status("Ultralytics package not found. Attempting to install...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                self._log_status("Ultralytics installed successfully. Please restart the script if issues persist.")
                import ultralytics 
                from ultralytics.utils.downloads import download
                from ultralytics import YOLO
            except Exception as e:
                self._log_status(f"Failed to install ultralytics: {str(e)}")
                self._log_status("Please install ultralytics manually: pip install ultralytics")
                return False
        
        try:
            # Add .pt extension if missing
            if not model_name.endswith('.pt') and not model_name.endswith('.yaml'):
                model_with_ext = f"{model_name}.pt"
            else:
                model_with_ext = model_name
            
            model_path = os.path.join(self.models_dir, model_with_ext)
            
            # Check if we already have the model locally
            if os.path.exists(model_path):
                self._log_status(f"Found local model at {model_path}")
            else:
                # Standard Ultralytics models that we can download
                standard_models = [
                    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
                    "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
                ]
                
                model_base = model_name.replace('.pt', '').replace('.yaml', '')
                
                if model_base in standard_models or model_name.startswith('yolo'):
                    self._log_status(f"Model {model_with_ext} not found in models directory: {self.models_dir}.")
                    self._log_status(f"Attempting to download to {model_path}...")
                    
                    try:
                        # Let YOLO handle the download first
                        self._log_status(f"Letting Ultralytics handle the download for '{model_name}'...")
                        temp_model = YOLO(model_name)  # Downloads to ultralytics cache
                        
                        # Try to copy from cache to our models directory
                        try:
                            if model_base in standard_models:
                                from ultralytics.hub.utils import check_model_file
                                model_file = check_model_file(model_name, hard=True)
                                self._log_status(f"Ultralytics downloaded/found model at: {model_file}")
                                if not os.path.exists(model_path):
                                    self._log_status(f"Copying model from {model_file} to {model_path}")
                                    shutil.copy2(model_file, model_path)
                                    self._log_status(f"Successfully copied model to {model_path}")
                                else:
                                    self._log_status(f"Model already exists in target directory: {model_path}")
                            else:
                                self._log_status(f"Model '{model_name}' is not a recognized standard model.")
                                pass  # Fall through to YOLO resolution

                        except Exception as direct_download_err:
                            self._log_status(f"Direct download/copy attempt failed: {direct_download_err}. Relying on YOLO to load/download.")

                    except Exception as download_err:
                        self._log_status(f"Error during model acquisition: {str(download_err)}")
                        self._log_status(traceback.format_exc())
                        if model_path != model_name:
                            self._log_status(f"Failed to ensure model is at {model_path}. Trying to load '{model_name}' directly.")
                            model_path = model_name
                        else:
                            self._log_status(f"Cannot find or download model: {model_name}")
                            return False
                
                elif os.path.exists(model_name):  # Custom model path
                    self._log_status(f"Using custom model at path: {model_name}")
                    model_path = model_name
                else:
                    self._log_status(f"Model '{model_name}' not found in models directory, not a standard model, and not a valid local path.")
                    self._log_status(f"Attempting to load '{model_name}' and let Ultralytics handle it (may download or fail).")
                    model_path = model_name

            # Actually load the model
            if not os.path.exists(model_path) and (model_path.endswith('.pt') or model_path.endswith('.yaml')):
                 self._log_status(f"Warning: Model file {model_path} does not exist. YOLO will attempt to download if it's a known model name.")
            
            self._log_status(f"Initializing YOLO model with: {model_path}")
            self.model = YOLO(model_path)
            
            # Extract class names from model
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
                self._log_status(f"Model has {len(self.class_names)} classes.")
                
                # Normalize class_names to dict {int: str}
                if isinstance(self.class_names, list):
                    self.class_names = {i: name for i, name in enumerate(self.class_names)}
                elif isinstance(self.class_names, dict):
                    # Re-index if keys aren't integers
                    if not all(isinstance(k, int) for k in self.class_names.keys()):
                        self._log_status("Class names dict has non-integer keys, re-indexing.")
                        self.class_names = {i: name for i, name in enumerate(self.class_names.values())}
                else:
                    self._log_status(f"Unexpected format for model.names: {type(self.model.names)}. Using fallback.")
                    self._use_coco_fallback_names()

            else:
                self._log_status("Model doesn't have 'names' attribute or it's empty. Using COCO classes as fallback.")
                self._use_coco_fallback_names()
            
            # Target all classes by default
            self.target_classes = list(self.class_names.values())
            
            self._log_status(f"Successfully loaded model '{model_name}' (resolved to '{model_path}').")
            self._log_status(f"Available classes ({len(self.class_names)}): {list(self.class_names.values())[:10]}...")
            
            return True
            
        except Exception as e:
            self._log_status(f"Failed to load model '{model_name}': {str(e)}")
            self._log_status(traceback.format_exc())
            self._log_status("Make sure 'ultralytics' is installed (pip install ultralytics) and the model name/path is correct.")
            return False

    def _use_coco_fallback_names(self):
        """Fallback to COCO class names if model doesn't provide them"""
        self.class_names = {i: name for i, name in enumerate([
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ])}
        self._log_status(f"Using COCO fallback class names ({len(self.class_names)} classes).")

    def set_target_classes(self, classes):
        """Set which classes to target (filters detections)"""
        if not self.class_names:
            self._log_status("Cannot set target classes: model not loaded or has no class names.")
            return

        valid_model_classes = list(self.class_names.values())
        
        if isinstance(classes, str):
            classes_to_set = [classes]
        elif isinstance(classes, list):
            classes_to_set = classes
        else:
            self._log_status(f"Invalid type for classes: {type(classes)}. Please provide a string or list of strings.")
            return

        # Filter against actual model classes
        new_target_classes = []
        unknown_classes = []
        for cls_name in classes_to_set:
            if cls_name in valid_model_classes:
                new_target_classes.append(cls_name)
            else:
                unknown_classes.append(cls_name)
        
        if unknown_classes:
            self._log_status(f"Warning: The following target classes are not available in the loaded model and will be ignored: {unknown_classes}")
            self._log_status(f"Available model classes: {valid_model_classes}")

        if not new_target_classes and classes_to_set:
             self._log_status(f"No valid target classes found from input: {classes_to_set}. Retaining previous target classes.")
        elif not new_target_classes and not classes_to_set:
            self._log_status("Setting target classes to all available model classes.")
            self.target_classes = valid_model_classes
        else:
            self.target_classes = list(set(new_target_classes))  # Remove duplicates
            self._log_status(f"Set target classes to: {self.target_classes}")

    def set_targeting_mode(self, mode):
        """Set how to pick the best target: largest, center, closest"""
        if mode in ["largest", "center", "closest"]:
            self.targeting_mode = mode
            self._log_status(f"Targeting mode set to: {mode}")
        else:
            self._log_status(f"Invalid targeting mode: {mode}. Using 'largest'. Valid modes are 'largest', 'center', 'closest'.")
            self.targeting_mode = "largest"
    
    def set_confidence_threshold(self, threshold):
        """Set minimum confidence for detections (0.01-0.99)"""
        try:
            threshold = float(threshold)
            self.confidence_threshold = max(0.01, min(threshold, 0.99))
            self._log_status(f"Confidence threshold set to: {self.confidence_threshold:.2f}")
        except ValueError:
             self._log_status(f"Invalid confidence threshold value: {threshold}. Must be a number.")
    
    def start_detection(self, frame_provider):
        """Start detection in background thread using frame_provider function"""
        if self.is_detecting:
            self._log_status("Detection already running.")
            return False
        
        if self.model is None:
            self._log_status("Cannot start detection: Model not loaded.")
            return False
        
        if not callable(frame_provider):
            self._log_status("Cannot start detection: frame_provider is not a callable function.")
            return False

        self._stop_detection_flag = False
        self.is_detecting = True
        
        self.detection_thread = Thread(
            target=self._detection_worker,
            args=(frame_provider,),
            name="YoloDetectionThread"
        )
        self.detection_thread.daemon = True  # Don't block program exit
        self.detection_thread.start()
        
        self._log_status("Object detection thread started.")
        return True
    
    def stop_detection(self):
        """Stop the detection thread"""
        self._log_status("Attempting to stop detection thread...")
        self._stop_detection_flag = True
        
        if self.detection_thread and self.detection_thread.is_alive():
            try:
                self.detection_thread.join(timeout=2.0)
                if self.detection_thread.is_alive():
                    self._log_status("Detection thread did not stop in time.")
                else:
                    self._log_status("Detection thread stopped.")
            except Exception as e:
                self._log_status(f"Error stopping detection thread: {str(e)}")
        elif not self.detection_thread:
             self._log_status("Detection thread was not running or already stopped.")
        
        self.is_detecting = False
        self.detection_thread = None
        self._log_status("Object detection process marked as stopped.")
    
    def _detection_worker(self, frame_provider):
        """Main detection loop running in background thread"""
        last_detection_time = 0
        min_detection_interval = 0.05  # Max 20 FPS for detection
        
        self._log_status("Detection worker started.")
        while not self._stop_detection_flag:
            # Get frame from provider
            try:
                frame = frame_provider()
            except Exception as fp_exc:
                self._log_status(f"Error getting frame from frame_provider: {fp_exc}")
                self._log_status(traceback.format_exc())
                time.sleep(0.5)  # Wait before retrying
                continue
            
            if frame is None:
                time.sleep(0.01)  # Brief wait if no frame
                continue
            
            if not isinstance(frame, np.ndarray):
                self._log_status(f"Frame provider returned invalid type: {type(frame)}. Expecting numpy array.")
                time.sleep(0.1)
                continue

            # Throttle detection rate
            current_time = time.time()
            if current_time - last_detection_time < min_detection_interval:
                time.sleep(0.005)
                continue
            
            # Run detection
            try:
                # Run YOLO inference
                results = self.model.predict(frame, verbose=False, conf=self.confidence_threshold) 
                
                # Process results into our format
                processed_detections = self._process_results(results, frame)
                
                # Update thread-safe detection state
                with self.detection_lock:
                    self.detections = processed_detections

                # Call callback if set and we have detections
                if self.detection_callback and processed_detections:
                    vis_frame = self._draw_detections(frame.copy(), processed_detections)
                    try:
                        self.detection_callback(processed_detections, vis_frame)
                    except Exception as cb_err:
                        self._log_status(f"Error in detection_callback: {cb_err}")
                        self._log_status(traceback.format_exc())
                
                last_detection_time = current_time
                
            except Exception as e:
                self._log_status(f"Detection error in worker: {str(e)}")
                self._log_status(traceback.format_exc())
                time.sleep(0.5)  # Wait longer on error
        
        self._log_status("Detection worker finished.")
    
    def _process_results(self, results, frame):
        """Convert YOLO results to our detection format"""
        detections_list = []
        frame_height, frame_width = frame.shape[:2]
        
        try:
            if not results or not isinstance(results, list):
                return detections_list

            for result in results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue

                boxes = result.boxes.data.cpu().numpy()  # xyxy, conf, cls format
                
                for box_data in boxes:
                    if len(box_data) < 6:  # Need at least x1,y1,x2,y2,conf,cls
                        self._log_status(f"Unexpected box data format: {box_data}")
                        continue

                    x1, y1, x2, y2 = map(int, box_data[:4])
                    confidence = float(box_data[4])
                    class_id = int(box_data[5])
                                        
                    # Double-check confidence threshold
                    if confidence < self.confidence_threshold:
                        continue
                    
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    # Filter by target classes
                    if self.target_classes and class_name not in self.target_classes:
                        continue
                    
                    # Clamp coordinates to frame bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_width -1 , x2)
                    y2 = min(frame_height -1, y2)

                    if x1 >= x2 or y1 >= y2:  # Invalid box after clamping
                        continue

                    # Calculate properties
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width // 2
                    center_y = y1 + height // 2
                    area = width * height
                    
                    # Skip tiny detections
                    if area < self.min_detection_area:
                        continue
                    
                    # Distance from frame center (for center targeting)
                    frame_center_x = frame_width // 2
                    frame_center_y = frame_height // 2
                    center_distance = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
                    
                    detection_item = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'box': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'area': area,
                        'center_distance': center_distance
                    }
                    detections_list.append(detection_item)
        
        except Exception as e:
            self._log_status(f"Error processing results: {str(e)}")
            self._log_status(traceback.format_exc())
        
        return detections_list
    
    def get_target(self):
        """Get the best target based on targeting mode"""
        with self.detection_lock:
            # Filter by target classes
            current_detections = [
                d for d in self.detections 
                if not self.target_classes or d['class_name'] in self.target_classes
            ]

            if not current_detections:
                return None
            
            # Sort based on targeting preference
            if self.targeting_mode == "largest":
                # Sort by area, then confidence as tiebreaker
                sorted_detections = sorted(current_detections, key=lambda d: (d['area'], d['confidence']), reverse=True)
            elif self.targeting_mode == "center":
                # Sort by distance to center, then confidence
                sorted_detections = sorted(current_detections, key=lambda d: (d['center_distance'], -d['confidence']))
            elif self.targeting_mode == "closest":
                # Without depth info, fall back to largest as proxy for closest
                self._log_status("Warning: 'closest' targeting mode requires depth information. Falling back to 'largest'.")
                sorted_detections = sorted(current_detections, key=lambda d: (d['area'], d['confidence']), reverse=True)
            else:
                sorted_detections = current_detections
            
            if sorted_detections:
                return sorted_detections[0]
            return None
    
    def get_all_detections(self):
        """Get copy of all current detections (filtered by target_classes if set)"""
        with self.detection_lock:
            if not self.detections:
                return []
            
            if not self.target_classes:
                return self.detections.copy()
            else:
                return [d for d in self.detections if d['class_name'] in self.target_classes].copy()

    def get_detection_frame_with_overlay(self):
        """Get latest detection frame with bounding boxes drawn"""
        with self.detection_lock:
            if self.detection_frame is None or not self.detections:
                return self.detection_frame.copy() if self.detection_frame is not None else None

            vis_frame = self._draw_detections(self.detection_frame.copy(), self.detections)
            return vis_frame

    def _draw_detections(self, frame, detections_to_draw):
        """Draw bounding boxes and labels on frame"""
        if not isinstance(frame, np.ndarray):
            self._log_status("Invalid frame for drawing.")
            return frame

        for detection in detections_to_draw:
            try:
                x1, y1, x2, y2 = map(int, detection['box'])
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Color based on confidence (green=high, red=low)
                green_val = int(255 * confidence)
                red_val = int(255 * (1 - confidence))
                color = (0, green_val, red_val)  # BGR format
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                label = f"{class_name} {confidence:.2f}"
                font_scale = 0.5
                font_thickness = 1
                
                # Get text dimensions for background
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness + 1)
                
                # Position label background above box
                label_bg_y1 = y1 - text_height - baseline - 2
                label_bg_y2 = y1 - baseline + 2
                
                # Move label below box if not enough space above
                if label_bg_y1 < 0:
                    label_bg_y1 = y1 + baseline
                    label_bg_y2 = y1 + text_height + baseline + 2

                # Draw label background
                cv2.rectangle(frame, (x1, label_bg_y1), (x1 + text_width, label_bg_y2), color, -1)
                
                # Draw label text (white on colored background)
                cv2.putText(frame, label, (x1, label_bg_y2 - baseline // 2 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                
                # Draw center point
                center_x, center_y = map(int, detection['center'])
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)  # Red dot
            
            except Exception as draw_err:
                self._log_status(f"Error drawing detection: {detection.get('class_name', 'N/A')}. Error: {draw_err}")
                continue
        
        return frame
    
    def _log_status(self, message):
        """Log status via callback or print as fallback"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[{timestamp}] YOLO Detector: {message}"
        
        if hasattr(self, 'status_callback') and self.status_callback:
            try:
                self.status_callback(log_message)
            except Exception as cb_ex:
                # Fallback to print if callback fails
                print(f"Status callback failed: {cb_ex}")
                print(log_message)
        else:
            print(log_message)

    def __del__(self):
        """Clean up when object is destroyed"""
        self._log_status("YoloObjectDetector is being deleted. Stopping detection...")
        self.stop_detection()