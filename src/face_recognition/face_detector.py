"""
Face detection module supporting multiple detection methods
"""
import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector supporting Haar Cascades, DNN, and other methods
    """
    
    def __init__(self, config):
        """
        Initialize face detector
        
        Args:
            config (dict): Face detection configuration
        """
        self.method = config.get('method', 'haar')
        self.config = config
        self.detector = None
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the face detector based on method"""
        if self.method == 'haar':
            self._init_haar_detector()
        elif self.method == 'dnn':
            self._init_dnn_detector()
        else:
            logger.warning(f"Unknown detection method: {self.method}, falling back to Haar")
            self.method = 'haar'
            self._init_haar_detector()
    
    def _init_haar_detector(self):
        """Initialize Haar Cascade detector"""
        try:
            cascade_path = self.config.get('haar_cascade_path', 'haarcascade_frontalface_default.xml')
            
            # Try to load from OpenCV data directory if not found
            if not os.path.exists(cascade_path):
                cv2_data_dir = cv2.data.haarcascades
                cascade_path = os.path.join(cv2_data_dir, 'haarcascade_frontalface_default.xml')
            
            self.detector = cv2.CascadeClassifier(cascade_path)
            
            if self.detector.empty():
                raise ValueError("Failed to load Haar Cascade classifier")
            
            logger.info("Haar Cascade detector initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Haar detector: {e}")
            raise
    
    def _init_dnn_detector(self):
        """Initialize DNN-based face detector"""
        try:
            model_path = self.config.get('dnn_model_path')
            weights_path = self.config.get('dnn_weights_path')
            
            if not model_path or not weights_path:
                raise ValueError("DNN model paths not specified in config")
            
            if not os.path.exists(model_path) or not os.path.exists(weights_path):
                raise ValueError(f"DNN model files not found: {model_path}, {weights_path}")
            
            self.detector = cv2.dnn.readNetFromCaffe(model_path, weights_path)
            logger.info("DNN face detector initialized")
            
        except Exception as e:
            logger.error(f"Error initializing DNN detector: {e}")
            raise
    
    def detect(self, frame):
        """
        Detect faces in frame
        
        Args:
            frame (np.ndarray): Input image frame
            
        Returns:
            list: List of face bounding boxes [(x, y, w, h), ...]
        """
        if self.method == 'haar':
            return self._detect_haar(frame)
        elif self.method == 'dnn':
            return self._detect_dnn(frame)
        
        return []
    
    def _detect_haar(self, frame):
        """
        Detect faces using Haar Cascades
        
        Args:
            frame (np.ndarray): Input image frame
            
        Returns:
            list: List of face bounding boxes
        """
        # Convert to grayscale for Haar detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        scale_factor = self.config.get('scale_factor', 1.1)
        min_neighbors = self.config.get('min_neighbors', 5)
        min_size = tuple(self.config.get('min_size', [30, 30]))
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return [tuple(face) for face in faces]
    
    def _detect_dnn(self, frame):
        """
        Detect faces using DNN
        
        Args:
            frame (np.ndarray): Input image frame
            
        Returns:
            list: List of face bounding boxes
        """
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        faces = []
        confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Convert to (x, y, w, h) format
                x = max(0, x1)
                y = max(0, y1)
                width = min(x2 - x1, w - x)
                height = min(y2 - y1, h - y)
                
                faces.append((x, y, width, height))
        
        return faces