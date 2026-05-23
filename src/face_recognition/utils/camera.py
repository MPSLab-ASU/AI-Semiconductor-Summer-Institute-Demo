"""
Camera capture utilities for video feed processing
"""
import cv2
import logging
import time

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Handle camera video capture with configuration options
    """
    
    def __init__(self, config):
        """
        Initialize camera capture
        
        Args:
            config (dict): Camera configuration
        """
        self.device_id = config.get('device_id', 0)
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.fps = config.get('fps', 30)
        
        self.cap = None
        self.is_opened = False
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        
    def open(self):
        """
        Open camera device
        
        Returns:
            bool: True if camera opened successfully
        """
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera device {self.device_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            self.is_opened = True
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def read(self):
        """
        Read a frame from camera
        
        Returns:
            tuple: (success, frame) - success is bool, frame is numpy array
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            
            # Update FPS every second
            elapsed = time.time() - self.start_time
            if elapsed >= 1.0:
                self.current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
        
        return ret, frame
    
    def get_fps(self):
        """
        Get current FPS
        
        Returns:
            float: Current frames per second
        """
        return self.current_fps
    
    def release(self):
        """
        Release camera resources
        """
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logger.info("Camera released")
    
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
        return False


def test_camera(device_id=0):
    """
    Test if camera is available
    
    Args:
        device_id (int): Camera device ID
        
    Returns:
        bool: True if camera is available
    """
    cap = cv2.VideoCapture(device_id)
    is_available = cap.isOpened()
    cap.release()
    return is_available
