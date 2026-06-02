"""
Face Recognition Application with Camera Video Feed Support
"""
import cv2
import yaml
import logging
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_recognition.utils.camera import CameraCapture
from face_recognition.utils.gpu_utils import configure_gpu, get_gpu_info, check_jetson_nano
from face_recognition.face_detector import FaceDetector
from face_recognition.face_recognizer import FaceRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceRecognitionApp:
    """
    Main face recognition application with video feed processing
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the application
        
        Args:
            config_path (str): Path to configuration file
        """
        if config_path is None or config_path == 'config/config.yaml':
            config_path = str(Path(__file__).parent / 'config' / 'config.yaml')
            
        self.config = self._load_config(config_path)
        self.camera = None
        self.detector = None
        self.recognizer = None
        self.running = False
        
        # Display settings
        self.display_config = self.config.get('display', {})
        self.show_fps = self.display_config.get('show_fps', True)
        self.show_confidence = self.display_config.get('show_confidence', True)
        self.bbox_color = tuple(self.display_config.get('bbox_color', [0, 255, 0]))
        self.text_color = tuple(self.display_config.get('text_color', [255, 255, 255]))
        self.font_scale = self.display_config.get('font_scale', 0.6)
        self.thickness = self.display_config.get('thickness', 2)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file and resolve paths"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            
            # Resolve relative paths relative to src directory
            src_dir = Path(__file__).parent
            
            fd = config.get('face_detection', {})
            for key in ['dnn_model_path', 'dnn_weights_path']:
                if key in fd and not os.path.isabs(fd[key]):
                    fd[key] = str(src_dir / fd[key])
                    
            fr = config.get('face_recognition', {})
            for key in ['model_path', 'database_path']:
                if key in fr and not os.path.isabs(fr[key]):
                    fr[key] = str(src_dir / fr[key])
                    
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'camera': {'device_id': 0, 'width': 640, 'height': 480, 'fps': 30},
            'face_detection': {'method': 'haar', 'scale_factor': 1.1, 'min_neighbors': 5, 'min_size': [30, 30]},
            'face_recognition': {'embedding_size': 128, 'similarity_threshold': 0.6},
            'gpu': {'enabled': True, 'memory_growth': True},
            'display': {'show_fps': True, 'show_confidence': True, 'bbox_color': [0, 255, 0], 'text_color': [255, 255, 255]}
        }
    
    def initialize(self):
        """
        Initialize all components
        
        Returns:
            bool: True if initialization successful
        """
        logger.info("Initializing Face Recognition Application...")
        
        # Check if running on Jetson
        is_jetson = check_jetson_nano()
        if is_jetson:
            logger.info("Detected NVIDIA Jetson platform")
        
        # Configure GPU
        gpu_config = self.config.get('gpu', {})
        gpu_available = configure_gpu(gpu_config)
        
        if gpu_available:
            gpu_info = get_gpu_info()
            logger.info(f"GPU Info: {gpu_info}")
        
        # Initialize camera
        try:
            camera_config = self.config.get('camera', {})
            self.camera = CameraCapture(camera_config)
            if not self.camera.open():
                logger.error("Failed to open camera")
                return False
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
        
        # Initialize face detector
        try:
            detection_config = self.config.get('face_detection', {})
            self.detector = FaceDetector(detection_config)
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
            return False
        
        # Initialize face recognizer
        try:
            recognition_config = self.config.get('face_recognition', {})
            self.recognizer = FaceRecognizer(recognition_config)
        except Exception as e:
            logger.error(f"Error initializing face recognizer: {e}")
            return False
        
        logger.info("Initialization complete")
        return True
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            np.ndarray: Processed frame with annotations
        """
        # Detect faces
        faces = self.detector.detect(frame)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Recognize face
            name, confidence = self.recognizer.recognize(face_roi)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), self.bbox_color, self.thickness)
            
            # Prepare label
            if name:
                label = f"{name}"
                if self.show_confidence:
                    label += f" ({confidence:.2f})"
            else:
                label = "Unknown"
                if self.show_confidence and confidence > 0:
                    label += f" ({confidence:.2f})"
            
            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
            )
            cv2.rectangle(frame, (x, y - label_height - 10), (x + label_width, y), self.bbox_color, -1)
            
            # Draw label text
            cv2.putText(
                frame, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.thickness
            )
        
        return frame
    
    def run(self):
        """
        Run the face recognition application
        """
        if not self.initialize():
            logger.error("Initialization failed")
            return
        
        self.running = True
        logger.info("Starting face recognition... Press 'q' to quit")
        
        try:
            while self.running:
                # Read frame from camera
                ret, frame = self.camera.read()
                
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Add FPS counter if enabled
                if self.show_fps:
                    fps = self.camera.get_fps()
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(
                        processed_frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                
                # Display frame
                cv2.imshow('Face Recognition', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"capture_{cv2.getTickCount()}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    logger.info(f"Saved frame to {filename}")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Face Recognition Application')
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = FaceRecognitionApp(config_path=args.config)
    app.run()


if __name__ == '__main__':
    main()
