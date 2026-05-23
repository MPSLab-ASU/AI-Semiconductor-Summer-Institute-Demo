"""
Example script demonstrating face recognition with camera feed
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from face_recognition_app import FaceRecognitionApp
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """Run face recognition with default configuration"""
    print("=" * 60)
    print("Face Recognition Application with Camera Feed")
    print("=" * 60)
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("\nStarting application...\n")
    
    # Create and run the application
    app = FaceRecognitionApp(config_path='config/config.yaml')
    app.run()


if __name__ == '__main__':
    main()
