"""
Test script for face recognition application components
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        from face_recognition.utils.camera import CameraCapture, test_camera
        logger.info("✓ Camera module imported")
        
        from face_recognition.utils.gpu_utils import configure_gpu, get_gpu_info
        logger.info("✓ GPU utils imported")
        
        from face_recognition.face_detector import FaceDetector
        logger.info("✓ Face detector imported")
        
        from face_recognition.face_recognizer import FaceRecognizer
        logger.info("✓ Face recognizer imported")
        
        from face_recognition_app import FaceRecognitionApp
        logger.info("✓ Main app imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_hardware_accelerator():
    """Test Hardware Accelerator (GPU/NPU) configuration"""
    logger.info("\nTesting Hardware Accelerator (GPU/NPU) configuration...")
    
    try:
        from face_recognition.utils.gpu_utils import get_gpu_info
        
        gpu_info = get_gpu_info()
        logger.info(f"Accelerator Available: {gpu_info['available']}")
        logger.info(f"Accelerator Count: {gpu_info['count']}")
        
        if gpu_info['available']:
            logger.info("✓ Hardware Accelerator (GPU/NPU) detected and available")
        else:
            logger.warning("⚠ No Hardware Accelerator detected, will run on CPU fallback")
        
        return True
    except Exception as e:
        logger.error(f"✗ Hardware Accelerator test failed: {e}")
        return False


def test_camera():
    """Test camera availability"""
    logger.info("\nTesting camera...")
    
    try:
        from face_recognition.utils.camera import test_camera
        
        # Test default camera
        camera_available = test_camera(0)
        
        if camera_available:
            logger.info("✓ Camera device 0 is available")
        else:
            logger.warning("⚠ Camera device 0 not available")
            logger.info("Testing alternate device IDs...")
            
            # Try other device IDs
            for device_id in range(1, 5):
                if test_camera(device_id):
                    logger.info(f"✓ Camera found at device {device_id}")
                    break
        
        return True
    except Exception as e:
        logger.error(f"✗ Camera test failed: {e}")
        return False


def test_face_detector():
    """Test face detector initialization"""
    logger.info("\nTesting face detector...")
    
    try:
        from face_recognition.face_detector import FaceDetector
        
        config = {
            'method': 'haar',
            'scale_factor': 1.1,
            'min_neighbors': 5,
            'min_size': [30, 30]
        }
        
        detector = FaceDetector(config)
        logger.info("✓ Face detector initialized successfully")
        
        return True
    except Exception as e:
        logger.error(f"✗ Face detector test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading"""
    logger.info("\nTesting configuration...")
    
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("✓ Configuration file loaded successfully")
            logger.info(f"  Camera device: {config['camera']['device_id']}")
            logger.info(f"  Detection method: {config['face_detection']['method']}")
            logger.info(f"  GPU enabled: {config['gpu']['enabled']}")
        else:
            logger.warning("⚠ Configuration file not found at expected location")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Face Recognition Application - Component Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Hardware Accelerator", test_hardware_accelerator),
        ("Camera", test_camera),
        ("Face Detector", test_face_detector),
        ("Configuration", test_configuration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
