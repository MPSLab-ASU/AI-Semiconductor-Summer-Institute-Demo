#!/usr/bin/env python3
"""
Face Database Management Script

This script helps manage the face recognition database:
- Add faces from images
- Remove faces
- List known faces
- Clear database
"""
import sys
from pathlib import Path
import argparse
import cv2
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from face_recognition.face_recognizer import FaceRecognizer
from face_recognition.face_detector import FaceDetector


def load_config(config_path='config/config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def add_face_from_image(recognizer, detector, image_path, name):
    """
    Add a face from an image file
    
    Args:
        recognizer: FaceRecognizer instance
        detector: FaceDetector instance
        image_path: Path to image file
        name: Name/identifier for the face
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    # Detect faces in image
    faces = detector.detect(image)
    
    if len(faces) == 0:
        print(f"Error: No face detected in {image_path}")
        return False
    
    if len(faces) > 1:
        print(f"Warning: Multiple faces detected, using the largest one")
        # Use the largest face
        faces = [max(faces, key=lambda f: f[2] * f[3])]
    
    # Extract face region
    x, y, w, h = faces[0]
    face_roi = image[y:y+h, x:x+w]
    
    # Add to database
    if recognizer.add_face(name, face_roi):
        print(f"✓ Added face for '{name}' from {image_path}")
        return True
    else:
        print(f"✗ Failed to add face for '{name}'")
        return False


def add_face_from_camera(recognizer, detector, name, camera_id=0):
    """
    Add a face by capturing from camera
    
    Args:
        recognizer: FaceRecognizer instance
        detector: FaceDetector instance
        name: Name/identifier for the face
        camera_id: Camera device ID
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return False
    
    print(f"\nCapturing face for '{name}'")
    print("Position your face in the frame and press SPACE to capture")
    print("Press 'q' to cancel")
    
    captured = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = detector.detect(frame)
        
        # Draw bounding boxes
        display_frame = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(display_frame, "Press SPACE to capture, 'q' to cancel", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Capture Face', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key
            if len(faces) == 0:
                print("No face detected, try again")
                continue
            
            # Use the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = frame[y:y+h, x:x+w]
            
            if recognizer.add_face(name, face_roi):
                print(f"✓ Captured and added face for '{name}'")
                captured = True
            else:
                print(f"✗ Failed to add face")
            break
            
        elif key == ord('q'):
            print("Cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return captured


def list_faces(recognizer):
    """List all known faces"""
    faces = recognizer.list_known_faces()
    
    if not faces:
        print("No faces in database")
        return
    
    print(f"\nKnown faces ({len(faces)}):")
    for i, name in enumerate(faces, 1):
        count = len(recognizer.known_faces[name])
        print(f"  {i}. {name} ({count} embedding{'s' if count > 1 else ''})")


def remove_face(recognizer, name):
    """Remove a face from database"""
    if recognizer.remove_face(name):
        print(f"✓ Removed '{name}' from database")
        return True
    else:
        print(f"✗ '{name}' not found in database")
        return False


def clear_database(recognizer):
    """Clear all faces from database"""
    confirm = input("Are you sure you want to clear all faces? (yes/no): ")
    if confirm.lower() == 'yes':
        recognizer.known_faces = {}
        print("✓ Database cleared")
        return True
    else:
        print("Cancelled")
        return False


def main():
    parser = argparse.ArgumentParser(description='Face Database Management')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Add face from image
    add_parser = subparsers.add_parser('add', help='Add face from image')
    add_parser.add_argument('name', help='Name/identifier for the face')
    add_parser.add_argument('image', help='Path to image file')
    
    # Add face from camera
    capture_parser = subparsers.add_parser('capture', help='Add face from camera')
    capture_parser.add_argument('name', help='Name/identifier for the face')
    capture_parser.add_argument('--camera', type=int, default=0,
                               help='Camera device ID (default: 0)')
    
    # List faces
    subparsers.add_parser('list', help='List all known faces')
    
    # Remove face
    remove_parser = subparsers.add_parser('remove', help='Remove a face')
    remove_parser.add_argument('name', help='Name/identifier to remove')
    
    # Clear database
    subparsers.add_parser('clear', help='Clear all faces from database')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize components
    recognizer = FaceRecognizer(config['face_recognition'])
    detector = FaceDetector(config['face_detection'])
    
    # Execute command
    if args.command == 'add':
        if add_face_from_image(recognizer, detector, args.image, args.name):
            recognizer.save_database()
            
    elif args.command == 'capture':
        if add_face_from_camera(recognizer, detector, args.name, args.camera):
            recognizer.save_database()
            
    elif args.command == 'list':
        list_faces(recognizer)
        
    elif args.command == 'remove':
        if remove_face(recognizer, args.name):
            recognizer.save_database()
            
    elif args.command == 'clear':
        if clear_database(recognizer):
            recognizer.save_database()


if __name__ == '__main__':
    main()
