"""
Face Recognition Application with Camera Video Feed Support
"""
import cv2
import yaml
import logging
import argparse
import sys
import os
import streamlit as st
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
            
        self.config_path = config_path
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
                    val = fd[key]
                    if val.startswith('src/') or val.startswith('src\\'):
                        val = val[4:]
                    fd[key] = str(src_dir / val)
                    
            fr = config.get('face_recognition', {})
            for key in ['model_path', 'database_path']:
                if key in fr and not os.path.isabs(fr[key]):
                    val = fr[key]
                    if val.startswith('src/') or val.startswith('src\\'):
                        val = val[4:]
                    fr[key] = str(src_dir / val)
                    
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

    def _inference_worker(self):
        """Background thread worker that runs heavy ML inference continuously."""
        import time
        while getattr(self, 'running', False):
            frame = getattr(self, 'current_frame_for_inference', None)
            if frame is not None:
                # Detect
                faces = self.detector.detect(frame)
                names = []
                for (x, y, w, h) in faces:
                    name, confidence = self.recognizer.recognize(frame,face_coordinates=(x, y, w, h))
                    names.append((name, confidence))
                
                # Safely update cached results
                self.last_faces = faces
                self.last_names = names
            
            # Rate limit inference thread to ~10 FPS to save CPU
            time.sleep(0.1)

    def process_frame(self, frame):
        """
        Process a single frame for rendering (does NOT run inference).
        Draws bounding boxes based on the latest background inference results.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (Processed frame with annotations, List of detected names)
        """
        detected_names = []
        
        # Safely get current cached detections
        faces = getattr(self, 'last_faces', [])
        names = getattr(self, 'last_names', [])
        
        # Process each detected face using cached bounding boxes and labels
        for (x, y, w, h), (name, confidence) in zip(faces, names):
            if name:
                detected_names.append(name)
                
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), self.bbox_color, self.thickness)
            
            # Prepare label
            if name:
                label = f"{name}"
                if getattr(self, 'show_confidence', True):
                    label += f" ({confidence:.2f})"
            else:
                label = "Unknown"
                if getattr(self, 'show_confidence', True) and confidence > 0:
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
        
        return frame, detected_names
    
    def run(self):
        """
        Run the face recognition application using Streamlit
        """
        st.set_page_config(page_title="Classroom Attendance App", layout="wide")
        st.title("Classroom Attendance App")
        
        mode = st.sidebar.radio("Mode:", ["Attendance", "Manage Students", "Settings"])
        
        if mode == "Attendance":
            st.header("Live Classroom Attendance")
            st.write("Point the webcam at the classroom entrance.")
            
            if not getattr(self, '_initialized', False):
                if not self.initialize():
                    st.error("Initialization failed. Check logs.")
                    return
                self._initialized = True
                
            if not self.recognizer or not self.recognizer.known_faces:
                st.warning("No students in the database. Please add students in 'Manage Students' mode first.")
            else:
                detector_method = self.config.get('face_detection', {}).get('method', 'haar').upper()
                st.info(f"🚀 Active Inference Engine: **{getattr(self.recognizer, 'active_accelerator', 'Unknown')}** | 👁️ Face Detector: **{detector_method}** | 🎯 Matching Threshold: **{self.recognizer.similarity_threshold:.2f}**")
            
            if "camera_running" not in st.session_state:
                st.session_state.camera_running = False
                
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Camera", use_container_width=True):
                    st.session_state.camera_running = True
                    st.rerun()
            with col2:
                if st.button("Stop Camera", use_container_width=True, type="primary"):
                    st.session_state.camera_running = False
                    if self.camera:
                        self.camera.release()
                    self._initialized = False
                    st.rerun()
            
            frame_placeholder = st.empty()
            present_students_placeholder = st.empty()
            
            present_students = set()
            
            if st.session_state.camera_running:
                self.running = True
                
                # Start background inference thread
                self.last_faces = []
                self.last_names = []
                self.current_frame_for_inference = None
                
                import threading
                inference_thread = threading.Thread(target=self._inference_worker)
                inference_thread.daemon = True
                inference_thread.start()
                
                try:
                    while self.running:
                        ret, frame = self.camera.read()
                        if not ret:
                            st.error("Failed to read from webcam.")
                            break
                            
                        # Share the latest frame with the inference thread
                        self.current_frame_for_inference = frame.copy()
                        self.last_frame = frame.copy()
                        
                        processed_frame, detected_names = self.process_frame(frame)
                        
                        for name in detected_names:
                            present_students.add(name)
                        
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB")
                        
                        with present_students_placeholder.container():
                            st.subheader("Present Students:")
                            for student in present_students:
                                st.write(f"✅ {student}")
                                
                except Exception as e:
                    # Ignore the Event loop closed error triggered by Streamlit interrupt
                    if "Event loop is closed" not in str(e):
                        st.error(f"Error: {e}")
                finally:
                    self.running = False
                    if self.camera:
                        self.camera.release()
                    self._initialized = False

        elif mode == "Manage Students":
            st.header("Manage Student Database")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Add Student")
                new_name = st.text_input("Student Name")
                input_method = st.radio("Input Method", ["Upload Photos", "Take Photos (Camera)"])
                images_data = []
                
                if input_method == "Upload Photos":
                    st.info("Please upload exactly 3 photos of the student at different angles.")
                    uploaded_files = st.file_uploader("Choose 3 photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
                    if uploaded_files and len(uploaded_files) == 3:
                        images_data = [f.getvalue() for f in uploaded_files]
                    elif uploaded_files and len(uploaded_files) != 3:
                        st.warning(f"Please upload exactly 3 photos. You uploaded {len(uploaded_files) if uploaded_files else 0}.")
                else:
                    st.info("Please take 3 photos of the student at different angles.")
                    cam1 = st.camera_input("Photo 1 (Front)", key="cam1")
                    cam2 = st.camera_input("Photo 2 (Slightly Left)", key="cam2")
                    cam3 = st.camera_input("Photo 3 (Slightly Right)", key="cam3")
                    
                    if cam1 and cam2 and cam3:
                        images_data = [cam1.getvalue(), cam2.getvalue(), cam3.getvalue()]
                        
                if st.button("Save Student") and new_name and len(images_data) == 3:
                    import numpy as np
                    embeddings = []
                    
                    if not getattr(self, '_initialized', False):
                        self.initialize()
                        self._initialized = True
                        
                    for i, bytes_data in enumerate(images_data):
                        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                        faces = self.detector.detect(cv2_img)
                        if len(faces) == 0:
                            st.error(f"No face detected in photo {i+1}!")
                        else:
                            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                            embedding = self.recognizer.get_embedding(cv2_img, (x, y, w, h))
                            if embedding is not None:
                                embeddings.append(embedding)
                                
                    if len(embeddings) == 3:
                        if new_name not in self.recognizer.known_faces:
                            self.recognizer.known_faces[new_name] = []
                        self.recognizer.known_faces[new_name].extend(embeddings)
                        self.recognizer.save_database()
                        st.success(f"Successfully saved {new_name} to database using 3 photos!")
                        
            with col2:
                st.subheader("Delete Student")
                if not getattr(self, '_initialized', False):
                    self.initialize()
                    self._initialized = True
                    
                student_names = self.recognizer.list_known_faces()
                if student_names:
                    student_to_delete = st.selectbox("Select a student to remove", student_names)
                    if st.button("Remove"):
                        self.recognizer.remove_face(student_to_delete)
                        self.recognizer.save_database()
                        st.success(f"Removed {student_to_delete} from database.")
                        st.rerun()
                else:
                    st.info("No students in database.")

        elif mode == "Settings":
            st.header("Settings")
            if "settings_success_msg" in st.session_state:
                st.success(st.session_state.settings_success_msg)
                del st.session_state.settings_success_msg
                
            st.subheader("Engine Configuration")
            current_method = self.config.get('face_detection', {}).get('method', 'haar')
            det_method = st.selectbox("Detection Method", ["haar", "dnn"], index=0 if current_method == 'haar' else 1)
            
            sim_thresh = st.slider("Matching Threshold", 0.0, 1.0, 
                float(self.config.get('face_recognition', {}).get('similarity_threshold', 0.6)))
                
            if st.button("Save Configuration"):
                self.config['face_detection']['method'] = det_method
                self.config['face_recognition']['similarity_threshold'] = sim_thresh
                if getattr(self, 'recognizer', None):
                    self.recognizer.similarity_threshold = sim_thresh
                    
                # Write back to disk
                try:
                    import yaml
                    with open(self.config_path, 'r') as f:
                        disk_config = yaml.safe_load(f)
                    
                    disk_config['face_detection']['method'] = det_method
                    disk_config['face_recognition']['similarity_threshold'] = float(sim_thresh)
                    
                    with open(self.config_path, 'w') as f:
                        yaml.dump(disk_config, f, default_flow_style=False)
                except Exception as e:
                    logger.error(f"Failed to save config to disk: {e}")
                    
                st.session_state.settings_success_msg = "Configuration updated!"
                st.rerun()
                
            st.markdown("---")
            st.subheader("Update AI Engine Model")
            st.write("Upload a new `.keras` or `.h5` model to replace the current facial recognition engine.")
            uploaded_model = st.file_uploader("Upload a model file", type=["keras", "h5", "tflite"])
            
            if uploaded_model is not None:
                if st.button("Apply New Model"):
                    src_dir = Path(__file__).parent
                    models_dir = src_dir / "models"
                    models_dir.mkdir(exist_ok=True)
                    
                    model_filename = "custom_model." + uploaded_model.name.split('.')[-1]
                    model_abspath = models_dir / model_filename
                    
                    with open(model_abspath, "wb") as f:
                        f.write(uploaded_model.getbuffer())
                        
                    model_relpath = os.path.join("models", model_filename)
                    self.config['face_recognition']['model_path'] = model_relpath
                    
                    # Write back to disk
                    try:
                        import yaml
                        with open(self.config_path, 'r') as f:
                            disk_config = yaml.safe_load(f)
                        
                        if 'face_recognition' not in disk_config:
                            disk_config['face_recognition'] = {}
                        disk_config['face_recognition']['model_path'] = model_relpath
                        
                        with open(self.config_path, 'w') as f:
                            yaml.dump(disk_config, f, default_flow_style=False)
                    except Exception as e:
                        logger.error(f"Failed to save config to disk: {e}")
                        
                    # Clear initialization to reload model
                    self._initialized = False
                    st.session_state.settings_success_msg = "New model successfully applied and converted to LiteRT!"
                    st.rerun()
    
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
