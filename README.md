# ASU AI Semiconductor Summer Institute: Face Recognition Lab

A real-time face recognition application with custom Keras 3 model support, optimized for modern laptops using NPU or GPU acceleration. The application processes video feeds from a webcam and performs face detection and recognition using a Streamlit web interface.

> [!TIP]
> **ASU AI Semi Institute Teachers**: Looking for the step-by-step implementation guide for the summer school program? Check out the [IMPLEMENTATION_GUIDE.md](file:///Users/vinayak/git/AI-Semiconductor-Summer-Institute-Demo/IMPLEMENTATION_GUIDE.md) in the root of this project.

## Features

- **Educational Focus**: Designed to teach computational complexity and silicon (NPU vs GPU vs CPU).
- **Streamlit Web App**: Easy-to-use graphical web interface.
- **Jupyter Training**: Step-by-step model training using a single Jupyter Notebook.
- **Hardware Acceleration**: Optimized for modern laptop NPUs and GPUs, with a CPU fallback.
- **Multiple Detection Methods**: Haar Cascades, DNN-based detection.
- **Face Database**: Store and recognize known faces locally.

## Hardware Requirements

- **Modern Laptop** (Windows, macOS, or Linux)
- **Processor**: NPU or Dedicated GPU recommended. CPU supported as a fallback.
- **Camera**: Built-in webcam or USB webcam

## Software Requirements

- **Python** 3.8+
- **Streamlit**
- **OpenCV**
- **TensorFlow** 2.15+
- **Keras** 3.0+

## Installation

### 1. Install System Dependencies (Linux only)

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libopencv-dev python3-opencv
```

### 2. Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/MPSLab-ASU/AI-Semiconductor-Summer-Institute-Demo.git
cd AI-Semiconductor-Summer-Institute-Demo

# Install Python requirements
pip3 install -r requirements.txt
```

### 3. Verify Hardware Accelerator Setup

```bash
# Check for hardware accelerators via component tests
python3 examples/test_components.py
```

## Configuration

Edit `config/config.yaml` to customize settings:

```yaml
# Camera settings
camera:
  device_id: 0  # Camera device ID (0 for default camera)
  width: 640
  height: 480
  fps: 30

# Face detection method
face_detection:
  method: "haar"  # Options: "haar", "dnn"
  
# Face recognition
face_recognition:
  model_path: "models/face_embeddings_model.keras"
  similarity_threshold: 0.6

# GPU settings
gpu:
  enabled: true
  memory_growth: true
```

## Usage

### 1. Model Training
Train the face embedding model using the provided Jupyter Notebook template:
```bash
jupyter notebook training/train_facenet_template.ipynb
```

### 2. Run the Application
Launch the Streamlit face recognition application:

```bash
# Start the Streamlit web interface
streamlit run src/face_recognition_app.py
```

### Creating a Custom Model

Create your own face embedding model:

```bash
# Create a simple CNN-based model
python3 src/face_recognition/models/create_model.py --type simple --output models/face_embeddings_model.keras

# Create a MobileNet-based model (recommended for Jetson Nano)
python3 src/face_recognition/models/create_model.py --type mobilenet --output models/face_embeddings_model.keras
```

### Managing Known Faces

Use the database management script to add, remove, and manage faces:

```bash
# Add face from image file
python3 examples/manage_database.py add "John Doe" path/to/photo.jpg

# Capture face from camera
python3 examples/manage_database.py capture "Jane Smith"

# List all known faces
python3 examples/manage_database.py list

# Remove a face
python3 examples/manage_database.py remove "John Doe"

# Clear all faces
python3 examples/manage_database.py clear
```

Or programmatically:

```python
from face_recognition.face_recognizer import FaceRecognizer
import cv2

# Initialize recognizer
config = {'model_path': 'models/face_embeddings_model.keras'}
recognizer = FaceRecognizer(config)

# Load an image of a face
image = cv2.imread('path/to/face/image.jpg')

# Add face to database
recognizer.add_face('person_name', image)

# Save database
recognizer.save_database()
```

## Project Structure

```
intel-project/
├── config/
│   └── config.yaml              # Application configuration
├── src/
│   ├── face_recognition/
│   │   ├── __init__.py
│   │   ├── face_detector.py     # Face detection module
│   │   ├── face_recognizer.py   # Face recognition module
│   │   ├── models/
│   │   │   └── create_model.py  # Model creation utilities
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── camera.py        # Camera capture utilities
│   │       └── gpu_utils.py     # GPU configuration utilities
│   └── face_recognition_app.py  # Main application
├── examples/
│   └── run_face_recognition.py  # Example usage
│   └── manage_database.py       # Face database management
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Docker Deployment

Docker is supported for consistent environments:

```bash
# Build the Docker image
docker build -t face-recognition-app .

# Run the container
docker run -p 8501:8501 --device=/dev/video0 face-recognition-app
```

## Performance Optimization

### 1. Utilize NPU/GPU
Monitor your laptop's Activity Monitor or Task Manager to ensure your NPU or GPU is being utilized. This dramatically increases inference speed and lowers power consumption compared to the CPU.

### 2. Optimize Detection
- Use smaller input resolution for face detection.
- Set `skip_frames` in config to process every nth frame.
- Use Haar Cascades for faster detection on lower-end hardware.

## Troubleshooting

### Camera Not Detected

```bash
# List available cameras
ls /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices

# Try different device IDs in config.yaml
```

### Out of Memory Errors

- Enable `memory_growth` in GPU config
- Reduce camera resolution
- Use smaller model architecture
- Process fewer frames (increase `skip_frames`)

### Low FPS

- Enable GPU acceleration
- Use TensorRT optimization
- Reduce input resolution
- Use Haar Cascades instead of DNN
- Enable frame skipping

## Development

### Adding Custom Features

The modular design allows easy extension:

1. **Custom Detection**: Extend `FaceDetector` class
2. **Custom Recognition**: Extend `FaceRecognizer` class
3. **Custom Models**: Use `create_model.py` as template

### Testing

Test individual components:

```bash
# Test camera
python3 -c "from src.face_recognition.utils.camera import test_camera; print(test_camera(0))"

# Test GPU
python3 -c "from src.face_recognition.utils.gpu_utils import get_gpu_info; print(get_gpu_info())"
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built for the ASU AI Semiconductor Summer Institute
- Uses OpenCV and Streamlit for the application interface
- Keras 3 for deep learning models
- Inspired by FaceNet and similar face recognition systems