# Face Recognition Application for NVIDIA Jetson Nano

A real-time face recognition application with custom Keras 3 model support, optimized for NVIDIA Jetson Nano with GPU acceleration. The application processes video feed from an attached camera and performs face detection and recognition.

> [!TIP]
> **ASU AI Semi Institute Teachers**: Looking for the step-by-step implementation guide for the summer school program? Check out the [IMPLEMENTATION_GUIDE.md](file:///Users/vinayak/git/intel-project/IMPLEMENTATION_GUIDE.md) in the root of this project.

## Features

- **Real-time Video Processing**: Processes live video feed from attached cameras
- **Custom Keras 3 Models**: Support for custom face embedding models
- **GPU Acceleration**: Optimized for NVIDIA Jetson Nano GPU
- **Multiple Detection Methods**: Haar Cascades, DNN-based detection
- **Face Database**: Store and recognize known faces
- **Flexible Configuration**: YAML-based configuration system
- **Easy Integration**: Modular design for easy customization

## Hardware Requirements

- **NVIDIA Jetson Nano** (4GB recommended)
- **Camera**: USB webcam or CSI camera module
- **Storage**: 16GB+ microSD card
- **Power**: 5V 4A power supply recommended for stable GPU operation

## Software Requirements

- **JetPack SDK** 4.6+ (includes CUDA, cuDNN, TensorRT)
- **Python** 3.8+
- **OpenCV** with CUDA support
- **TensorFlow** 2.15+ (with GPU support)
- **Keras** 3.0+

## Installation

### 1. Setup Jetson Nano

Flash JetPack SDK to your Jetson Nano:
```bash
# Download JetPack SDK from NVIDIA Developer website
# Flash to microSD card using balenaEtcher or NVIDIA SDK Manager
```

### 2. Install System Dependencies

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade

# Install required system packages
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libhdf5-serial-dev hdf5-tools
sudo apt-get install -y libopencv-dev python3-opencv
```

### 3. Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/MPSLab-ASU/intel-project.git
cd intel-project

# Install Python requirements
pip3 install -r requirements.txt

# For Jetson Nano, you may need to install TensorFlow from NVIDIA's repo
# Download pre-built TensorFlow wheel from:
# https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/
# pip3 install tensorflow-2.x.x-cp38-cp38-linux_aarch64.whl
```

### 4. Verify GPU Setup

```bash
# Check CUDA availability
python3 -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
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

### Basic Usage

Run the face recognition application:

```bash
# Using default configuration
python3 src/face_recognition_app.py

# Using custom configuration
python3 src/face_recognition_app.py --config path/to/config.yaml
```

Or use the example script:

```bash
python3 examples/run_face_recognition.py
```

### Controls

- **'q'**: Quit the application
- **'s'**: Save current frame as image

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

### Using Docker (Recommended for Production)

```bash
# Build the Docker image
docker build -t face-recognition-app .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Manual Docker Run

```bash
docker run --runtime=nvidia --privileged \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  --network host \
  face-recognition-app
```

## Performance Optimization for Jetson Nano

### 1. Use TensorRT

For maximum performance, convert your Keras model to TensorRT:

```python
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('models/face_embeddings_model.keras')

# Convert to TensorRT
from tensorflow.python.compiler.tensorrt import trt_convert as trt

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='models/face_embeddings_model',
    precision_mode=trt.TrtPrecisionMode.FP16
)
converter.convert()
converter.save('models/face_embeddings_model_trt')
```

### 2. Optimize Detection

- Use smaller input resolution for face detection
- Set `skip_frames` in config to process every nth frame
- Use Haar Cascades for faster detection on lower-end hardware

### 3. Power Mode

Set Jetson Nano to maximum performance:

```bash
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks   # Max clock speeds
```

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

- Built for NVIDIA Jetson Nano platform
- Uses OpenCV for computer vision
- Keras 3 for deep learning models
- Inspired by FaceNet and similar face recognition systems