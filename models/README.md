# Models Directory

This directory contains the face embedding models used for recognition.

## Current Model

We use a custom **MobileNetV2** model trained with **Triplet Loss** (Metric Learning).
- **File**: `face_embedding_model` (Directory)
- **Format**: TensorFlow SavedModel (ready for TF-TRT conversion)
- **Input**: (1, 224, 224, 3) - RGB images, normalized [0, 1]
- **Output**: (1, 128) - 128-dimensional L2-normalized embedding

## Pre-trained Models

You can use pre-trained models from:
- FaceNet
- ArcFace
- VGGFace

Just convert them to Keras 3 format and ensure they meet the requirements above.

## DNN Detection Models (Optional)

If using DNN-based face detection, place the models here:
- `face_detection/deploy.prototxt`
- `face_detection/res10_300x300_ssd_iter_140000.caffemodel`

Download from OpenCV's repository or use custom models.
