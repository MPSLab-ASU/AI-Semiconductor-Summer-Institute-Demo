"""
Example Keras 3 Model for Face Embeddings

This script demonstrates how to create a custom Keras 3 model for generating
face embeddings that can be used with the face recognition application.
"""
import keras
from keras import layers, models
import numpy as np


def create_facenet_like_model(input_shape=(160, 160, 3), embedding_size=128):
    """
    Create a FaceNet-like model for face embeddings
    
    This is a simplified version for demonstration. For production use,
    consider using pre-trained models like FaceNet, ArcFace, or training
    on large face datasets.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        embedding_size (int): Size of output embedding vector
        
    Returns:
        keras.Model: Face embedding model
    """
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Initial convolution block
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Convolutional blocks
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Global pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Embedding layer (no activation for embeddings)
    embeddings = layers.Dense(embedding_size, name='embeddings')(x)
    
    # L2 normalization for cosine similarity
    embeddings = layers.Lambda(lambda x: keras.ops.l2_normalize(x, axis=1), name='l2_normalize')(embeddings)
    
    model = models.Model(inputs=inputs, outputs=embeddings, name='face_embedding_model')
    
    return model


def create_mobilenet_based_model(input_shape=(160, 160, 3), embedding_size=128):
    """
    Create a MobileNetV2-based model for face embeddings
    
    This uses transfer learning with MobileNetV2, which is efficient for
    edge devices like Jetson Nano.
    
    Args:
        input_shape (tuple): Input image shape
        embedding_size (int): Size of output embedding vector
        
    Returns:
        keras.Model: Face embedding model
    """
    # Load MobileNetV2 as base (without top layers)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers for transfer learning
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Preprocess for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Embedding layer
    embeddings = layers.Dense(embedding_size, name='embeddings')(x)
    
    # L2 normalization
    embeddings = layers.Lambda(lambda x: keras.ops.l2_normalize(x, axis=1), name='l2_normalize')(embeddings)
    
    model = models.Model(inputs=inputs, outputs=embeddings, name='mobilenet_face_embedding')
    
    return model


def save_example_model(model_path='models/face_embeddings_model.keras', model_type='simple'):
    """
    Create and save an example model
    
    Args:
        model_path (str): Path to save the model
        model_type (str): Type of model ('simple' or 'mobilenet')
    """
    import os
    
    print(f"Creating {model_type} face embedding model...")
    
    if model_type == 'mobilenet':
        model = create_mobilenet_based_model()
    else:
        model = create_facenet_like_model()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Print model summary
    model.summary()
    
    # Test the model with random input
    print("\nTesting model with random input...")
    test_input = np.random.rand(1, 160, 160, 3).astype('float32')
    output = model.predict(test_input, verbose=0)
    print(f"Output shape: {output.shape}")
    print(f"Output (first 10 values): {output[0][:10]}")
    print(f"Output L2 norm: {np.linalg.norm(output[0]):.4f} (should be ~1.0)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create example face embedding model')
    parser.add_argument(
        '--output', '-o',
        default='models/face_embeddings_model.keras',
        help='Output path for model file'
    )
    parser.add_argument(
        '--type', '-t',
        choices=['simple', 'mobilenet'],
        default='simple',
        help='Model architecture type'
    )
    
    args = parser.parse_args()
    
    save_example_model(args.output, args.type)
