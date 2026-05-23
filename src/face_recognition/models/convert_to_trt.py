"""
TF-TRT Conversion Script for Keras 3 Models

This script converts a Keras 3 model to a TensorFlow-TensorRT (TF-TRT) optimized SavedModel.
It must be run in an environment with TensorFlow (with TensorRT support) installed,
ideally on the target device (Jetson) or a matching container to ensure engine compatibility.
"""
import os
import argparse
import shutil
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def convert_to_tf_trt(input_path, output_path, precision='FP16'):
    """
    Convert Keras model to TF-TRT SavedModel
    
    Args:
        input_path (str): Path to .keras model
        output_path (str): Output directory for SavedModel
        precision (str): 'FP32' or 'FP16'
    """
    print(f"Loading Keras model from {input_path}...")
    
    # Keras 3: Load and export to TF SavedModel first if it's not already one
    # Note: TF-TRT works on SavedModel format.
    
    temp_saved_model_dir = "temp_saved_model"
    
    try:
        import keras
        model = keras.models.load_model(input_path)
        # Export to standard TF SavedModel
        model.export(temp_saved_model_dir) # Keras 3 export
        print(f"Exported to temporary SavedModel at {temp_saved_model_dir}")
    except Exception as e:
        print(f"Keras load failed or export failed: {e}")
        try:
             # Fallback for older Keras/TF compat
            model = tf.keras.models.load_model(input_path)
            tf.saved_model.save(model, temp_saved_model_dir)
            print("Saved using tf.saved_model.save")
        except Exception as e2:
            print(f"FATAL: Could not load/save model: {e2}")
            return

    print(f"Converting to TF-TRT ({precision})...")
    
    # Set conversion params
    conversion_params = trt.TrtConversionParams(
        precision_mode=trt.TrtPrecisionMode.FP16 if precision == 'FP16' else trt.TrtPrecisionMode.FP32
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=temp_saved_model_dir,
        conversion_params=conversion_params
    )

    # Converter requires running the conversion
    converter.convert()
    
    # Saving the engine
    print(f"Saving TRT model to {output_path}...")
    converter.save(output_path)
    
    # Cleanup
    shutil.rmtree(temp_saved_model_dir)
    print("Conversion complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Keras 3 model to TF-TRT')
    parser.add_argument('--input', '-i', required=True, help='Path to .keras model')
    parser.add_argument('--output', '-o', required=True, help='Path to output SavedModel directory')
    parser.add_argument('--precision', '-p', default='FP16', choices=['FP32', 'FP16'], help='Precision mode')
    
    args = parser.parse_args()
    
    convert_to_tf_trt(args.input, args.output, args.precision)
