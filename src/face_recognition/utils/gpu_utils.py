"""
GPU configuration and utilities for NVIDIA Jetson Nano
"""
import os
import logging

logger = logging.getLogger(__name__)


def configure_gpu(config):
    """
    Configure GPU settings for TensorFlow/Keras on Jetson Nano
    
    Args:
        config (dict): GPU configuration from config file
    """
    if not config.get('enabled', True):
        logger.info("GPU disabled in configuration")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return False
    
    try:
        import tensorflow as tf
        
        # List available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            logger.info(f"Found {len(gpus)} GPU(s)")
            
            # Configure memory growth to avoid allocating all GPU memory at once
            if config.get('memory_growth', True):
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Enabled memory growth for {gpu}")
                    except RuntimeError as e:
                        logger.warning(f"Could not set memory growth: {e}")
            
            # Set specific GPU device if specified
            device_id = config.get('device_id', 0)
            if device_id < len(gpus):
                tf.config.set_visible_devices(gpus[device_id], 'GPU')
                logger.info(f"Using GPU device {device_id}")
            
            return True
        else:
            logger.warning("No GPU detected, running on CPU")
            return False
            
    except ImportError:
        logger.error("TensorFlow not installed")
        return False
    except Exception as e:
        logger.error(f"Error configuring GPU: {e}")
        return False


def get_gpu_info():
    """
    Get information about available GPUs
    
    Returns:
        dict: GPU information
    """
    info = {
        'available': False,
        'count': 0,
        'devices': []
    }
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            info['available'] = True
            info['count'] = len(gpus)
            info['devices'] = [str(gpu) for gpu in gpus]
            
        # Try to get NVIDIA-specific info for Jetson
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                info['devices'].append({
                    'id': i,
                    'name': name,
                    'memory_total': memory.total,
                    'memory_free': memory.free,
                    'memory_used': memory.used
                })
                
            pynvml.nvmlShutdown()
        except:
            pass  # NVML not available or failed
            
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
    
    return info


def check_jetson_nano():
    """
    Check if running on NVIDIA Jetson Nano
    
    Returns:
        bool: True if running on Jetson Nano
    """
    try:
        # Check for Jetson-specific files
        if os.path.exists('/etc/nv_tegra_release'):
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                if 'Jetson' in content:
                    logger.info("Running on NVIDIA Jetson platform")
                    return True
    except:
        pass
    
    return False
