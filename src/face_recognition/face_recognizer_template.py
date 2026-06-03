"""
Face Recognition Module — Solved
============================================
Completed inference pipeline.

Learning objectives
-------------------
After completing this file you will understand:
  1. How to crop, resize, and format raw face images so the computer chip can read them
  2. How to run a trained model and extract a face embedding (our 128-number "facial passport")
  3. How to compare passports using basic multiplications and additions
  4. How to do a simple yearbook lookup (nearest-neighbor search) to match a face
"""

import numpy as np
import cv2
import logging
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Face recognition using a MobileNetV3Small embedding model.

    The recognition pipeline is:
        raw face image
            → _extract_face()       # Get the cropped image of a face
            → preprocess_face()    # resize + scale values
            → get_embedding()      # run model to get 128-number passport
            → cosine_similarity()  # compare passports using multiplication & addition
            → recognize()          # lookup the best match in our database
    """

    def __init__(self, config):
        """
        Initialize face recognizer.

        Args:
            config (dict): Face recognition configuration.
        """
        self.config = config
        self.model = None
        self.embedding_size = config.get("embedding_size", 128)
        self.similarity_threshold = config.get("similarity_threshold", 0.6)
        self.database_path = config.get(
            "database_path", "data/face_database.pkl"
        )

        # Dictionary that maps a person's name to a list of their embeddings.
        # Structure: { "Alice": [embedding_1, embedding_2, ...], "Bob": [...], ... }
        self.known_faces = {}

        self._load_model()
        self._load_database()

    # -----------------------------------------------------------------------
    # Infrastructure — provided for you. No changes needed below this line
    # until the first TODO.
    # -----------------------------------------------------------------------

    def _load_model(self):
        """Dispatch to the correct model loader based on the path type."""
        model_path = self.config.get("model_path")

        if not model_path:
            logger.warning(
                "No model path specified, face recognition disabled"
            )
            return

        if not os.path.exists(model_path):
            logger.warning(
                f"Model not found at {model_path}, face recognition disabled"
            )
            return

        if model_path.endswith('.tflite'):
            self.model_path = model_path
            self._init_litert_model(self.config.get("accelerator", "Auto"))
        else:
            self._load_keras_model(model_path)

    def _init_litert_model(self, requested_accelerator):
        """Initializes the ai_edge_litert CompiledModel with fallback."""
        try:
            import ai_edge_litert as litert
            from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator
        except ImportError:
            logger.error("ai_edge_litert not installed. Inference will fail.")
            self.model = None
            return

        self.active_accelerator = "None"
        
        def try_npu():
            logger.info("Attempting to load model on NPU...")
            if hasattr(HardwareAccelerator, 'NPU'):
                self.model = CompiledModel.from_file(self.model_path, hardware_accel=HardwareAccelerator.NPU)
                self.active_accelerator = "NPU"
            else:
                raise Exception("NPU is not supported by this version of ai_edge_litert.")
            
        def try_gpu():
            logger.info("Attempting to load model on GPU...")
            self.model = CompiledModel.from_file(self.model_path, hardware_accel=HardwareAccelerator.GPU)
            self.active_accelerator = "GPU"
            
        def try_cpu():
            logger.info("Falling back to CPU...")
            self.model = CompiledModel.from_file(self.model_path, hardware_accel=HardwareAccelerator.CPU)
            self.active_accelerator = "CPU"

        try:
            if requested_accelerator == "NPU":
                try_npu()
            elif requested_accelerator == "GPU":
                try_gpu()
            elif requested_accelerator == "CPU":
                try_cpu()
            else:
                try:
                    try_npu()
                except Exception as e:
                    logger.debug(f"NPU init failed: {e}")
                    try:
                        try_gpu()
                    except Exception as e2:
                        logger.debug(f"GPU init failed: {e2}")
                        try_cpu()
        except Exception as e:
            logger.error(f"Failed to load model on {requested_accelerator}: {e}")
            try_cpu()
            self.active_accelerator = f"CPU (Fallback)"
            
        if self.model:
            self.is_litert = True
            self.is_trt = False
            
            try:
                # Use standard LiteRT Interpreter for python inference
                if os.environ.get('DISABLE_TENSORFLOW') == '1':
                    from ai_edge_litert.interpreter import Interpreter
                else:
                    try:
                        from ai_edge_litert.interpreter import Interpreter
                    except ImportError:
                        import tensorflow as tf
                        Interpreter = tf.lite.Interpreter
                
                self.interpreter = Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                self.litert_signature = self.interpreter.get_signature_runner()
                self.litert_sig_name = "serving_default"
            except Exception as e:
                logger.error(f"Failed to initialize LiteRT interpreter: {e}")
                self.model = None
                
            logger.info(f"Loaded LiteRT model on {self.active_accelerator}")

    def _load_keras_model(self, model_path):
        """Load a standard Keras model (.keras / .h5) and convert to LiteRT."""
        try:
            import tensorflow as tf
            import keras
        except ImportError:
            logger.error("TensorFlow and Keras must be installed to load and convert Keras models (.keras / .h5). Please install tensorflow and keras, or use a .tflite model.")
            self.model = None
            return

        try:
            import os

            tflite_path = model_path + ".tflite"
            
            if not os.path.exists(tflite_path):
                # Suppress verbose TF logging during conversion
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                tf.get_logger().setLevel('ERROR')
                
                logger.info(f"Loading Keras model from {model_path} for LiteRT conversion")
                keras_model = keras.models.load_model(model_path)
                
                logger.info("Converting Keras model to TFLite (this may take a moment)...")
                converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
                tflite_model = converter.convert()
                
                with open(tflite_path, "wb") as f:
                    f.write(tflite_model)
                
                logger.info(f"Saved converted TFLite model to {tflite_path}")
            else:
                logger.info(f"Found cached TFLite model at {tflite_path}, skipping conversion.")
            
            self.model_path = tflite_path
            self._init_litert_model(self.config.get("accelerator", "Auto"))
            
        except Exception as e:
            logger.error(f"Error loading Keras model: {e}")
            self.model = None

    def _load_database(self):
        """Deserialise the known-faces dictionary from disk."""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, "rb") as f:
                    self.known_faces = pickle.load(f)
                logger.info(
                    f"Loaded {len(self.known_faces)} known faces from database"
                )
            except Exception as e:
                logger.error(f"Error loading face database: {e}")
                self.known_faces = {}
        else:
            logger.info("No existing face database found, starting fresh")
            self.known_faces = {}

    def save_database(self):
        """Serialise the known-faces dictionary to disk."""
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            with open(self.database_path, "wb") as f:
                pickle.dump(self.known_faces, f)
            logger.info(
                f"Saved face database with {len(self.known_faces)} faces"
            )
            return True
        except Exception as e:
            logger.error(f"Error saving face database: {e}")
            return False

    # -----------------------------------------------------------------------
    # ★ STUDENT SECTION — complete the four functions below ★
    # -----------------------------------------------------------------------

    def _extract_face(self,full_frame,face_coordinates):
        """
        Args:
            full_frame (np.ndarray): The full image frame.
            face_coordinates (tuple): The coordinates of the face (x, y, w, h).

        Returns:
            np.ndarray: A cropped resized face matching training data's aspect ratio
        """
        x,y,w,h = face_coordinates

        # WHY: The LFW dataset used for training has images of size 125x94 (height x width).
        #      When resized to 224x224, faces were stretched horizontally by ~33%.
        #      To match this, we must extract a taller bounding box (aspect ratio 94:125)
        #      around the face's center point.
        #
        # TODO 1a: Unpack the face_coordinates (x, y, w, h), calculate the center point
        #          (center_x, center_y), and calculate target_w and target_h where
        #          target_h = target_w * (125 / 94).
        
        # Find the center of the face
        center_x = None #YOUR CODE HERE 
        center_y = None #YOUR CODE HERE 
        
        # Set the target width
        target_w = None #YOUR CODE HERE 
        
        # Calculate the target height using the aspect ratio of LFW dataset
        target_h = None #YOUR CODE HERE 

        # WHY: The face box center might be close to the edges of the image. We must
        #      ensure our new top-left coordinates (new_x, new_y) are at least 0,
        #      and that the new width and height do not extend past the image boundaries.
        #
        # TODO 1b: Calculate new top-left coordinates new_x and new_y using max(0, ...).
        #          Get frame_h and frame_w from full_frame.shape, then calculate new_w
        #          and new_h by clamping them within the image dimensions using min().

        # Find the max width
        new_x = None #YOUR CODE HERE 
        # Find the max height
        new_y = None #YOUR CODE HERE 
        frame_h, frame_w = full_frame.shape[:2]
        
        # Clamp the width and height to the image boundaries
        new_w = None #YOUR CODE HERE 
        new_h = None #YOUR CODE HERE 

        # WHY: Slicing the NumPy array extracts the target sub-region (the face ROI)
        #      without copying the underlying image data in memory, making it fast.
        #   
        # TODO 1c: Slice full_frame using the calculated new_y, new_h, new_x, and new_w,
        #          and return the cropped face image.
        return None #YOUR CODE HERE 


    def preprocess_face(self, face_image, target_size=(224, 224)):
        """
        Prepare a face crop for model input.

        The model was trained on 224x224 RGB images with pixel values in [0, 1].
        OpenCV reads images in BGR format, so we must convert the color order.

        Args:
            face_image (np.ndarray): Face crop in BGR format, any size.
            target_size (tuple): (width, height) the model expects.

        Returns:
            np.ndarray: Shape (1, height, width, 3), dtype float32, values in [0, 1].
        """

        # WHY: The network was trained on a fixed grid size. Resizing to 224x224
        #      limits the number of multiplications and additions the hardware
        #      needs to calculate per second, saving power and running faster.
        #
        # TODO 2a: Resize face_image to target_size using cv2.resize.
        #           cv2.resize takes (image, (width, height)).
        face_resized = None #YOUR CODE HERE 

        # WHY: OpenCV reads colors in Blue-Green-Red (BGR) order, but our model
        #      expects Red-Green-Blue (RGB). Failing to swap them is like looking
        #      through a filter where red and blue are swapped (a red apple looks blue),
        #      resulting in a corrupted passport.
        #
        # TODO 2b: Convert face_resized from BGR to RGB.
        #           Use cv2.cvtColor with the correct conversion code.
        #           HINT: the constant you need is cv2.COLOR_BGR2RGB
        face_rgb = #YOUR CODE HERE 

        # WHY: Raw pixels are whole numbers from 0 to 255 (representing brightness).
        #      Dividing by 255.0 converts them to decimal numbers between 0.0 and 1.0,
        #      which makes calculations stable and easier for the computer chip to process.
        #
        # TODO 2c: Cast face_rgb to float32 and divide by 255.0 so values are
        #           in the range [0.0, 1.0].
        face_normalized = #YOUR CODE HERE 

        # WHY: The model is designed to process folders (batches) of images. Even if
        #      we only have one face image, we must place it inside a "folder of 1 image"
        #      (expanding the array at axis=0) so the model knows how to read it.
        #
        # TODO 2d: Add a batch dimension at axis=0 using np.expand_dims.
        #           HINT: np.expand_dims(array, axis=0)
        face_batch = #YOUR CODE HERE 

        return face_batch

    def get_embedding(self, full_frame, face_coordinates=None):
        """
        Run the neural network and return a 128-dimensional face embedding.

        An "embedding" is a 128-number "facial passport" list that captures
        unique features of a face. Similar faces produce similar lists of numbers;
        different people produce very different ones.

        Args:
            face_image (np.ndarray): Face crop (BGR format, any size) or full frame if face_coordinates is provided.
            face_coordinates (tuple, optional): (x, y, w, h) bounding box of the face.

        Returns:
            np.ndarray: 1-D array of shape (embedding_size,), L2-normalised,
                        or None if the model is not loaded.
        """

        # Use the extract_face helper to crop the face from the full frame
        if face_coordinates is not None:
            # TODO 2a: Use the extract faces so that only the photo of the face is
            # passed to our model
            face_image = None #YOUR CODE HERE 
        else:
            face_image = full_frame

        # The TRT path has its own inference function — skip the standard path.
        if getattr(self, "is_trt", False):
            return self._get_trt_embedding(face_image)
            
        if getattr(self, "is_litert", False):
            if self.model is None or not hasattr(self, "litert_signature"):
                return None
            try:
                face_input = self.preprocess_face(face_image)
                
                # Execute using LiteRT SignatureRunner
                input_details = self.litert_signature.get_input_details()
                input_name = list(input_details.keys())[0]
                output = self.litert_signature(**{input_name: face_input})
                
                if isinstance(output, dict):
                    embedding = list(output.values())[0][0]
                elif isinstance(output, list) and len(output) > 0:
                    embedding = output[0][0]
                else:
                    embedding = output[0]
                
                embedding = embedding.flatten()
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            except Exception as e:
                logger.error(f"LiteRT inference error: {e}")
                return None

        if self.model is None:
            return None

        try:
            # Resize, convert color, normalize, and add batch dimension.
            face_input = None #YOUR CODE HERE 

            # WHY: Running the model (the guessing phase) takes the processed image
            #      and performs ~300 million multiplications and additions, producing
            #      a list of 128 raw numbers. For real-time video, this must happen in
            #      less than 30 milliseconds!
            #
            # TODO 3b: Call self.model.predict on face_input.
            #           Pass verbose=0 to suppress progress bar output.
            #           (Printing a progress bar for every video frame ruins CLI output!)
            embedding = None #YOUR CODE HERE 

            # WHY: The output is a batch folder containing one passport list.
            #      We flatten it to a single 1-D list of 128 numbers (our Facial Passport).
            #
            # TODO 3c: Flatten the embedding to a 1-D array.
            #           HINT: numpy arrays have a .flatten() method.
            embedding = None #YOUR CODE HERE 

            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def _get_trt_embedding(self, face_image):
        """
        Run inference with a TF-TRT SavedModel (GPU-optimised path).
        Provided for you — no changes needed here.
        """
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow is required for TF-TRT inference.")
            return None

        try:
            face_input = self.preprocess_face(face_image)
            face_tensor = tf.convert_to_tensor(face_input)
            output = self.inference_fn(face_tensor)
            output_key = list(output.keys())[0]
            embedding = output[output_key].numpy()
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            logger.error(f"Error in TRT inference: {e}")
            return None

    def cosine_similarity(self, embedding1, embedding2):
        """
        Measure how similar two face embeddings are.

        Cosine similarity ranges from -1 (opposite) to 1 (identical). When the overall
        mathematical length of both passports is 1.0, finding their similarity simplifies
        to a simple dot product (multiplying the corresponding numbers together and
        adding them up).

        You do NOT need to divide by the norms — they are both 1 by definition.

        Args:
            embedding1 (np.ndarray): First L2-normalised embedding.
            embedding2 (np.ndarray): Second L2-normalised embedding.

        Returns:
            float: Cosine similarity score in [-1, 1].
        """
        # WHY: Since both passports are scaled to a length of 1.0, similarity is just a
        #      dot product (multiply the 128 numbers pairwise and add them up). This takes
        #      only 128 multiplications and additions per person, which runs incredibly fast!
        #      (Note: Our model was trained using Euclidean Distance, but when vectors
        #      are L2-normalized, minimizing Euclidean Distance is mathematically identical
        #      to maximizing Cosine Similarity. We swap to Cosine Similarity here purely
        #      to save CPU cycles during real-time inference!)
        #
        # TODO 4: Return the dot product of embedding1 and embedding2.
        #         HINT: np.dot(a, b) computes the dot product of two vectors.
        return None #YOUR CODE HERE 

    def recognize(self, face_image, face_coordinates=None):
        """
        Identify who is in the face image.

        Strategy:
          1. Convert the face into a 128-number passport.
          2. Compare it to every passport in our database using multiplication and addition.
          3. Return the closest match's name, but only if it's above our security cutoff
             threshold (otherwise label as "Unknown").

        Args:
            face_image (np.ndarray): Face crop (BGR format) or full frame if face_coordinates is provided.
            face_coordinates (tuple, optional): (x, y, w, h) bounding box of the face.

        Returns:
            tuple: (name, confidence) where name is a string (or None if
                   unrecognised) and confidence is the best similarity score.
        """
        if getattr(self, "model", None) is None or not self.known_faces:
            return None, 0

        # Get the embedding for the face we want to identify.
        embedding = self.get_embedding(face_image, face_coordinates)

        if embedding is None:
            return None, 0

        # We will keep track of the best (most similar) match found so far.
        best_match = None
        best_similarity = 0

        # self.known_faces is a dict: { "Alice": [emb1, emb2], "Bob": [emb3], ... }
        for name, known_embeddings in self.known_faces.items():
            for known_embedding in known_embeddings:

                # WHY: Compare the video face passport with one stored in our database.
                #
                # TODO 5a: Compute the cosine similarity between `embedding`
                #           (the face we want to identify) and `known_embedding`
                #           (one stored embedding for `name`).
                #           Use self.cosine_similarity().
                similarity = None #YOUR CODE HERE 

                # WHY: We perform a "yearbook lookup" (Nearest Neighbor search) to find
                #      which stored face matches our target face with the highest similarity score.
                #
                # TODO 5b: If this similarity is greater than best_similarity,
                #           update best_similarity and set best_match = name.
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name

        # WHY: The threshold is our security cutoff. If the best similarity is below this
        #      (e.g., 0.60), we declare the face "Unknown". This is like a security guard
        #      refusing entry if someone's photo ID doesn't look at least 60% similar to them.
        #
        # TODO 5c: If best_similarity is greater than or equal to
        #           self.similarity_threshold, return (best_match, best_similarity).
        #           Otherwise the face is unrecognised — return (None, best_similarity).
        None #YOUR CODE HERE 

    # -----------------------------------------------------------------------
    # Database helpers — provided for you. No changes needed.
    # -----------------------------------------------------------------------

    def add_face(self, name, face_image, face_coordinates=None):
        """
        Compute an embedding for face_image and store it under `name`.

        Args:
            name (str): Name / identifier for the person.
            face_image (np.ndarray): Face crop (BGR format).

        Returns:
            bool: True if the face was added successfully.
        """
        embedding = self.get_embedding(face_image,face_coordinates)

        if embedding is None:
            return False

        if name not in self.known_faces:
            self.known_faces[name] = []

        self.known_faces[name].append(embedding)
        logger.info(
            f"Added face for '{name}' (total: {len(self.known_faces[name])} embeddings)"
        )
        return True

    def remove_face(self, name):
        """
        Remove all embeddings for `name` from the database.

        Args:
            name (str): Name / identifier to remove.

        Returns:
            bool: True if the name was found and removed.
        """
        if name in self.known_faces:
            del self.known_faces[name]
            logger.info(f"Removed face '{name}' from database")
            return True
        return False

    def list_known_faces(self):
        """
        Return the list of names currently stored in the database.

        Returns:
            list: List of known face names.
        """
        return list(self.known_faces.keys())
