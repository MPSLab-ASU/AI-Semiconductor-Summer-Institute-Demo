"""
Face Recognition Module — Student Template
============================================
Fill in every section marked  # TODO  to complete the inference pipeline.

Learning objectives
-------------------
After completing this file you will understand:
  1. How to preprocess raw face images for a neural network
  2. How to run a trained model and extract a face embedding
  3. What cosine similarity is and why L2-normalised vectors make it simple
  4. How a nearest-neighbour search over embeddings produces a face ID
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
    Face recognition using a MobileNetV2 embedding model.

    The recognition pipeline is:
        raw face image
            → preprocess_face()    # resize + normalise
            → get_embedding()      # run neural network
            → cosine_similarity()  # compare with stored faces
            → recognize()          # pick the best match
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

        # A directory → TF-TRT SavedModel; a file → regular Keras model
        if os.path.isdir(model_path):
            self._load_trt_model(model_path)
        else:
            self._load_keras_model(model_path)

    def _load_trt_model(self, model_path):
        """Load a TF-TRT SavedModel (optimised for GPU inference)."""
        try:
            import tensorflow as tf

            self.model = tf.saved_model.load(model_path)
            self.inference_fn = self.model.signatures["serving_default"]
            self.is_trt = True
            self.is_tflite = False
            logger.info(f"Loaded TF-TRT SavedModel from {model_path}")
        except Exception as e:
            logger.error(f"Error loading TF-TRT model: {e}")
            self.model = None

    def _load_keras_model(self, model_path):
        """Load a standard Keras model (.keras / .h5)."""
        try:
            import keras

            self.model = keras.models.load_model(model_path)
            logger.info(f"Loaded Keras 3 model from {model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
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

    def preprocess_face(self, face_image, target_size=(160, 160)):
        """
        Prepare a face crop for model input.

        The model was trained on 160×160 RGB images with pixel values in [0, 1].
        OpenCV reads images in BGR format, so we must convert the colour order.

        Args:
            face_image (np.ndarray): Face crop in BGR format, any size.
            target_size (tuple): (width, height) the model expects.

        Returns:
            np.ndarray: Shape (1, height, width, 3), dtype float32, values in [0, 1].
        """
        # WHY: The network was trained at a fixed resolution. We must resize
        #      every face crop to that same size before feeding it in.
        #      On edge devices, we keep the resolution low (like 160x160) to limit
        #      the number of multiplication operations (FLOPs) the GPU must run.
        #
        # TODO 7a: Resize face_image to target_size using cv2.resize.
        #           cv2.resize takes (image, (width, height)).
        face_resized = None  # YOUR CODE HERE

        # WHY: OpenCV stores images as Blue-Green-Red (BGR), but the model was
        #      trained on Red-Green-Blue (RGB) images. Getting this wrong causes
        #      the model to see "wrong" colours and produce bad embeddings.
        #      This colour channel swap is a standard software preprocessing step.
        #
        # TODO 7b: Convert face_resized from BGR to RGB.
        #           Use cv2.cvtColor with the correct conversion code.
        #           HINT: the constant you need is cv2.COLOR_BGR2RGB
        face_rgb = None  # YOUR CODE HERE

        # WHY: Neural networks work best when inputs are small, bounded numbers.
        #      Pixel values run from 0 to 255 (uint8). Dividing by 255.0 maps
        #      them to the range [0.0, 1.0] (float32). This makes calculations
        #      stable and compatible with GPU FP32 precision.
        #
        # TODO 7c: Cast face_rgb to float32 and divide by 255.0 so values are
        #           in the range [0.0, 1.0].
        face_normalized = None  # YOUR CODE HERE

        # WHY: The model expects a *batch* of images, not a single image.
        #      Its input shape is (batch_size, H, W, 3). A single image has
        #      shape (H, W, 3), so we add an extra dimension at position 0
        #      to make it (1, H, W, 3) — a "batch of one".
        #
        # TODO 7d: Add a batch dimension at axis=0 using np.expand_dims.
        #           HINT: np.expand_dims(array, axis=0)
        face_batch = None  # YOUR CODE HERE

        return face_batch

    def get_embedding(self, face_image):
        """
        Run the neural network and return a 128-dimensional face embedding.

        An "embedding" is a compact numeric representation of a face — a list
        of 128 numbers that captures the unique features of that face. Similar
        faces produce similar embeddings; different people produce different ones.

        Args:
            face_image (np.ndarray): Face crop (BGR format, any size).

        Returns:
            np.ndarray: 1-D array of shape (embedding_size,), L2-normalised,
                        or None if the model is not loaded.
        """
        # The TRT path has its own inference function — skip the standard path.
        if getattr(self, "is_trt", False):
            return self._get_trt_embedding(face_image)

        if self.model is None:
            return None

        try:
            # Resize, convert colour, normalise, and add batch dimension.
            face_input = self.preprocess_face(face_image)

            # WHY: model.predict() runs the neural network on the input batch
            #      and returns its output — here a (1, 128) array of raw numbers.
            #      This runs inference. For real-time applications, we want this
            #      to happen in < 30 milliseconds!
            #
            # TODO 8a: Call self.model.predict on face_input.
            #           Pass verbose=0 to suppress progress bar output.
            #           (Printing a progress bar for every video frame ruins CLI output!)
            embedding = None  # YOUR CODE HERE

            # WHY: The output shape is (1, 128) — a batch of one embedding.
            #      We want a 1-D array of shape (128,), so we flatten it.
            #      Think of this 128-D vector as our "Facial Passport".
            #
            # TODO 8b: Flatten the embedding to a 1-D array.
            #           HINT: numpy arrays have a .flatten() method.
            embedding = None  # YOUR CODE HERE

            # WHY: L2 normalisation divides every element by the vector's length
            #      (its L2 norm), projecting it onto the unit sphere (length = 1.0).
            #      This makes comparing embeddings using dot product extremely fast,
            #      since we don't have to divide by vector lengths during real-time matching.
            #
            # TODO 8c: L2-normalise the embedding by dividing it by its norm.
            #           HINT: np.linalg.norm(embedding) returns the L2 norm.
            embedding = None  # YOUR CODE HERE

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

        Cosine similarity ranges from -1 (opposite) to 1 (identical direction).
        A value close to 1 means the two embeddings point in the same direction
        in 128-dimensional space — i.e. they probably belong to the same person.

        Because both embeddings are L2-normalised (unit vectors), the formula
        simplifies beautifully:
            cosine_similarity(a, b) = a · b   (the dot product)

        You do NOT need to divide by the norms — they are both 1 by definition.

        Args:
            embedding1 (np.ndarray): First L2-normalised embedding.
            embedding2 (np.ndarray): Second L2-normalised embedding.

        Returns:
            float: Cosine similarity score in [-1, 1].
        """
        # WHY: Since both embeddings are normalized to a length of 1.0, 
        #      the Cosine Similarity formula simplifies to just the dot product!
        #      This is computationally cheap and runs incredibly fast.
        #
        # TODO 9: Return the dot product of embedding1 and embedding2.
        #         HINT: np.dot(a, b) computes the dot product of two vectors.
        pass  # YOUR CODE HERE

    def recognize(self, face_image):
        """
        Identify who is in the face image.

        Strategy:
          1. Convert the face into a 128-d embedding.
          2. Compare that embedding against every stored embedding using
             cosine similarity.
          3. Return the name of the person whose stored embedding is most
             similar — but only if that similarity exceeds the threshold
             (otherwise the face is treated as "unknown").

        Args:
            face_image (np.ndarray): Face crop (BGR format).

        Returns:
            tuple: (name, confidence) where name is a string (or None if
                   unrecognised) and confidence is the best similarity score.
        """
        if getattr(self, "model", None) is None or not self.known_faces:
            return None, 0

        # Get the embedding for the face we want to identify.
        embedding = self.get_embedding(face_image)

        if embedding is None:
            return None, 0

        # We will keep track of the best (most similar) match found so far.
        best_match = None
        best_similarity = 0

        # self.known_faces is a dict: { "Alice": [emb1, emb2], "Bob": [emb3], ... }
        for name, known_embeddings in self.known_faces.items():
            for known_embedding in known_embeddings:

                # WHY: Compare the current video face embedding with one of the
                #      embeddings in our local database.
                #
                # TODO 10a: Compute the cosine similarity between `embedding`
                #           (the face we want to identify) and `known_embedding`
                #           (one stored embedding for `name`).
                #           Use self.cosine_similarity().
                similarity = None  # YOUR CODE HERE

                # WHY: We perform a "Nearest Neighbor" search. We want to find
                #      which identity matches our target face with the highest score.
                #
                # TODO 10b: If this similarity is greater than best_similarity,
                #           update best_similarity and set best_match = name.
                # YOUR CODE HERE

        # WHY: "Open-set" recognition means a face could be someone outside our
        #      database. If the best similarity is below our threshold (e.g. 0.6),
        #      we flag the face as "Unknown" rather than guessing incorrectly.
        #
        # TODO 10c: If best_similarity is greater than or equal to
        #           self.similarity_threshold, return (best_match, best_similarity).
        #           Otherwise the face is unrecognised — return (None, best_similarity).
        # YOUR CODE HERE

    # -----------------------------------------------------------------------
    # Database helpers — provided for you. No changes needed.
    # -----------------------------------------------------------------------

    def add_face(self, name, face_image):
        """
        Compute an embedding for face_image and store it under `name`.

        Args:
            name (str): Name / identifier for the person.
            face_image (np.ndarray): Face crop (BGR format).

        Returns:
            bool: True if the face was added successfully.
        """
        embedding = self.get_embedding(face_image)

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
