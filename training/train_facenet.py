import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print(f"TensorFlow version: {tf.__version__}")

# ==========================================
# 1. Load Dataset
# ==========================================
print("Fetching LFW dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)

X = lfw_people.images
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print(f"Total images: {len(X)}")
print(f"Number of classes: {n_classes}")

# Preprocess: Normalize to [0, 1] and resize to (224, 224)
X = X.astype('float32') / 255.0
if X.shape[1:3] != (224, 224):
    print("Resizing images to 224x224...")
    # Resize in batches to avoid OOM if dataset is very large, though LFW is small enough
    X = tf.image.resize(X, (224, 224)).numpy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. Triplet Generator
# ==========================================
def create_triplets(X, y, batch_size=32):
    while True:
        anchors = []
        positives = []
        negatives = []
        
        for _ in range(batch_size):
            # Randomly select a person for Anchor and Positive
            random_label = np.random.choice(np.unique(y))
            label_indices = np.where(y == random_label)[0]
            
            # Need at least 2 images for a pair
            if len(label_indices) < 2:
                continue
                
            idx_a, idx_p = np.random.choice(label_indices, 2, replace=False)
            
            # Randomly select a different person for Negative
            negative_label = np.random.choice(np.unique(y[y != random_label]))
            idx_n = np.random.choice(np.where(y == negative_label)[0])
            
            anchors.append(X[idx_a])
            positives.append(X[idx_p])
            negatives.append(X[idx_n])
            
        # Yield as dictionary to match input layer names
        yield (
            {
                "anchor": np.array(anchors), 
                "positive": np.array(positives), 
                "negative": np.array(negatives)
            }, 
            np.zeros((batch_size, 1))
        )

# Define output signature for tf.data
output_signature = (
    {
        "anchor": tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        "positive": tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        "negative": tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
    },
    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
)

# Wrap in tf.data.Dataset
train_gen = tf.data.Dataset.from_generator(
    lambda: create_triplets(X_train, y_train, batch_size=32),
    output_signature=output_signature
)

val_gen = tf.data.Dataset.from_generator(
    lambda: create_triplets(X_val, y_val, batch_size=32),
    output_signature=output_signature
)

# ==========================================
# 3. Model Architecture (Siamese)
# ==========================================
def get_embedding_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    inputs = layers.Input((224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128)(x)
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="embedding_norm")(x)
    
    return keras.Model(inputs, outputs, name="embedding_model")

embedding_model = get_embedding_model()
embedding_model.summary()

# Siamese Network Inputs
anchor_input = layers.Input((224, 224, 3), name="anchor")
positive_input = layers.Input((224, 224, 3), name="positive")
negative_input = layers.Input((224, 224, 3), name="negative")

# Sharing weights
anchor_embedding = embedding_model(anchor_input)
positive_embedding = embedding_model(positive_input)
negative_embedding = embedding_model(negative_input)

# ==========================================
# 4. Triplet Loss Layer
# ==========================================
class TripletLossLayer(layers.Layer):
    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        
    def call(self, inputs):
        anchor, positive, negative = inputs
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        basic_loss = pos_dist - neg_dist + self.margin
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        self.add_loss(loss)
        return loss

loss_layer = TripletLossLayer(margin=0.2)([anchor_embedding, positive_embedding, negative_embedding])

trainable_siamese_model = keras.Model(
    inputs=[anchor_input, positive_input, negative_input], 
    outputs=loss_layer
)

trainable_siamese_model.compile(optimizer='adam')

# ==========================================
# 5. Training
# ==========================================
print("Starting training...")
history = trainable_siamese_model.fit(
    train_gen, 
    steps_per_epoch=20, 
    epochs=10, 
    validation_data=val_gen, 
    validation_steps=5
)

# ==========================================
# 6. Evaluation
# ==========================================
print("\nEvaluating model performance using KNN classifier on embeddings...")

# Extract embeddings
print("Generating embeddings for training and validation sets...")
# Predict in batches to avoid OOM
train_embeddings = embedding_model.predict(X_train, batch_size=32)
val_embeddings = embedding_model.predict(X_val, batch_size=32)

# Train a simple KNN classifier on the embeddings
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(train_embeddings, y_train)

# Predict on validation set
y_pred = knn.predict(val_embeddings)
acc = accuracy_score(y_val, y_pred)

print(f"\n------------------------------------------------")
print(f"Validation Accuracy (1-NN on Embeddings): {acc:.4f}")
print(f"------------------------------------------------\n")

# ==========================================
# 7. Export
# ==========================================
def export_saved_model(model, filename='face_embedding_model'):
    # Export to TensorFlow SavedModel format using Keras 3 export
    model.export(filename)
    print(f"Embedding Model exported to {filename}")

export_saved_model(embedding_model)
