# 🎓 Teacher's Implementation Guide: Edge AI & Face Recognition

Welcome to the **ASU AI Semi Institute** Face Recognition Lab! 🚀 

This guide is designed for high school teachers to implement a state-of-the-art, local face recognition system. You will build and train a neural network that runs entirely on local hardware, without relying on cloud servers or the internet.

---

## 🧭 Learning Map & Core Concepts

Before we write code, let's look at the big picture. Traditional image models do **classification** (e.g., predicting "cat", "dog", or "car"). However, classification doesn't work well for face recognition because:
1. **New people are added constantly**: You don't want to retrain a massive network every time a new student joins your class.
2. **We have very few photos**: You might only have 1 or 2 reference photos per student, but classification requires thousands of photos per person.

Instead, we use **Metric Learning** (specifically **FaceNet**).

### 💡 The Analogy Sandbox

Here are the key concepts explained using simple, classroom-friendly analogies:

#### 1. Face Embeddings (The "Facial Passport")
> **Analogy**: Imagine a passport that doesn't show your photo, but instead lists $128$ precise numbers describing your facial structure (e.g., distance between eyes, width of nose, height of forehead). 
> 
> A neural network acts as the passport officer. It takes a raw picture and translates it into this **128-dimensional vector** (a list of 128 decimal numbers). If two pictures are of the same person, their facial passports will contain very similar numbers.

#### 2. Siamese Networks (The "Shared Brain")
> **Analogy**: Imagine three identical clone workers working at three desks. They share the exact same brain, memory, and skills.
> 
> During training, we feed three images simultaneously: an **Anchor** (a target person's photo), a **Positive** (another photo of the same person), and a **Negative** (a photo of a different person). The three branches of our Siamese network process these images. Since they share the same weights ("shared brain"), what one branch learns about distinguishing eyes or noses is instantly known by the others.

```
                  ┌─────────────────┐
Anchor (A)   ────▶│ Embedding Model │────▶ f(A) ───┐
                  └─────────────────┘              │
                  ┌─────────────────┐              ▼
Positive (P) ────▶│  (Shared Brain) │────▶ f(P) ────────▶ [ Triplet Loss ]
                  └─────────────────┘              ▲
                  ┌─────────────────┐              │
Negative (N) ────▶│  (Shared Brain) │────▶ f(N) ───┘
                  └─────────────────┘
```

#### 3. Triplet Loss (The "Rubber Band Rule")
> **Analogy**: Imagine three pins on a corkboard representing the Anchor, Positive, and Negative embeddings.
> - We hook an elastic rubber band between the **Anchor** and **Positive** (pulling them close).
> - We place a stiff wooden peg between the **Anchor** and **Negative** (pushing them apart by at least a safety margin).
> 
> Triplet Loss calculates the mathematical tension: if the positive photo is far away or the negative photo is too close, the network receives a "penalty" and adjusts its weights to fix it.

---

## 💻 Part 1: Model Training & Computational Costs
📂 **Template File**: `training/train_facenet_template.ipynb`  
📂 **Coded Solution**: `training/train_facenet.ipynb`

In this section, we will load a dataset of celebrity faces (Labeled Faces in the Wild - LFW), set up a Siamese training pipeline, and train our model.

### ⚡ Understanding Computational Costs
Training a model from scratch is computationally expensive. Here is why we use **Transfer Learning** and how we control computational costs:

*   **Model Parameter Size**: A model's size is determined by its parameters (weights and biases). A large model like ResNet-50 has over $25$ million parameters. Our backbone, **MobileNetV2**, has only **$2.2$ million parameters**—making it roughly $10\times$ smaller and ideal for local edge hardware!
*   **Transfer Learning**: Instead of starting with a blank network, we use a MobileNetV2 backbone pre-trained on ImageNet (a massive dataset of 1.2 million general images). It already knows how to detect edges, curves, and textures.
*   **Freezing Weights**: By setting `trainable = False` on the backbone, we "freeze" those 2.2 million parameters. During training, we only compute mathematical adjustments (gradients) for our final face embedding layer (only about $160,000$ parameters). This cuts training time on a typical computer from hours to just a couple of minutes!

---

### 📝 Step-by-Step Implementation Guide (Part 1)

#### **Section 2: Load & Preprocess Dataset**
We fetch the images and need to format them for MobileNetV2.

> [!NOTE]
> We will gloss over the data loader, but remember: the neural network expects float inputs rather than integers, and images must be resized to a uniform dimension.

*   **TODO 5a: Normalise X**
    *   **Goal**: Scale pixel values from raw integers `[0, 255]` to floats `[0.0, 1.0]`.
    *   **Guidance**: Convert the numpy array `X` to `'float32'` using the `.astype()` method, and then divide the entire array by the maximum pixel value `255.0`.
*   **TODO 5b: Resize to 224x224**
    *   **Goal**: Make sure all images match MobileNetV2's expected input dimensions.
    *   **Guidance**: If the image shape (`X.shape[1:3]`) is not already `(224, 224)`, use the TensorFlow function `tf.image.resize(X, (224, 224))` and convert it back to a numpy array using `.numpy()`.

---

#### **Section 3: Triplet Sampling**
We must group images into triplets (Anchor, Positive, Negative) to feed the Siamese network.

*   **TODO 6a: Pick Anchor and Positive Indices**
    *   **Goal**: Choose two random, different images of the same person.
    *   **Guidance**: You have `label_indices` (all photos of the chosen person). Use `np.random.choice()` with `size=2` and make sure `replace=False` so you don't pick the exact same photo twice. Unpack the two resulting indices into `idx_a` and `idx_p`.
*   **TODO 6b: Pick a Negative Label**
    *   **Goal**: Choose a random person who is *not* the current person.
    *   **Guidance**: Filter out the current `random_label` from the target labels `y` (e.g., `y[y != random_label]`). Extract the unique labels from this filtered array using `np.unique()`, and then use `np.random.choice()` to select one label at random.
*   **TODO 6c: Append Triplet Images**
    *   **Goal**: Add the selected images to their respective lists.
    *   **Guidance**: Retrieve the three images from the dataset `X` using the indices you found (`idx_a`, `idx_p`, and `idx_n`). Append them to the `anchors`, `positives`, and `negatives` lists.

---

#### **Section 4: Model Architecture**
We will build the face embedding model using Keras.

```
Input (224, 224, 3) ──▶ [Preprocess] ──▶ [MobileNetV2 (Frozen)] ──▶ [Global Pooling] ──▶ [Dense (128)] ──▶ [L2 Norm] ──▶ 128-D Vector
```

*   **TODO 7a: Load MobileNetV2 Backbone**
    *   **Goal**: Instantiate the pre-trained feature extractor.
    *   **Guidance**: Call `tf.keras.applications.MobileNetV2()`. Provide it with:
        *   `input_shape=(224, 224, 3)`
        *   `include_top=False` (this discards the original 1000-class classifier head)
        *   `weights='imagenet'` (to load the pre-trained weights)
*   **TODO 7b: Freeze the Backbone**
    *   **Goal**: Prevent updates to the backbone weights.
    *   **Guidance**: Set the `trainable` property of your `base_model` to `False`.
*   **TODO 7c: Build the Embedding Head**
    *   **Goal**: Connect the layers to map the backbone output to a 128-D vector.
    *   **Guidance**: Create a chain of Keras functional layers:
        1.  Define the `inputs` layer using `layers.Input()` with shape `(224, 224, 3)`.
        2.  Preprocess the input by multiplying `inputs * 255.0` (scaling back to raw pixels) and passing it to `tf.keras.applications.mobilenet_v2.preprocess_input()`.
        3.  Pass the preprocessed output to `base_model`, explicitly setting `training=False` to keep Batch Normalization stats frozen.
        4.  Collapse spatial dimensions using `layers.GlobalAveragePooling2D()`.
        5.  Create a dense projection layer using `layers.Dense(128)` (no activation function).
        6.  Create the final `outputs` by normalizing the dense vector. Use a Lambda layer containing `lambda x: tf.math.l2_normalize(x, axis=1)`. Set the layer name to `"embedding_norm"`.
        7.  Return `keras.Model(inputs, outputs)`.

---

#### **Section 5: Triplet Loss & Training**
Now we implement the custom Triplet Loss mathematical layer.

$$\mathcal{L} = \max\Big(\|\text{Anchor} - \text{Positive}\|^2 - \|\text{Anchor} - \text{Negative}\|^2 + \alpha,\ 0\Big)$$

*   **TODO 8a & 8b: Compute Positive and Negative Squared L2 Distances**
    *   **Goal**: Find the squared distance between embeddings.
    *   **Guidance**: Compute the element-wise difference between the embeddings (e.g., `anchor - positive` or `anchor - negative`). Square the difference using `tf.square()`. Sum up the squared values along the last axis using `tf.reduce_sum(..., axis=-1)`.
*   **TODO 8c: Compute Basic Loss**
    *   **Goal**: Combine the distances and apply the margin $\alpha$.
    *   **Guidance**: Subtract `neg_dist` from `pos_dist`, and add the margin `self.margin` (which represents $\alpha$, set to $0.2$).
*   **TODO 8d: Clamp and Average**
    *   **Goal**: Ignore "easy" triplets (where loss is negative) and compute the average loss.
    *   **Guidance**: Clamp any values less than 0 to 0 using `tf.maximum(basic_loss, 0.0)`. Find the batch average using `tf.reduce_mean()`. Register this value by calling `self.add_loss(loss)`, and then return `loss`.
*   **TODO 9: Fit the Model**
    *   **Goal**: Start the training loop.
    *   **Guidance**: Call `trainable_siamese_model.fit()` with:
        *   `train_gen` as the training dataset.
        *   `steps_per_epoch=20` (number of batches processed per epoch).
        *   `epochs=10` (total passes through the training data).
        *   `validation_data=val_gen` (validation dataset).
        *   `validation_steps=5` (batches to check for validation).

---

#### **Section 6: Evaluation**
We measure how well-separated our clusters are using a Nearest-Neighbor classifier.

*   **TODO 10a: Generate Embeddings**
    *   **Goal**: Run the trained embedding model on our train and validation splits.
    *   **Guidance**: Call `embedding_model.predict()` on `X_train` and `X_val` respectively, passing `batch_size=32` to avoid running out of memory (OOM). Store these in `train_embeddings` and `val_embeddings`.
*   **TODO 10b: Train 1-NN Classifier**
    *   **Goal**: Create a database classifier to lookup embeddings.
    *   **Guidance**: Create an instance of `KNeighborsClassifier` setting `n_neighbors=1` and `metric='euclidean'`. Fit this classifier using `train_embeddings` and `y_train`.
*   **TODO 10c: Predict and Measure Accuracy**
    *   **Goal**: Calculate performance on the validation set.
    *   **Guidance**: Run `knn.predict(val_embeddings)` to get predictions, and compare them to `y_val` using `accuracy_score(y_val, y_pred)`.

---

## 🔌 Part 2: Local Deployment & Hardware Acceleration
📂 **Template File**: `src/face_recognition/face_recognizer_template.py`  
📂 **Coded Solution**: `src/face_recognition/face_recognizer.py`

Once a model is trained, we need to run it in real-time on local hardware (like an NVIDIA Jetson Nano or local computer). This phase is called **inference**.

### ⚙️ Why Run Locally?
*   **Privacy**: Facial data never leaves the device. If you use this for student attendance, no biometric data is sent to the cloud.
*   **Zero Latency**: Real-time video processing requires processing 30 frames per second (about 33ms per frame). Round trips to a cloud server are too slow!
*   **No Internet Required**: The system works in remote school sites, basements, or during network outages.

### 🏎️ Software Optimizations (Translation & Compression)
Standard neural network models are saved in formats designed for editing and training (like Keras `.keras` or TensorFlow SavedModel). To run them quickly on edge hardware, we translate and optimize them:

#### 1. Layer Fusion
Standard training frameworks treat every mathematical step as a separate block (e.g., Convolution $\rightarrow$ Activation $\rightarrow$ Pooling). 
A software optimizer like **NVIDIA TensorRT** combines these steps into a single instruction set, saving time spent moving data back and forth in memory.

```
[Training Model]    Conv2D ──▶ ReLU ──▶ MaxPooling2D
                     ▼         ▼           ▼
[Optimized Model]   └──────────Fused Layer─────────┘
```

#### 2. Quantization (Bit-Width Reduction)
Computers represent neural network weights as 32-bit floats (FP32). 
*   **FP16 (Half-Precision)**: Reduces values to 16-bit floats. This cuts memory usage in half and runs twice as fast on modern GPUs with almost zero loss in accuracy.
*   **INT8 (8-bit Integer)**: Converts decimals to basic 8-bit integers (values 0-255). This runs extremely fast on hardware accelerators but requires careful scaling to maintain accuracy.

---

### 🧠 Hardware Architectures: CPU vs. GPU
Understanding the hardware architecture makes these software optimizations click:

| Metric | CPU (Central Processing Unit) | GPU (Graphics Processing Unit) |
|---|---|---|
| **Core Count** | Few cores ($4$ to $16$) | Thousands of tiny cores ($128$ to $3000+$) |
| **Core Speed** | Very fast clock speed (sequential tasks) | Slower clock speed (parallel tasks) |
| **Execution** | Processes instructions step-by-step | Processes thousands of operations at once |
| **Best Used For** | Running the OS, file operations, web servers | Matrix multiplications (images, neural networks) |

#### 💡 Unified Memory (The Jetson Advantage)
On standard computers, the CPU and GPU have separate RAM blocks. When processing a camera frame:
1. The CPU reads the frame from the camera into System RAM.
2. The CPU copies the frame over a slow bus (PCIe) to GPU RAM.
3. The GPU runs the model and copies the results back to System RAM.

This copying process is a massive bottleneck. The **NVIDIA Jetson Nano** uses a **Unified Memory Architecture (UMA)** where the CPU and GPU share the exact same physical memory block. This allows the GPU to instantly read the camera frame without copying any data, drastically increasing frame rates!

---

### 📝 Step-by-Step Implementation Guide (Part 2)

#### **Method: `preprocess_face`**
We crop a face out of our video feed and must format it to match what our trained model expects.

*   **TODO 1a: Resize Face Image**
    *   **Goal**: Match the input dimensions of the model.
    *   **Guidance**: Use `cv2.resize()`. It expects `(image, (width, height))`. Pass `face_image` and `target_size` (usually `(160, 160)` or `(224, 224)` depending on model configuration).
*   **TODO 1b: Convert BGR to RGB**
    *   **Goal**: Match the color channel order of the model.
    *   **Guidance**: OpenCV reads cameras in Blue-Green-Red (BGR) order, but models are trained on Red-Green-Blue (RGB). Use `cv2.cvtColor()` with the conversion flag `cv2.COLOR_BGR2RGB`.
*   **TODO 1c: Normalize to [0.0, 1.0]**
    *   **Goal**: Match the scale of pixel values used during training.
    *   **Guidance**: Cast the RGB image array to `'float32'` and divide by `255.0`.
*   **TODO 1d: Add Batch Dimension**
    *   **Goal**: Change the input shape from `(H, W, 3)` to `(1, H, W, 3)`.
    *   **Guidance**: Standard models expect a "batch" of images. Use `np.expand_dims()` at `axis=0` to create a "batch of 1 image".

---

#### **Method: `get_embedding`**
We run the preprocessed face through the network to generate its 128-D passport.

*   **TODO 2a: Predict Embedding**
    *   **Goal**: Pass the face crop to the neural network.
    *   **Guidance**: Call `self.model.predict()`, passing the preprocessed image batch (`face_input`). Pass `verbose=0` to prevent Keras from printing console logs on every single frame!
*   **TODO 2b: Flatten Array**
    *   **Goal**: Convert the batch output shape `(1, 128)` to a 1D vector `(128,)`.
    *   **Guidance**: Use the numpy `.flatten()` method on the embedding array.
*   **TODO 2c: L2 Normalisation**
    *   **Goal**: Project the embedding onto a unit sphere.
    *   **Guidance**: Divide the flattened embedding by its L2 Norm. Use `np.linalg.norm(embedding)` to compute the norm.

---

#### **Method: `cosine_similarity`**
We measure how similar two facial passports are.

*   **TODO 3: Compute Dot Product**
    *   **Goal**: Calculate the similarity score ($[-1.0, 1.0]$).
    *   **Guidance**: Because our vectors are L2-normalized (length of 1.0), the cosine similarity is simply their dot product! Compute and return `np.dot(embedding1, embedding2)`.

---

#### **Method: `recognize`**
We search our database of known teachers/students to find a match.

*   **TODO 4a: Compare to Stored Database**
    *   **Goal**: Loop through all known templates and compute similarities.
    *   **Guidance**: Inside the nested loops, call `self.cosine_similarity()`, passing the input face's `embedding` and the current `known_embedding` from the database.
*   **TODO 4b: Update Best Match**
    *   **Goal**: Keep track of the highest score.
    *   **Guidance**: Check if `similarity` is greater than `best_similarity`. If it is, update `best_similarity` to the new score and assign `best_match = name`.
*   **TODO 4c: Apply Threshold**
    *   **Goal**: Filter out unknown people.
    *   **Guidance**: Compare `best_similarity` to `self.similarity_threshold` (e.g., $0.6$). If the score is higher or equal, return the tuple `(best_match, best_similarity)`. Otherwise, return `(None, best_similarity)` to mark them as "Unknown".

---

## 🛠️ Verification Checklist for Teachers

When testing your final implementations:
1.  **Run the local component tests**:
    ```bash
    python3 examples/test_components.py
    ```
2.  **Verify GPU recognition speed**:
    Open a terminal and run `tegrastats` on the Jetson Nano. Verify that GPU utilization spikes when running face recognition, showing that the hardware accelerator is active!
3.  **Tune the threshold**:
    If the system calls a teacher by another teacher's name, increase `similarity_threshold` in `config/config.yaml` to make matching stricter (e.g., to `0.70`). If it fails to recognize known faces, lower it slightly (e.g., to `0.55`).

***

**Good luck running your local Face Recognition Lab!** 🤖✨
