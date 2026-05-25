# 🎓 Teacher's Implementation Guide: Edge AI & Face Recognition

Welcome to the **ASU AI Semiconductor Summer Institute** Face Recognition Lab! 🚀 (https://sites.google.com/asu.edu/ai-semi-institute/home)

This guide is designed specifically for school teachers for the Day 2 and Day 3 sessions. The core focus of this guide is to understand **AI and its relationship to computational complexity and silicon**. By the end of this lab, you will build a practical face recognition application that you can use in your daily job!

The demo consists of two main parts:
1. **Model Training**, contained entirely within a single Jupyter Notebook.
2. **Local Deployment**, which runs as an easy-to-use Streamlit web application.

You will build and train a neural network that runs entirely on local laptops, ideally targeting an NPU (Neural Processing Unit) or GPU for acceleration, with the CPU acting as a fallback. You will not rely on cloud servers or the internet, ensuring privacy and speed.

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
📂 **Training Notebook**: `training/train_facenet_template.ipynb`

In this section, we will load a dataset of celebrity faces (Labeled Faces in the Wild - LFW), set up a Siamese training pipeline, and train our model.

### 🧮 The Math of Training: Memory & MAC Analysis

Before writing code, we must understand the immense computational scale of training a model compared to merely running it (inference). Let's calculate the computational cost based on Image Size, Batch Size, Dataset Size, and Epochs, including the hidden cost of Backpropagation.

#### 1. The MAC Cost of Training
At the silicon level, the core operation is a **MAC (Multiply and Accumulate)**: $a \times b + c$. For a standard MobileNetV2 model processing a 224x224 image:
*   **Forward Pass:** ~300 Million MACs per image.
*   **Backward Pass (Backprop):** Calculating gradients and updating weights typically costs about **2x** the forward pass, adding ~600 Million MACs per image.
*   **Total per Image:** ~900 Million MACs.

Now, let's scale this up to a full training run:
*   **1 Batch (e.g., 32 images):** $32 \text{ images} \times 900\text{M} = \mathbf{28.8 \text{ Billion MACs}}$
*   **1 Epoch (e.g., 1,000 images):** $1,000 \text{ images} \times 900\text{M} = \mathbf{900 \text{ Billion MACs}}$
*   **Full Training (e.g., 50 Epochs):** $50 \times 900\text{B} = \mathbf{45 \text{ Trillion MACs!}}$

*(Note on Siamese Networks: Because our training uses Triplets (Anchor, Positive, Negative), we process **3 images** for every single training step, effectively tripling the forward pass cost!)*

#### 2. The Memory Cost of Training
Training isn't just mathematically heavier; it is incredibly memory-hungry. During the forward pass of training, the computer must **save all intermediate mathematical activations** in RAM so it can use them later during the backward pass to calculate gradients. If your batch size is 32, you must store the intermediate states for all 32 images simultaneously. This easily consumes several Gigabytes of memory, which is why training often causes "Out of Memory" (OOM) crashes on laptops!

#### 3. Our Solution: Transfer Learning (Freezing Weights)
Performing 45+ Trillion MACs from scratch requires massive data centers. Here is how we make it run on your laptop:
*   **Pre-trained Weights:** We load a MobileNetV2 backbone (2.2 million parameters) that has already been trained on 1.2 million images.
*   **Freezing:** By setting `trainable = False`, we freeze the backbone. **This completely eliminates the backward pass (backprop) for those 2.2 million parameters!** 
*   **The Result:** We only perform the forward pass (300M MACs) and compute gradients for our tiny custom embedding layer (~160,000 parameters). This cuts training time from days to just a few minutes, bypassing massive memory and MAC bottlenecks.

---

### 📝 Step-by-Step Implementation Guide & Code Scaffolds (Part 1)

#### **Section 2: Load & Preprocess Dataset**
We fetch the images and need to format them for MobileNetV2.

> [!NOTE]
> We will gloss over the data loader, but remember: the neural network expects float inputs rather than integers, and images must be resized to a uniform dimension.

##### **TODO 5a: Normalise X**
*   **The Logic**: Raw pixel values are integers in the range `[0, 255]`. Neural networks converge faster and are more stable when inputs are scaled to floating point numbers in `[0.0, 1.0]`.
*   **Code Scaffold**:
    ```python
    X = X.astype("___") / ___
    ```
*   **Cheat Sheet**:
    *   Cast the numpy array `X` to `"float32"` using the `.astype()` method.
    *   Divide the resulting array by `255.0`.

##### **TODO 5b: Resize to 224x224**
*   **The Logic**: MobileNetV2's convolutional layers are hardwired to process images of a specific resolution (224x224). Passing any other size will crash the model.
*   **Code Scaffold**:
    ```python
    if X.shape[1:3] != (224, 224):
        print("Resizing images to 224x224...")
        X = tf.image.resize(X, (___, ___)).___()
    ```
*   **Cheat Sheet**:
    *   Pass the target dimensions `(224, 224)` to `tf.image.resize()`.
    *   Call the `.numpy()` method on the output tensor to convert it back to a NumPy array for compatibility with the rest of the script.

---

#### **Section 3: Triplet Sampling**
We must group images into triplets (Anchor, Positive, Negative) to feed the Siamese network.

##### **TODO 6a: Pick Anchor and Positive Indices**
*   **The Logic**: You have `label_indices`, which lists the locations of all photos belonging to the same person. You need to randomly pick 2 *distinct* photos.
*   **Code Scaffold**:
    ```python
    idx_a, idx_p = np.random.choice(label_indices, size=___, replace=___)
    ```
*   **Cheat Sheet**:
    *   Set `size=2` to get two indices.
    *   Set `replace=False` so that `np.random.choice` does not pick the same index twice (an image cannot be compared with itself).

##### **TODO 6b: Pick a Negative Label**
*   **The Logic**: You must find a person *other than* the current person, and then pick one of their photos.
*   **Code Scaffold**:
    ```python
    # 1. Filter out random_label from y
    other_labels = y[y != ___]
    # 2. Find unique labels in the remaining set
    unique_others = np.unique(___)
    # 3. Randomly choose one label
    negative_label = np.random.choice(___)
    ```
*   **Cheat Sheet**:
    *   `y != random_label` creates a boolean mask that is True only for labels of *different* people.
    *   `np.unique()` extracts the unique names/labels.
    *   `np.random.choice(unique_others)` selects one of those labels.

##### **TODO 6c: Append Triplet Images**
*   **The Logic**: Add the actual image arrays corresponding to our chosen indices to the batch lists.
*   **Code Scaffold**:
    ```python
    anchors.append(X[___])
    positives.append(X[___])
    negatives.append(X[___])
    ```
*   **Cheat Sheet**:
    *   Use the indices `idx_a`, `idx_p`, and `idx_n` to index the dataset array `X`.

---

#### **Section 4: Model Architecture**
We will build the face embedding model using Keras.

##### **TODO 7a: Load MobileNetV2 Backbone**
*   **The Logic**: Instantiate the MobileNetV2 architecture initialized with pre-trained weights, but discard its final 1000-class classification layers.
*   **Code Scaffold**:
    ```python
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(___, ___, ___),
        include_top=___,
        weights="___"
    )
    ```
*   **Cheat Sheet**:
    *   `input_shape` should be `(224, 224, 3)`.
    *   `include_top` must be `False` (removes the default classification head).
    *   `weights` must be `"imagenet"`.

##### **TODO 7b: Freeze the Backbone**
*   **The Logic**: We do not want to change the visual features learned on ImageNet; we only want to train our new projection head.
*   **Code Scaffold**:
    ```python
    base_model.trainable = ___
    ```
*   **Cheat Sheet**:
    *   Set this property to `False`.

##### **TODO 7c: Build the Embedding Head**
*   **The Logic**: Connect the input, preprocessing, backbone, pooling, and Dense projection layers together in a functional pipeline.
*   **Code Scaffold**:
    ```python
    inputs  = layers.Input((224, 224, 3))
    # MobileNetV2 expects input rescaled to [-1, 1], so we scale up X from [0, 1] to [0, 255] first
    x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x       = base_model(x, training=___)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(___)(x)
    outputs = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name="embedding_norm")(x)
    ```
*   **Cheat Sheet**:
    *   Set `training=False` inside the `base_model` call (this ensures Batch Normalization layers don't update their sliding statistics during training).
    *   The `layers.Dense` layer output size should be `128` (our final embedding size).

---

#### **Section 5: Triplet Loss & Training**
Now we implement the custom Triplet Loss mathematical layer.

##### **TODO 8a & 8b: Compute Positive and Negative Squared L2 Distances**
*   **The Logic**: Calculate the squared distance between vectors: $d^2(u, v) = \sum (u_i - v_i)^2$.
*   **Code Scaffold**:
    ```python
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=___)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=___)
    ```
*   **Cheat Sheet**:
    *   Use `tf.square(difference)` to square the values.
    *   Use `tf.reduce_sum(..., axis=-1)` to sum along the embedding dimension (the last axis, which contains the 128 elements).

##### **TODO 8c: Compute Basic Loss**
*   **The Logic**: Determine if the positive is closer than the negative by at least our safety margin.
*   **Code Scaffold**:
    ```python
    basic_loss = pos_dist - neg_dist + self.margin
    ```

##### **TODO 8d: Clamp and Average**
*   **The Logic**: If a triplet is "easy" (loss < 0), we ignore it by clamping it to 0.0. Then, we average the loss across the entire training batch.
*   **Code Scaffold**:
    ```python
    loss = tf.reduce_mean(tf.maximum(basic_loss, ___))
    self.add_loss(loss)
    ```
*   **Cheat Sheet**:
    *   Pass `0.0` to `tf.maximum()` to ensure we do not optimize already-correct triplets.
    *   Call `self.add_loss(loss)` so Keras tracks this loss internally for backpropagation.

##### **TODO 9: Fit the Model**
*   **The Logic**: Start the training process with our custom generator.
*   **Code Scaffold**:
    ```python
    history = trainable_siamese_model.fit(
        train_gen,
        steps_per_epoch=___,
        epochs=___,
        validation_data=___,
        validation_steps=___
    )
    ```
*   **Cheat Sheet**:
    *   `steps_per_epoch=20` (run 20 batches of triplets per epoch).
    *   `epochs=10` (train for 10 epochs total).
    *   `validation_data=val_gen`.
    *   `validation_steps=5` (run 5 validation batches to monitor performance).

---

#### **Section 6: Evaluation**
We measure how well-separated our clusters are using a Nearest-Neighbor classifier.

##### **TODO 10a: Generate Embeddings**
*   **The Logic**: Extract the 128-D vectors for all images in the train and validation sets.
*   **Code Scaffold**:
    ```python
    train_embeddings = embedding_model.predict(X_train, batch_size=___)
    val_embeddings = embedding_model.predict(X_val, batch_size=___)
    ```
*   **Cheat Sheet**:
    *   Set `batch_size=32` to avoid running out of memory (OOM) on low-resource machines.

##### **TODO 10b: Train 1-NN Classifier**
*   **The Logic**: Train a simple database lookup model (Nearest Neighbors) on our training embeddings.
*   **Code Scaffold**:
    ```python
    knn = KNeighborsClassifier(n_neighbors=___, metric="___")
    knn.fit(train_embeddings, ___)
    ```
*   **Cheat Sheet**:
    *   `n_neighbors=1` (check the single nearest database point).
    *   `metric="euclidean"`.
    *   Fit using `train_embeddings` and their corresponding labels `y_train`.

##### **TODO 10c: Predict and Measure Accuracy**
*   **The Logic**: Predict labels for our validation embeddings and check what percentage is correct.
*   **Code Scaffold**:
    ```python
    y_pred = knn.predict(val_embeddings)
    acc = accuracy_score(y_val, ___)
    ```
*   **Cheat Sheet**:
    *   Pass `y_pred` to `accuracy_score()` along with the true labels `y_val`.

---

## 🔌 Part 2: Local Deployment & Hardware Acceleration
📂 **Deployment**: Streamlit Web App
📂 **Coded Solution**: `src/face_recognition_app.py`

Once a model is trained, we need to run it in real-time on your local laptop. This phase is called **inference**.

### ⚙️ Why Run Locally?
*   **Privacy**: Facial data never leaves the device. If you use this for student attendance, no biometric data is sent to the cloud.
*   **Zero Latency**: Real-time video processing requires processing 30 frames per second (about 33ms per frame). Round trips to a cloud server are too slow!
*   **No Internet Required**: The system works in remote school sites, basements, or during network outages.

### 🧮 The Math of Inference: Memory & MAC Analysis
Now that we've seen the massive MAC and memory costs of *training* (Trillions of MACs and Gigabytes of VRAM), let's break down the cost of *inference* (running the model on a single camera frame).

**1. MAC Analysis (Computational Cost)**
For every face detected, the inference pipeline has two stages:
*   **Neural Network Run:** Pushing a 224x224 image through MobileNetV2 to generate the 128-D embedding vector requires exactly **1 Forward Pass** (~300 Million MACs). There is no backpropagation during inference!
*   **Vector Comparison:** Comparing the new 128-D vector to a known student's vector using cosine similarity requires exactly **128 MACs**. For a database of $N$ students, the cost is $128 \times N$ MACs.

**2. Memory Analysis (RAM/VRAM Cost)**
*   Unlike training, which must store all intermediate activations across a whole batch to compute gradients, inference only processes **1 frame at a time (Batch Size = 1)**.
*   Furthermore, the computer can instantly discard layer $L$'s activations as soon as layer $L+1$ is calculated. This drops memory requirements from Gigabytes down to just a few Megabytes!

**The Hardware Insight:**
The neural network extraction (~300 Million MACs) completely dwarfs the database search (e.g., just $1.28$ Million MACs for 10,000 students). This perfectly illustrates why we need an NPU or GPU for the network, while the CPU easily handles the database math. Meanwhile, the incredibly low memory footprint of inference is what allows this application to run smoothly on edge devices and standard laptops!

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

### 🧠 Hardware Architectures: CPU vs. GPU vs. NPU
Understanding the hardware architecture makes these software optimizations click:

| Metric | CPU (Central Processing Unit) | GPU (Graphics Processing Unit) | NPU (Neural Processing Unit) |
|---|---|---|---|
| **Core Count** | Few cores ($4$ to $16$) | Thousands of tiny cores ($128$ to $3000+$) | Dedicated Matrix Multiply blocks (Systolic Arrays) |
| **Core Speed** | Very fast clock speed (sequential tasks) | Slower clock speed (parallel tasks) | Extremely fast at specific math (MAC operations) |
| **Execution** | Processes instructions step-by-step | Processes thousands of general operations at once | Continuous data flow (passes data directly between calculators) |
| **Best Used For** | Running the OS, file operations, web servers | Graphics, video editing, training AI models | Running AI inference (like face recognition) at very low power |

#### 💡 The Edge Advantage: How an NPU works
When deploying AI locally on laptops, the silicon you target drastically affects computational complexity and performance:
1. **CPU (Fallback):** Handles sequential tasks well but is slow for the massive parallel mathematical operations required by neural networks.
2. **Dedicated GPU:** Excellent for parallel matrix operations, but uses high power. Often, moving camera frame data from System RAM to GPU RAM over the PCIe bus creates a bottleneck.
3. **NPU (Neural Processing Unit):** Modern laptops (like Apple Silicon, Snapdragon X, or new Intel Core Ultra/AMD Ryzen AI chips) feature NPUs. While a GPU is a "jack of all trades" for parallel tasks, an NPU is **hardwired specifically for the math of neural networks** (Multiply-Accumulate operations). Instead of reading data from memory for every single calculation, an NPU uses a "systolic array"—a grid where data flows directly from one calculator to the next like an assembly line. This allows NPUs to run AI models incredibly fast while sipping a fraction of the power of a GPU, saving laptop battery life.
4. **Unified Memory Architecture (UMA):** Many of these modern chips also share the same physical memory between the CPU, GPU, and NPU. This allows the NPU to instantly read the camera frame without copying data across a slow bus, drastically increasing inference frame rates!

---

### 📝 Step-by-Step Implementation Guide & Code Scaffolds (Part 2)

#### **Method: `preprocess_face`**
We crop a face out of our video feed and must format it to match what our trained model expects.

##### **TODO 1a: Resize Face Image**
*   **The Logic**: The camera crop can be any size (depending on how close you are to the lens). We must resize it to match the exact input shape of our embedding model.
*   **Code Scaffold**:
    ```python
    face_resized = cv2.resize(face_image, ___)
    ```
*   **Cheat Sheet**:
    *   Pass `target_size` (the second argument of this method, representing width and height) to `cv2.resize()`.

##### **TODO 1b: Convert BGR to RGB**
*   **The Logic**: OpenCV loads and displays images in Blue-Green-Red (BGR) color channel order, but standard neural networks expect Red-Green-Blue (RGB) format. Failing to swap channels will make colors look wrong to the network, resulting in corrupted face passports.
*   **Code Scaffold**:
    ```python
    face_rgb = cv2.cvtColor(face_resized, ___)
    ```
*   **Cheat Sheet**:
    *   Use the conversion constant `cv2.COLOR_BGR2RGB`.

##### **TODO 1c: Normalize to [0.0, 1.0]**
*   **The Logic**: Cast the pixel values to decimal floats and scale them down.
*   **Code Scaffold**:
    ```python
    face_normalized = face_rgb.astype("___") / ___
    ```
*   **Cheat Sheet**:
    *   Cast using `"float32"`.
    *   Divide by `255.0`.

##### **TODO 1d: Add Batch Dimension**
*   **The Logic**: A single camera frame crop has dimensions `(Height, Width, Channels)`. Our model expects a batch dimension: `(BatchSize, Height, Width, Channels)`. We add an extra dimension at index 0 to represent a "batch of 1 image".
*   **Code Scaffold**:
    ```python
    face_batch = np.expand_dims(face_normalized, axis=___)
    ```
*   **Cheat Sheet**:
    *   Set `axis=0` to add the dimension at the start of the shape tuple.

---

#### **Method: `get_embedding`**
We run the preprocessed face through the network to generate its 128-D passport.

##### **TODO 2a: Predict Embedding**
*   **The Logic**: Pass the preprocessed image batch to the model. We explicitly disable logging printouts because showing a progress bar inside a real-time video stream loop will freeze the frame rate.
*   **Code Scaffold**:
    ```python
    embedding = self.model.predict(face_input, verbose=___)
    ```
*   **Cheat Sheet**:
    *   Set `verbose=0`.

##### **TODO 2b: Flatten Array**
*   **The Logic**: The model's prediction returns a 2D batch tensor of shape `(1, 128)`. We need a 1D vector of shape `(128,)` to perform our database math.
*   **Code Scaffold**:
    ```python
    embedding = embedding.___()
    ```
*   **Cheat Sheet**:
    *   Call the `.flatten()` method on the numpy array.

##### **TODO 2c: L2 Normalisation**
*   **The Logic**: Project the embedding onto a unit sphere. If we divide the vector by its mathematical length (L2 norm), the final vector will have a length of exactly $1.0$. This allows us to calculate Cosine Similarity using only a simple dot product later.
*   **Code Scaffold**:
    ```python
    embedding = embedding / np.linalg.norm(___)
    ```
*   **Cheat Sheet**:
    *   Pass `embedding` to `np.linalg.norm()` to calculate its length.

---

#### **Method: `cosine_similarity`**
We measure how similar two facial passports are.

##### **TODO 3: Compute Dot Product**
*   **The Logic**: Since both vectors are L2-normalized, the cosine similarity simplifies to the vector dot product: $\cos(\theta) = u \cdot v$.
*   **Code Scaffold**:
    ```python
    return np.dot(___, ___)
    ```
*   **Cheat Sheet**:
    *   Pass `embedding1` and `embedding2` to `np.dot()`.

---

#### **Method: `recognize`**
We search our database of known teachers/students to find a match.

##### **TODO 4a: Compare to Stored Database**
*   **The Logic**: Compare our unknown face embedding against all embeddings stored in the database.
*   **Code Scaffold**:
    ```python
    similarity = self.cosine_similarity(embedding, ___)
    ```
*   **Cheat Sheet**:
    *   Compare `embedding` (input face) with the current database item `known_embedding` inside the loop.

##### **TODO 4b: Update Best Match**
*   **The Logic**: If the similarity score is the highest we've seen so far, store it along with the person's name.
*   **Code Scaffold**:
    ```python
    if similarity > ___:
        best_similarity = ___
        best_match = ___
    ```
*   **Cheat Sheet**:
    *   Compare `similarity` against `best_similarity`.
    *   Update `best_similarity = similarity`.
    *   Update `best_match = name`.

##### **TODO 4c: Apply Threshold**
*   **The Logic**: If the best match score is below our security threshold, we declare the face "Unknown". This is critical to prevent false recognition of visitors or students not in our class database.
*   **Code Scaffold**:
    ```python
    if best_similarity >= self.___:
        return best_match, best_similarity
    else:
        return ___, best_similarity
    ```
*   **Cheat Sheet**:
    *   Check against `self.similarity_threshold`.
    *   Return `(None, best_similarity)` if below the threshold.

---

## 🛠️ Verification Checklist for Teachers

When testing your final implementations:
1.  **Run the local component tests**:
    This will verify that your camera, dependencies, and hardware accelerators (NPU/GPU) are correctly detected:
    ```bash
    python3 examples/test_components.py
    ```
2.  **Start the Streamlit Application**:
    Launch the web interface to test the real-time inference:
    ```bash
    streamlit run src/face_recognition_app.py
    ```
3.  **Verify NPU/GPU recognition speed**:
    Open your laptop's task manager (Activity Monitor on Mac, Task Manager on Windows) to monitor your silicon's utilization. Verify that your NPU or GPU utilization spikes when running the Streamlit face recognition app, showing that the hardware accelerator is active!
4.  **Tune the threshold**:
    If the system calls a teacher by another teacher's name, increase `similarity_threshold` in `config/config.yaml` to make matching stricter (e.g., to `0.70`). If it fails to recognize known faces, lower it slightly (e.g., to `0.55`).

***

**Good luck running your local Face Recognition Lab!** 🤖✨
