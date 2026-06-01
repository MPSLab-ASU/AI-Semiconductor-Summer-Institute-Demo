# 🎓 Teacher's Implementation Guide: Edge AI & Face Recognition

Welcome to the **ASU AI Semiconductor Summer Institute** Face Recognition Lab! 🚀 (https://sites.google.com/asu.edu/ai-semi-institute/home)

This guide is designed specifically for school teachers for the Day 2 and Day 3 sessions. The core focus of this guide is to understand **AI and its relationship to computational complexity and silicon**. By the end of this lab, you will build a practical face recognition application that you can use in your daily job!

The demo consists of two main parts:
1. **Model Training**, contained entirely within a single **Jupyter Notebook** (an interactive web-based document that allows you to run and test code one step at a time).
2. **Local Deployment**, which runs as an easy-to-use **Streamlit web application** (a framework that lets us build a simple website interface using Python).

You will build and train a neural network that runs entirely on local laptops, targeting dedicated hardware accelerators for speed (like a GPU or NPU) with your standard processor (CPU) acting as a fallback. You will not rely on cloud servers or the internet, ensuring privacy and speed.

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

### 🧮 The Math of Training: Multiplications, Additions, and Memory

Before writing code, we must understand the immense scale of mathematical calculations needed to train a model compared to merely running it (inference). Let's calculate the computational cost based on Image Size, Batch Size, Dataset Size, and Epochs (rounds of training).

#### 1. The Math Calculations (Multiplications & Additions)
At the computer chip level, every operation is a simple combination of **multiplication and addition** (e.g., $a \times b + c$, sometimes called a Multiply-Accumulate or MAC operation). For a standard model processing a single 224x224x3 image (a color image of width = height = 224 pixels) :
*   **The Guessing Phase (Forward Pass):** The model takes an image and makes a prediction. This requires about **300 Million** multiplications and additions.
*   **The Learning Phase (Backward Pass):** The model checks if its guess was right, calculates its mistakes, and goes backward to adjust its internal settings. This "learning backward" process is much harder, costing about **2x** the guessing phase, adding **600 Million** multiplications and additions.
*   **Total per Image:** **900 Million** multiplications and additions!

Now, let's scale this up to a full training session:
*   **1 Batch (e.g., 32 images processed together):** $32 \text{ images} \times 900\text{ Million} = \mathbf{28.8 \text{ Billion}}$ multiplications and additions.
*   **1 Epoch (e.g., 1,000 images):** $1,000 \text{ images} \times 900\text{ Million} = \mathbf{900 \text{ Billion}}$ multiplications and additions.
*   **Full Training (e.g., 50 rounds/epochs):** $50 \text{ rounds} \times 900\text{ Billion} = \mathbf{45 \text{ Trillion}}$ multiplications and additions!

*(Note on Siamese Networks: Because our training uses Triplets (Anchor, Positive, Negative), we process **3 images** for every single training step, effectively tripling the guessing phase cost!)*

#### 2. The Memory Cost: The "Scratchpad" Analogy
Training isn't just mathematically heavy; it is incredibly memory-hungry. Why?

> 📝 **The Scratchpad Analogy**: 
> Imagine you are solving a massive, multi-step math problem. 
> - **Running the model (Inference)** is like using a calculator to get a final result. You only need to remember the current number on the screen. As soon as you perform the next step, you can discard the previous one. This requires almost no memory (RAM).
> - **Training (Learning)** is like taking a test where you **must show all your work**. To figure out where you made a mistake at the very end and fix it, you must keep every single line of intermediate calculations written down on a scratchpad. 
> 
> If you have a batch size of 32, it's like keeping the step-by-step scratchpads of 32 students active at the exact same time. This takes up a huge amount of table space (RAM/Memory). If you run out of table space, the computer crashes with an "Out of Memory" (OOM) error!

#### 3. Our Solution: Transfer Learning (Freezing the Brain)
Performing 45+ Trillion calculations from scratch would take days on a standard laptop. Here is how we make it run in minutes:
*   **Pre-trained Brain:** We load a MobileNetV2 model that has already been trained on 1.2 million images. It already has 2.2 million pre-tuned parameters (settings).
*   **Freezing:** By setting `trainable = False`, we lock these 2.2 million parameters. **This completely eliminates the need to calculate adjustments (the 600 Million backward-pass calculations) for this large part of the model!**
*   **The Result:** We only perform the guessing phase (300 Million calculations) and only calculate adjustments for our tiny, custom face-matching layer (~160,000 parameters). This slashes training time from days to just a few minutes, bypassing memory and calculation bottlenecks.

---

### 📝 Step-by-Step Implementation Guide & Code Scaffolds (Part 1)

#### **Section 1: Setup Environment**
Before starting the notebook TODOs, confirm that the lab environment is ready.

*   Install the tested Python dependencies from `requirements.txt`.
*   Run the component tests (`python3 examples/test_components.py`) before editing the notebook.
*   Open `training/train_facenet_template.ipynb` and run the setup cell to import TensorFlow, Keras, NumPy, Matplotlib, and scikit-learn.

---

#### **Section 2: Load & Preprocess Dataset**
We fetch the images and format them for our model.

> [!NOTE]
> We will gloss over the data loader, but remember: the neural network expects decimal inputs rather than whole numbers, and images must be resized to a uniform dimension.

##### **TODO 1a: Normalise X**
*   **The Logic**: Raw pixel values are whole numbers from `0` to `255` (representing pixel brightness). Neural networks learn faster and are more stable when these inputs are scaled to decimal numbers between `0.0` and `1.0`.
*   **Code Scaffold**:
    ```python
    X = X.astype("___") / ___
    ```
*   **Cheat Sheet**:
    *   Cast the numpy array `X` to `"float32"` (representing decimal numbers) using the `.astype()` method.
    *   Divide the resulting array by `255.0`.

##### **TODO 1b: Resize to 224x224**
*   **The Logic**: The model's image processing layers are hardwired to process images of a specific grid size (224 pixels wide by 224 pixels high). Passing any other size will crash the model.
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

#### **Section 3: Triplet Sampling (Selecting Groups of 3 Images)**
We must group images into triplets (Anchor, Positive, Negative) to feed the Siamese network.

##### **TODO 2a: Pick Anchor and Positive Indices**
*   **The Logic**: You have `label_indices`, which lists the locations of all photos belonging to the same person. You need to randomly pick 2 *distinct* photos.
*   **Code Scaffold**:
    ```python
    idx_a, idx_p = np.random.choice(label_indices, size=___, replace=___)
    ```
*   **Cheat Sheet**:
    *   Set `size=2` to get two indices.
    *   Set `replace=False` so that `np.random.choice` does not pick the same index twice (an image cannot be compared with itself).

##### **TODO 2b: Pick a Negative Label**
*   **The Logic**: You must find a person *other than* the current person, and then pick one of their photos.
*   **Code Scaffold**:
    ```python
    # 1. Filter out the current person (random_label) from our list of names (y)
    other_labels = y[y != ___]
    # 2. Find unique names in the remaining set
    unique_others = np.unique(___)
    # 3. Randomly choose one name
    negative_label = np.random.choice(___)
    ```
*   **Cheat Sheet**:
    *   `y != random_label` creates a filter of True/False values that isolates everyone *except* the target person.
    *   `np.unique()` extracts the unique names/labels.
    *   `np.random.choice(unique_others)` selects one of those labels.

##### **TODO 2c: Append Triplet Images**
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

#### **Section 4: Model Architecture (Building the Model)**
We will build the face embedding model using Keras.

##### **TODO 3a: Load MobileNetV2 Backbone**
*   **The Logic**: Load the pre-made MobileNetV2 architecture. Think of this as the **main visual cortex** of the model—it has already been trained on 1.2 million images and knows how to recognize shapes, lines, and textures. We discard its default "classification head" (the final layers that guess category labels like "cat" or "dog") because we want to output a custom facial passport instead.
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

##### **TODO 3b: Freeze the Backbone**
*   **The Logic**: We want to keep the general vision features already learned by the backbone. By freezing it, we tell the computer not to waste energy adjusting any of the 2.2 million parameters in this section.
*   **Code Scaffold**:
    ```python
    base_model.trainable = ___
    ```
*   **Cheat Sheet**:
    *   Set this property to `False`.

##### **TODO 3c: Build the Embedding Head**
*   **The Logic**: Connect the input, preprocessing, backbone, pooling, and Dense projection layers together in a functional pipeline. Add a small custom layer to the end of the backbone whose job is to shrink the backbone's complex visual data into our final 128-number facial passport list.
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

##### **TODO 4a & 4b: Compute Positive and Negative Squared Distances**
*   **The Logic**: Calculate the straight-line distance between two facial passports. This is like finding the distance between two points on a graph: for each of the 128 numbers, find the difference, square it (to remove negative signs), and add them all up.
*   **Code Scaffold**:
    ```python
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=___)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=___)
    ```
*   **Cheat Sheet**:
    *   Use `tf.square(difference)` to square the values.
    *   Use `tf.reduce_sum(..., axis=-1)` to sum along the embedding dimension (the last axis, which contains the 128 elements).

##### **TODO 4c: Compute Basic Loss**
*   **The Logic**: Determine if the positive is closer than the negative by at least our safety margin.
*   **Code Scaffold**:
    ```python
    basic_loss = pos_dist - neg_dist + self.margin
    ```

##### **TODO 4d: Clamp and Average**
*   **The Logic**: If a triplet is already correct (the positive is close and the negative is far), the loss is less than zero. We clamp it to `0.0` (ignore it) so the model doesn't waste effort adjusting weights for things it already knows. We then average the remaining errors across the batch.
*   **Code Scaffold**:
    ```python
    loss = tf.reduce_mean(tf.maximum(basic_loss, ___))
    self.add_loss(loss)
    ```
*   **Cheat Sheet**:
    *   Pass `0.0` to `tf.maximum()` to ensure we do not optimize already-correct triplets.
    *   Call `self.add_loss(loss)` so Keras tracks this loss internally for backpropagation.

##### **TODO 5: Fit the Model**
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

#### **Section 6: Evaluation (Testing the Model)**
We measure how well-separated our face clusters are using a **Nearest-Neighbor classifier (1-NN)**. Think of this as a simple yearbook lookup: it takes a new face passport and searches our database to find the single closest passport, matching the identity.

##### **TODO 6a: Generate Embeddings**
*   **The Logic**: Extract the 128-number facial passports for all images in the train and validation sets.
*   **Code Scaffold**:
    ```python
    train_embeddings = embedding_model.predict(X_train, batch_size=___)
    val_embeddings = embedding_model.predict(X_val, batch_size=___)
    ```
*   **Cheat Sheet**:
    *   Set `batch_size=32` to avoid running out of memory (OOM) on low-resource machines.

##### **TODO 6b: Train 1-NN Classifier**
*   **The Logic**: Train a simple yearbook lookup model (Nearest Neighbors) on our training face passports.
*   **Code Scaffold**:
    ```python
    knn = KNeighborsClassifier(n_neighbors=___, metric="___")
    knn.fit(train_embeddings, ___)
    ```
*   **Cheat Sheet**:
    *   `n_neighbors=1` (check the single nearest database point).
    *   `metric="euclidean"`.
    *   Fit using `train_embeddings` and their corresponding labels `y_train`.

##### **TODO 6c: Predict and Measure Accuracy**
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

### 🧮 The Math of Inference: Multiplications, Additions, and Memory
Now let's break down the cost of *inference* (running the model on a single camera frame).

**1. Multiplications & Additions (Computational Cost)**
For every face detected, the pipeline has two stages:
*   **Neural Network Run:** Pushing a 224x224 image through our model to generate the 128-number passport requires exactly **1 Guessing Phase (Forward Pass)** (~300 Million multiplications and additions). There is no learning backward pass during inference!
*   **Passport Comparison:** Comparing the new 128-number passport to a known person's passport in our database is super fast: we just multiply each of the 128 numbers together and add up the results (128 multiplications and additions per person in our database). For a database of $N$ students, the cost is $128 \times N$ calculations.

**2. Memory Analysis (RAM Cost)**
*   Unlike training, which must store all intermediate math steps across a batch of 32 images (requiring Gigabytes of space), inference only processes **1 frame at a time**.
*   Furthermore, the computer can instantly throw away the math steps for a layer as soon as it computes the next layer. This drops memory requirements from Gigabytes down to just a few Megabytes!

**The Hardware Insight:**
Running the neural network to get the passport (~300 Million calculations) completely dwarfs the database search (e.g., just $1.28$ Million calculations for a database of 10,000 students). This explains why we want a GPU or NPU for the network, while the CPU easily handles the database math. The low memory footprint of inference is why it can run smoothly on standard laptops!

### 🏎️ Software Optimizations (Translation & Compression)
Standard neural network models are saved in formats designed for editing and training (like Keras `.keras`). To run them quickly on edge hardware, we optimize them:

#### 1. Layer Fusion
Standard training frameworks treat every mathematical step as a separate block (e.g., Convolution $\rightarrow$ Activation $\rightarrow$ Pooling).

> 🍰 **The Baking Analogy**: Imagine baking a cake. Instead of preheating the oven, mixing the ingredients, and greasing the pan in separate, slow, back-and-forth trips to different rooms, you group all your ingredients and tools together at once. Combining these mathematical steps reduces the time the computer spends moving numbers back and forth in memory (bypassing memory bandwidth bottlenecks).

```
[Training Model]    Conv2D ──▶ ReLU ──▶ MaxPooling2D
                     ▼         ▼           ▼
[Optimized Model]   └──────────Fused Layer─────────┘
```

#### 2. Quantization (Decimal-to-Integer Simplification)
Computers usually represent neural network settings as 32-bit decimal numbers (FP32), which are highly precise but take up more memory and space.

> 📏 **The Measurement Analogy**: Imagine measuring classroom desks. Instead of writing down a length as `1.52483 meters` (which is highly detailed but takes up a lot of space and makes the math slower to compute), you round it to `1.5 meters` (16-bit floats) or even `2 meters` (8-bit integers). Doing math with rounded whole numbers is incredibly fast for computer chips, and for face recognition, it is still accurate enough to tell faces apart!
*   **FP16 (Half-Precision)**: Reduces values to 16-bit decimals. This cuts memory usage in half and runs twice as fast on modern GPUs with almost zero loss in accuracy.
*   **INT8 (8-bit Integer)**: Converts decimals to basic 8-bit integers (values 0-255). This runs extremely fast on hardware accelerators but requires careful scaling to maintain accuracy.

---

### 🧠 Hardware Architectures: CPU vs. GPU vs. NPU
Understanding the hardware architecture makes these software optimizations click:

| Metric | CPU (Central Processing Unit) | GPU (Graphics Processing Unit) | NPU (Neural Processing Unit) |
|---|---|---|---|
| **Analogy** | **A few super-smart professors** ($4$ to $16$ cores) who can solve any complex problem, but work step-by-step. | **A stadium of elementary students** (thousands of cores) doing basic math all at the exact same time. | **A specialized assembly line** designed specifically to do matrix multiplication math at maximum speed. |
| **Core Speed** | Very fast clock speed (excellent for single, sequential tasks) | Slower clock speed (designed to run thousands of calculations in parallel) | Hardwired specifically for model calculations (multiplications and additions) |
| **Execution** | Processes instructions step-by-step | Processes thousands of general operations at once | Continuous data flow (passes numbers directly between calculators in a grid) |
| **Best Used For** | Running the OS, file operations, web servers | Graphics, video editing, training AI models | Running AI inference (like face recognition) at very low power |

#### 💡 The Edge Advantage: How an NPU works
When deploying AI locally on laptops, the silicon you target drastically affects how fast your model runs:
1. **CPU (Fallback):** Handles step-by-step tasks well but is slow for the massive parallel mathematical operations required by neural networks.
2. **Dedicated GPU:** Excellent for parallel operations, but consumes high power. Often, moving camera frame data from the computer's memory (RAM) to the GPU's memory creates a slow traffic bottleneck.
3. **NPU (Neural Processing Unit):** Modern laptops (like Apple Silicon, Snapdragon X, or new Intel/AMD AI chips) feature NPUs. While a GPU is a generalist for parallel tasks, an NPU is **hardwired specifically for neural network math**. Instead of reading and writing data to memory for every single calculation, an NPU uses a grid where data flows directly from one calculator to the next like a physical assembly line (called a "systolic array"). This allows NPUs to run AI models incredibly fast while sipping a fraction of the power of a GPU, saving laptop battery life.
4. **Unified Memory Architecture (UMA):** Many of these modern chips also share the exact same physical memory pool between the CPU, GPU, and NPU. This allows the NPU to instantly read the camera frame without copying data across a slow internal bus, drastically increasing speed!

---

### 📝 Step-by-Step Implementation Guide & Code Scaffolds (Part 2)

#### **Method: `preprocess_face`**
We crop a face out of our video feed and must format it to match what our trained model expects.

##### **TODO 7a: Resize Face Image**
*   **The Logic**: The camera crop can be any size (depending on how close you are to the lens). We must resize it to match the exact input shape of our embedding model.
*   **Code Scaffold**:
    ```python
    face_resized = cv2.resize(face_image, ___)
    ```
*   **Cheat Sheet**:
    *   Pass `target_size` (the second argument of this method, representing width and height) to `cv2.resize()`.

##### **TODO 7b: Convert BGR to RGB**
*   **The Logic**: OpenCV loads and displays images in Blue-Green-Red (BGR) color order, but standard neural networks expect Red-Green-Blue (RGB). Failing to swap channels will make colors look wrong to the network. It's like looking through a filter where red and blue are swapped (a red apple looks blue), resulting in corrupted face passports.
*   **Code Scaffold**:
    ```python
    face_rgb = cv2.cvtColor(face_resized, ___)
    ```
*   **Cheat Sheet**:
    *   Use the conversion constant `cv2.COLOR_BGR2RGB`.

##### **TODO 7c: Normalize to [0.0, 1.0]**
*   **The Logic**: Cast the pixel values to decimal floats and scale them down.
*   **Code Scaffold**:
    ```python
    face_normalized = face_rgb.astype("___") / ___
    ```
*   **Cheat Sheet**:
    *   Cast using `"float32"`.
    *   Divide by `255.0`.

##### **TODO 7d: Add Batch Dimension**
*   **The Logic**: A single camera frame crop has dimensions `(Height, Width, Channels)`. Our model expects a "batch" folder of images. We add an extra dimension to turn our single face image into a "folder containing 1 face image".
*   **Code Scaffold**:
    ```python
    face_batch = np.expand_dims(face_normalized, axis=___)
    ```
*   **Cheat Sheet**:
    *   Set `axis=0` to add the dimension at the start of the shape tuple.

---

#### **Method: `get_embedding`**
We run the preprocessed face through the network to generate its 128-D passport.

##### **TODO 8a: Predict Embedding**
*   **The Logic**: Pass the preprocessed image batch to the model. We explicitly disable logging printouts because showing a progress bar inside a real-time video stream loop will freeze the frame rate.
*   **Code Scaffold**:
    ```python
    embedding = self.model.predict(face_input, verbose=___)
    ```
*   **Cheat Sheet**:
    *   Set `verbose=0`.

##### **TODO 8b: Flatten Array**
*   **The Logic**: The model's prediction returns a 2D batch tensor of shape `(1, 128)`. We need a 1D vector of shape `(128,)` to perform our database math.
*   **Code Scaffold**:
    ```python
    embedding = embedding.___()
    ```
*   **Cheat Sheet**:
    *   Call the `.flatten()` method on the numpy array.

##### **TODO 8c: L2 Normalisation**
*   **The Logic**: This is like shrinking or stretching a line so its length is exactly $1.0$. When all face passports are scaled to a length of 1.0, comparing them is as simple as multiplying their corresponding numbers and adding them up, regardless of how bright or large the original image was.
*   **Code Scaffold**:
    ```python
    embedding = embedding / np.linalg.norm(___)
    ```
*   **Cheat Sheet**:
    *   Pass `embedding` to `np.linalg.norm()` to calculate its length.

---

#### **Method: `cosine_similarity`**
We measure how similar two facial passports are.

##### **TODO 9: Compute Dot Product**
*   **The Logic**: Since the passports are scaled to a length of 1.0, finding similarity is just multiplying the corresponding numbers in the two lists and adding them up (the dot product). The higher the sum (closer to 1.0), the more similar the faces.
*   **Code Scaffold**:
    ```python
    return np.dot(___, ___)
    ```
*   **Cheat Sheet**:
    *   Pass `embedding1` and `embedding2` to `np.dot()`.

---

#### **Method: `recognize`**
We search our database of known teachers/students to find a match.

##### **TODO 10a: Compare to Stored Database**
*   **The Logic**: Compare our unknown face embedding against all embeddings stored in the database.
*   **Code Scaffold**:
    ```python
    similarity = self.cosine_similarity(embedding, ___)
    ```
*   **Cheat Sheet**:
    *   Compare `embedding` (input face) with the current database item `known_embedding` inside the loop.

##### **TODO 10b: Update Best Match**
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

##### **TODO 10c: Apply Threshold**
*   **The Logic**: The threshold is a cutoff score. If the best similarity is below this cutoff (e.g., 0.60), we declare the face "Unknown". This is like a security guard refusing entry if someone's photo ID doesn't look at least 60% similar to them. This is critical to prevent false recognition of visitors or students not in our database.
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
