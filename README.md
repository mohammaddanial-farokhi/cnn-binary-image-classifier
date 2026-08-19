# cnn-binary-image-classifier

**Binary Image Classifier using CNN**

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to perform binary classification on **two distinct image classes**.  
It is designed to be completely generic – you can use it for any two categories, such as:
- Cats vs Dogs
- Apples vs Oranges
- Healthy vs Diseased leaves
- Happy vs Sad faces
- Or any other two custom classes you have.

The code includes full **Exploratory Data Analysis (EDA)**, **data preprocessing**, **model training**, and **automatic model saving/loading**.

---
## 📁 Dataset Structure

Place your dataset in the `dataset/` folder with two subfolders – one for each class.  
**You can name them anything**, but for this example we use `class_1` and `class_2`:

```
project-root/
│
├── dataset/
│   ├── class_1/               # All images for the first class
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── class_2/               # All images for the second class
│       ├── img_001.jpg
│       ├── img_002.jpg
│       └── ...
│
├── saved_models/
│   └── animal_cnn.keras       # Saved model (created after training)
│
├── main.py                    # Main script (your complete code)
└── README.md
```

> **Important**: The core training pipeline (`create_data_generators()`) automatically detects **any** subfolder names inside `dataset/`.  
> However, the helper EDA functions in the `__main__` block currently reference `cat` and `dog` as examples. If you use different folder names:
> - **Option 1 (simplest)**: Rename your folders to `cat` and `dog` to run everything without changes.
> - **Option 2**: Update the paths inside `main.py`:
>   ```python
>   class1_path = "dataset/class_1"
>   class2_path = "dataset/class_2"
>   ```
>   and adjust the function calls accordingly.

---

## ⚙️ Requirements

Install the required libraries with:

```bash
pip install tensorflow numpy matplotlib Pillow scikit-learn
```

Or use a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Dependencies**:
- `tensorflow >= 2.8.0`
- `numpy`
- `matplotlib`
- `Pillow`
- `scikit-learn`

---

## 🚀 Usage

### 1. Activating Different Parts of the Code

At the bottom of `main.py`, inside `if __name__ == "__main__"`, all function calls are **commented out**.  
Uncomment the sections you wish to execute:

```python
if __name__ == "__main__":

    dataset_path = "dataset"
    # Update these paths to match your actual class folders
    class1_path = "dataset/class_1"
    class2_path = "dataset/class_2"

    # 1. EDA (Exploratory Data Analysis)
    # CalcutePixels(dataset_path)          # Calculate average image dimensions
    # showSomeExmples(class1_path, class2_path)  # Show sample images
    # count_and_split_report(dataset_path) # Train/Test split report
    # analyze_dataset(dataset_path)        # Full dataset analysis
    # bad_images = tensorflow_image_check(dataset_path) # Find corrupted images
    # analyze_image_dimensions(dataset_path) # Analyse image dimensions

    # 2. PreProcess (Create Data Generators)
    # train_dataset, validation_dataset, dataset_info = create_data_generators()
    # print_generator_summary(dataset_info)
    # analyze_dataset_labels(train_dataset, validation_dataset, dataset_info)

    # 3. Modeling (Build the CNN)
    # model = modeling()

    # 4. Compile & Fit (Train or load the model)
    # model, history = compile_and_fit(model, train_dataset, validation_dataset)
```

### 2. Training the Model

To train the model from scratch, uncomment these lines:

```python
train_dataset, validation_dataset, dataset_info = create_data_generators()
model = modeling()
model, history = compile_and_fit(model, train_dataset, validation_dataset)
```

Then execute:

```bash
python main.py
```

**Training Configuration**:
- **Image size**: `224x224` pixels
- **Max epochs**: `30` with **Early Stopping** (stops if validation loss does not improve for 5 consecutive epochs)
- **Batch size**: `32`
- **Validation split**: `25%` (stratified to preserve class balance)

Once training finishes, the model is saved to `./saved_models/animal_cnn.keras`.  
If this file already exists, the model is **loaded** directly and training is skipped – allowing you to reuse a previously trained model without retraining.

---
## 🧠 Model Architecture

The CNN architecture is defined as follows:

| Layer                 | Description                              |
|-----------------------|------------------------------------------|
| `Conv2D(32, 3x3)`     | Low‑level feature extraction             |
| `MaxPool2D(2,2)`      | Dimensionality reduction                 |
| `Conv2D(64, 3x3)`     | Mid‑level feature extraction             |
| `MaxPool2D(2,2)`      | Dimensionality reduction                 |
| `Conv2D(128, 3x3)`    | High‑level feature extraction            |
| `MaxPool2D(2,2)`      | Dimensionality reduction                 |
| `GlobalAveragePooling2D` | Convert feature maps to a vector      |
| `Dense(32, ReLU)`     | Fully connected layer                    |
| `Dropout(0.5)`        | Regularization (prevents overfitting)    |
| `Dense(1, Sigmoid)`   | Output layer (probability for class 2)   |

> **Loss function**: `binary_crossentropy`  
> **Optimizer**: `Adam`

---

## 📊 Results 

After training, typical performance metrics are:

- **Training accuracy**: `~83%` (varies with dataset size and complexity)
- **Validation accuracy**: `~85%`

You can plot accuracy and loss curves using the `history` object returned by `compile_and_fit()` for deeper insight.
