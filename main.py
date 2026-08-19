import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import random
from sklearn.model_selection import train_test_split
from collections import Counter


# ==================================#
# 1. EDA
# ==================================#
def CalcutePixels(PATH):
    train_path = PATH
    widths = []
    heights = []

    for class_name in ["cat", "dog"]:
        folder = os.path.join(train_path, class_name)
        if not os.path.exists(folder):
            continue

        files = os.listdir(folder)[:200]

        for file_name in files:
            file_path = os.path.join(folder, file_name)
            try:
                with Image.open(file_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except:
                pass

    if widths:
        print(f"Total images checked: {len(widths)}")
        print(f"Average width: {np.mean(widths):.0f} pixels")
        print(f"Average height: {np.mean(heights):.0f} pixels")
        print(f"Smallest width: {min(widths)} and Largest width: {max(widths)}")
        print(f"Smallest height: {min(heights)} and Largest height: {max(heights)}")

        # Intelligent suggestion for target_size
        avg_w = int(np.mean(widths))
        avg_h = int(np.mean(heights))
        suggested_size = int((avg_w + avg_h) / 4)  # A reasonable number
        print(f"\nMy suggestion for target_size: around ({suggested_size}, {suggested_size}) ")
        print(f"   (or if you prefer a square size, choose {suggested_size})")
    else:
        print("No images found! Please check the path.")


def showSomeExmples(firstClassPath, secClassPath):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    first_folder = firstClassPath
    first_samples = random.sample(os.listdir(first_folder), 4)
    for i, img_name in enumerate(first_samples):
        img = mpimg.imread(os.path.join(first_folder, img_name))
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Cat {i+1}")
        axes[0, i].axis("off")

    sec_folder = secClassPath
    sec_samples = random.sample(os.listdir(sec_folder), 4)
    for i, img_name in enumerate(sec_samples):
        img = mpimg.imread(os.path.join(sec_folder, img_name))
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Dog {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def count_and_split_report(dataset_path, train_ratio=0.75, test_ratio=0.25):
    classes = ["cat", "dog"]

    print("=" * 50)
    print("DATASET SPLIT REPORT")
    print("=" * 50)

    total_train = 0
    total_test = 0
    total_all = 0

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.exists(class_path):
            print(f"⚠️ Class folder not found: {class_path}")
            continue

        # Get all files (assume all are images)
        all_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        total_count = len(all_files)

        # Calculate split counts (round to nearest integer)
        train_count = int(round(total_count * train_ratio))
        test_count = total_count - train_count  # ensures sum equals total

        # If test_count is less than 1 (very small dataset), adjust
        if test_count < 1:
            test_count = 1
            train_count = total_count - 1

        print(f"\nClass: {class_name.upper()}")
        print(f"   Total images: {total_count}")
        print(f"   ➜ Train: {train_count} ({train_count/total_count*100:.1f}%)")
        print(f"   ➜ Test:  {test_count} ({test_count/total_count*100:.1f}%)")

        total_train += train_count
        total_test += test_count
        total_all += total_count

    print("\n" + "=" * 50)
    print("OVERALL SUMMARY")
    print("=" * 50)
    print(f"Total images across all classes: {total_all}")
    print(f"Total training images: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"Total testing images:  {total_test} ({total_test/total_all*100:.1f}%)")
    print("=" * 50)


def analyze_dataset(dataset_path):

    from collections import Counter

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
    )

    class_counts = Counter()
    extension_counts = Counter()
    total_images = 0

    for class_name in os.listdir(dataset_path):

        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for root, _, files in os.walk(class_path):

            for file in files:

                if not file.lower().endswith(image_extensions):
                    continue

                total_images += 1

                extension = os.path.splitext(file)[1].lower()

                extension_counts[extension] += 1
                class_counts[class_name] += 1

    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    print(f"Total images: {total_images}")

    print("\nImages per class:")

    for class_name, count in class_counts.items():

        percentage = (count / total_images) * 100

        print(f"  {class_name}: " f"{count} " f"({percentage:.2f}%)")

    print("\nFile extensions:")

    for extension, count in extension_counts.items():

        print(f"  {extension}: {count}")

    print("=" * 60)


def tensorflow_image_check(dataset_path):

    bad_images = []

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
    )

    for root, _, files in os.walk(dataset_path):

        for file in files:

            if not file.lower().endswith(valid_extensions):
                continue

            image_path = os.path.join(root, file)

            try:

                image_bytes = tf.io.read_file(image_path)

                image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)

                # Force TensorFlow evaluation
                _ = image.numpy()

            except Exception as e:

                bad_images.append({"path": image_path, "error": str(e)})

    print("\n" + "=" * 60)
    print("TENSORFLOW IMAGE CHECK")
    print("=" * 60)

    print(f"Bad images: {len(bad_images)}")

    for item in bad_images:

        print("\nFILE:")
        print(item["path"])

        print("\nERROR:")
        print(item["error"])

    print("=" * 60)

    return bad_images


def analyze_image_dimensions(dataset_path):

    dimensions = Counter()

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
    )

    for root, _, files in os.walk(dataset_path):

        for file in files:

            if not file.lower().endswith(image_extensions):
                continue

            image_path = os.path.join(root, file)

            try:

                with Image.open(image_path) as img:

                    dimensions[img.size] += 1

            except Exception:
                pass

    print("\n" + "=" * 60)
    print("IMAGE DIMENSIONS")
    print("=" * 60)

    print(f"Unique dimensions: " f"{len(dimensions)}")

    print("\nMost common dimensions:")

    for size, count in dimensions.most_common(20):

        print(f"  {size}: {count}")

    print("=" * 60)

    return dimensions


# ==================================#
# 2. PreProcess
# ==================================#
def create_data_generators():

    DATASET_PATH = "dataset"

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32

    VALIDATION_SPLIT = 0.25
    SEED = 42

    # 1-Get image paths
    class_names = sorted([name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name))])

    print("Classes:", class_names)

    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    )

    image_paths = []
    labels = []

    # 2-Read images from each class
    for class_name in class_names:

        class_path = os.path.join(DATASET_PATH, class_name)

        class_index = class_to_index[class_name]

        for root, _, files in os.walk(class_path):

            for file in files:

                if file.lower().endswith(image_extensions):

                    image_path = os.path.join(root, file)

                    image_paths.append(image_path)

                    labels.append(class_index)

    image_paths = np.array(image_paths)

    labels = np.array(labels, dtype=np.int32)


    # 3-Stratified train/validation split
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        image_paths, labels, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=labels, shuffle=True
    )

    print("\nSplit results:")
    print(f"Training samples: " f"{len(train_paths)}")

    print(f"Validation samples: " f"{len(validation_paths)}")

    # 4-Create tf.data datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))

    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_paths, validation_labels))

    # 5-Image loading function
    def load_image(image_path, label):

        image = tf.io.read_file(image_path)

        image = tf.io.decode_image(image, channels=3, expand_animations=False)

        image.set_shape([None, None, 3])

        image = tf.image.resize(image, IMAGE_SIZE)

        image = tf.cast(image, tf.float32)

        image = image / 255.0

        return image, label

    # 6-Map image loading
    train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    validation_dataset = validation_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)


    # 7-Shuffle training data
    train_dataset = train_dataset.shuffle(buffer_size=len(train_paths), seed=SEED, reshuffle_each_iteration=True)


    # 8-Batch
    train_dataset = train_dataset.batch(BATCH_SIZE)

    validation_dataset = validation_dataset.batch(BATCH_SIZE)


    # 9- Prefetch
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    dataset_info = {
        "class_names": class_names,
        "train_samples": len(train_paths),
        "validation_samples": len(validation_paths),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
    }
    return train_dataset, validation_dataset, dataset_info


def print_generator_summary(dataset_info):

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)

    print(f"Classes: {dataset_info['class_names']}")

    print(f"Total classes: " f"{len(dataset_info['class_names'])}")

    print(f"Training samples: " f"{dataset_info['train_samples']}")

    print(f"Validation samples: " f"{dataset_info['validation_samples']}")

    print(f"Image size: " f"{dataset_info['image_size']}")

    print(f"Batch size: " f"{dataset_info['batch_size']}")

    print("=" * 50)


def analyze_dataset_labels(train_dataset, validation_dataset, dataset_info):
    class_names = dataset_info["class_names"]

    print("\n" + "=" * 60)
    print("LABEL & DATASET ANALYSIS")
    print("=" * 60)

    # ----------------------------------
    # Class mapping
    # ----------------------------------

    print("\nClass mapping:")

    for index, class_name in enumerate(class_names):
        print(f"  {index} -> {class_name}")

    print("\nNumber of classes:")
    print(f"  {len(class_names)}")

    # ----------------------------------
    # Count labels in training dataset
    # ----------------------------------

    train_counts = np.zeros(len(class_names), dtype=np.int64)

    for _, labels in train_dataset:

        labels = labels.numpy().astype(int)

        for label in labels:
            train_counts[label] += 1

    # ----------------------------------
    # Count labels in validation dataset
    # ----------------------------------

    validation_counts = np.zeros(len(class_names), dtype=np.int64)

    for _, labels in validation_dataset:

        labels = labels.numpy().astype(int)

        for label in labels:
            validation_counts[label] += 1

    # ----------------------------------
    # Print distribution
    # ----------------------------------

    print("\nTraining label distribution:")

    for index, class_name in enumerate(class_names):

        percentage = (train_counts[index] / train_counts.sum()) * 100

        print(f"  {index} ({class_name}): " f"{train_counts[index]} " f"({percentage:.2f}%)")

    print("\nValidation label distribution:")

    for index, class_name in enumerate(class_names):

        percentage = (validation_counts[index] / validation_counts.sum()) * 100

        print(f"  {index} ({class_name}): " f"{validation_counts[index]} " f"({percentage:.2f}%)")

    # ----------------------------------
    # Dataset sizes
    # ----------------------------------

    print("\nDataset sizes:")

    print(f"  Training: " f"{train_counts.sum()}")

    print(f"  Validation: " f"{validation_counts.sum()}")

    # ----------------------------------
    # First few batches
    # ----------------------------------

    print("\nFirst 3 training batches:")

    for batch_index, (_, labels) in enumerate(train_dataset.take(3)):

        labels = labels.numpy().astype(int)

        print(f"  Batch {batch_index + 1}: " f"{labels.tolist()}")

    print("\nFirst 3 validation batches:")

    for batch_index, (_, labels) in enumerate(validation_dataset.take(3)):

        labels = labels.numpy().astype(int)

        print(f"  Batch {batch_index + 1}: " f"{labels.tolist()}")

    print("=" * 60)

# ==================================#
# 3. modeling
# ==================================#
def modeling():

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(224, 224, 3)),
            MaxPool2D(pool_size=2, strides=2),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPool2D(pool_size=2, strides=2),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPool2D(pool_size=2, strides=2),
            GlobalAveragePooling2D(),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.summary()

    return model

# ==================================#
# 4. compile and fit
# ==================================#
def compile_and_fit(model, train_dataset, validation_dataset):

    MODEL_PATH = "./saved_models/animal_cnn.keras"

    if os.path.exists(MODEL_PATH):
        print("Model found. Loading the trained model...")
        model = load_model(MODEL_PATH)
        history = None

    else:
        print("No saved model found. Training a new model...")

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=30,
            callbacks=[early_stopping],
            verbose=1,
        )

        model.save(MODEL_PATH)
        print("Model saved successfully.")

    return model, history

# ==================================#
# main
# ==================================#

if __name__ == "__main__":

    dataset_path = "dataset"
    cat_path = "dataset/cat"
    dog_path = "dataset/dog"

    # 1.EDA
    # CalcutePixels(dataset_path)
    # showSomeExmples(cat_path, dog_path)
    # count_and_split_report(dataset_path)
    # analyze_dataset("dataset")
    # bad_images = tensorflow_image_check("dataset")
    # analyze_image_dimensions("dataset")

    # 2.PreProcess  
    # train_dataset, validation_dataset, dataset_info = create_data_generators()
    # print_generator_summary(dataset_info)
    # analyze_dataset_labels(train_dataset, validation_dataset, dataset_info)

    # 3.Modeling
    # model = modeling()
    
    # 4.compile and fit
    # model, history = compile_and_fit(model, train_dataset, validation_dataset)
