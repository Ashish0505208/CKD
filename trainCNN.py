import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ======================================================
# 1️⃣ Mount Drive
# ======================================================
from google.colab import drive
drive.mount('/content/drive')

# ======================================================
# 2️⃣ Paths
# ======================================================
BASE_DIR = "/content/drive/MyDrive/Datasetcapstone/kidney-dataset"
TRAIN_DIR = BASE_DIR + "/train"
VALID_DIR = BASE_DIR + "/valid"
SAVE_PATH = "/content/drive/MyDrive/Datasetcapstone/ckd_cnn_224.h5"

print("Train:", TRAIN_DIR)
print("Valid:", VALID_DIR)

# ======================================================
# 3️⃣ Classes (ORDER LOCKED)
# ======================================================
classes = ["cyst", "stone", "tumor", "normal"]
num_classes = len(classes)

def get_label_from_name(filename):
    f = filename.lower()
    if "tumor" in f:
        return 2
    for i, c in enumerate(classes):
        if c in f:
            return i
    return None

# ======================================================
# 4️⃣ Collect paths + labels (RAM SAFE)
# ======================================================
def collect_paths(folder):
    paths, labels = [], []
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            label = get_label_from_name(f)
            if label is not None:
                paths.append(os.path.join(folder, f))
                labels.append(label)
    return paths, labels

train_paths, train_labels = collect_paths(TRAIN_DIR)
valid_paths, valid_labels = collect_paths(VALID_DIR)

print("Train images:", len(train_paths))
print("Valid images:", len(valid_paths))

# ======================================================
# 5️⃣ tf.data PIPELINE (NO RAM ISSUE)
# ======================================================
IMG_SIZE = 224
BATCH = 32

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, tf.one_hot(label, num_classes)

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(4000)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

# ======================================================
# 6️⃣ Class Weights
# ======================================================
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)

# ======================================================
# 7️⃣ CNN ARCHITECTURE (UNCHANGED)
# ======================================================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================================================
# 8️⃣ Train
# ======================================================
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20,
    class_weight=class_weights
)

# ======================================================
# 9️⃣ Save Model to Drive
# ======================================================
model.save(SAVE_PATH)
print(f"💾 Model saved at {SAVE_PATH}")

# ======================================================
# 🔟 Plot
# ======================================================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="valid")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="valid")
plt.title("Loss")
plt.legend()

plt.show()
