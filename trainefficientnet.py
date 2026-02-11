#old
#old
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetB0, efficientnet
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ===============================
# SPEED BOOST (T4 GPU)
# ===============================
mixed_precision.set_global_policy("mixed_float16")

# ===============================
# PATHS
# ===============================
dataset_dir = "/content/drive/MyDrive/Datasetcapstone/kidney-dataset"
train_dir = dataset_dir + "/train"
valid_dir = dataset_dir + "/valid"

# ===============================
# CLASSES (FIXED ORDER)
# ===============================
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

def load_paths_labels(folder):
    paths, labels = [], []
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            label = get_label_from_name(f)
            if label is not None:
                paths.append(os.path.join(folder, f))
                labels.append(label)
    return paths, labels

train_paths, train_labels = load_paths_labels(train_dir)
valid_paths, valid_labels = load_paths_labels(valid_dir)

print("Train images:", len(train_paths))
print("Valid images:", len(valid_paths))

# ===============================
# TF.DATA PIPELINE (FAST + RAM SAFE)
# ===============================
IMG_SIZE = 160
BATCH = 64

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = efficientnet.preprocess_input(img)
    return img, tf.one_hot(label, num_classes)

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(4000)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

# ===============================
# CLASS WEIGHTS
# ===============================
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)

# ===============================
# MODEL (FAST EfficientNetB0)
# ===============================
base = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation="softmax", dtype="float32")(x)

model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=Adam(0.0007),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# TRAIN
# ===============================
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=15,
    class_weight=class_weights
)

# ===============================
# SAVE MODEL
# ===============================
model.save("/content/drive/MyDrive/efficientnetb0_kidney_fast.h5")
print("✅ Model saved to Drive")

# ===============================
# PLOTS
# ===============================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Valid")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Valid")
plt.title("Loss")
plt.legend()

plt.show()
