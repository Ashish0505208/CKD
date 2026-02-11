import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

dataset_dir = "/content/drive/MyDrive/Datasetcapstone/kidney-dataset"
train_dir = dataset_dir + "/train"
valid_dir = dataset_dir + "/valid"

print("Train:", train_dir)
print("Valid:", valid_dir)

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
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    paths, labels = [], []
    for f in files:
        label = get_label_from_name(f)
        if label is None:
            continue
        paths.append(os.path.join(folder, f))
        labels.append(label)
    return paths, labels

train_paths, train_labels = load_paths_labels(train_dir)
valid_paths, valid_labels = load_paths_labels(valid_dir)

print("Train images:", len(train_paths))
print("Valid images:", len(valid_paths))

IMG_SIZE = 224

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = mobilenet_v2.preprocess_input(img)
    return img, tf.one_hot(label, num_classes)

batch = 32

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(5000)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch)
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch)
    .prefetch(tf.data.AUTOTUNE)
)

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)

base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))

for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
out = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=Adam(0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20,
    class_weight=class_weights
)

model.save("/content/drive/MyDrive/mobilenet_kidney_final.h5")
print("Saved model to Drive!")
plt.figure(figsize=(12,5))

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
