import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import accuracy_score, classification_report

# --------------------------
# PATHS
# --------------------------
model_path = "/content/drive/MyDrive/Datasetcapstone/efficientnetb0_kidney_fast.h5"
test_dir   = "/content/drive/MyDrive/Datasetcapstone/kidney-dataset/test"

classes = ["cyst", "stone", "tumor", "normal"]

# --------------------------
# LOAD MODEL
# --------------------------
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded!\n")

# --------------------------
# OPTIONAL label extractor (ONLY for metrics)
# --------------------------
def get_label_from_name(filename):
    f = filename.lower()
    if "tumor" in f:
        return 2
    for i, c in enumerate(classes):
        if c in f:
            return i
    return None  # unlabeled

# --------------------------
# PREDICT ALL IMAGES
# --------------------------
y_true = []
y_pred = []

files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
print(f"📂 Found {len(files)} images\n")

for file in files:
    img_path = os.path.join(test_dir, file)

    img = load_img(img_path, target_size=(160, 160))
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)
    pred_idx = np.argmax(pred)
    confidence = np.max(pred) * 100

    print(f"{file:45s} → {classes[pred_idx]:6s} ({confidence:.2f}%)")

    # Only add to metrics IF label exists
    true_label = get_label_from_name(file)
    if true_label is not None:
        y_true.append(true_label)
        y_pred.append(pred_idx)

# --------------------------
# METRICS (only for labeled images)
# --------------------------
if len(y_true) > 0:
    print("\n🎯 Accuracy (labeled images only):", accuracy_score(y_true, y_pred) * 100)
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
else:
    print("\nℹ️ No labeled images found → metrics skipped")
