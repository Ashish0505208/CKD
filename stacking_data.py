import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ======================================================
# PATHS
# ======================================================
BASE = "/content/drive/MyDrive/Datasetcapstone"
TRAIN_DIR = BASE + "/kidney-dataset/train"

CNN_PATH = BASE + "/ckd_cnn_224.h5"
MOB_PATH = BASE + "/mobilenet_kidney_final.h5"
EFF_PATH = BASE + "/efficientnetb0_kidney_fast.h5"

SAVE_DIR = BASE + "/stacking_data"
os.makedirs(SAVE_DIR, exist_ok=True)

classes = ["cyst", "stone", "tumor", "normal"]

# ======================================================
# LOAD MODELS
# ======================================================
cnn = tf.keras.models.load_model(CNN_PATH)
mob = tf.keras.models.load_model(MOB_PATH)
eff = tf.keras.models.load_model(EFF_PATH)

print("✅ Models loaded")

# ======================================================
# LABEL FUNCTION
# ======================================================
def get_label(fname):
    f = fname.lower()
    if "tumor" in f:
        return 2
    for i, c in enumerate(classes):
        if c in f:
            return i
    return None

# ======================================================
# BATCHED STREAMING (FAST)
# ======================================================
BATCH = 32

p1, p2, p3, y = [], [], [], []

files = [f for f in os.listdir(TRAIN_DIR)
         if f.lower().endswith(('.jpg','.jpeg','.png'))]

print("Train images:", len(files))

for i in range(0, len(files), BATCH):
    batch_files = files[i:i+BATCH]

    X224, X160, labels = [], [], []

    for f in batch_files:
        label = get_label(f)
        if label is None:
            continue

        path = os.path.join(TRAIN_DIR, f)

        img224 = load_img(path, target_size=(224,224))
        arr224 = img_to_array(img224) / 255.0

        img160 = load_img(path, target_size=(160,160))
        arr160 = img_to_array(img160) / 255.0

        X224.append(arr224)
        X160.append(arr160)
        labels.append(label)

    if not X224:
        continue

    X224 = np.array(X224)
    X160 = np.array(X160)

    # 🔥 FAST GPU BATCH PREDICTION
    p1.extend(cnn.predict(X224, verbose=0))
    p2.extend(mob.predict(X224, verbose=0))
    p3.extend(eff.predict(X160, verbose=0))
    y.extend(labels)

    if i % (BATCH * 20) == 0:
        print(f"Processed {i}/{len(files)}")

# ======================================================
# SAVE STACKING DATA
# ======================================================
p1 = np.array(p1)
p2 = np.array(p2)
p3 = np.array(p3)
y  = np.array(y)

np.save(SAVE_DIR + "/p1_cnn_train.npy", p1)
np.save(SAVE_DIR + "/p2_mob_train.npy", p2)
np.save(SAVE_DIR + "/p3_eff_train.npy", p3)
np.save(SAVE_DIR + "/y_train.npy", y)

print("💾 Base predictions saved")
print("Shapes:", p1.shape, p2.shape, p3.shape)
