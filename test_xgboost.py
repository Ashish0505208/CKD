import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

# ==============================
# PATHS (UPDATE IF NEEDED)
# ==============================
IMAGE_FOLDER = "/content/drive/MyDrive/Datasetcapstone/kidney-dataset/test"

CNN_PATH  = "/content/drive/MyDrive/Datasetcapstone/ckd_cnn_224.h5"
MOB_PATH  = "/content/drive/MyDrive/Datasetcapstone/mobilenet_kidney_final.h5"
EFF_PATH  = "/content/drive/MyDrive/Datasetcapstone/efficientnetb0_kidney_fast.h5"
XGB_PATH  = "/content/drive/MyDrive/Datasetcapstone/xgb_stacking_model.pkl"

# ==============================
# LOAD MODELS
# ==============================
cnn = tf.keras.models.load_model(CNN_PATH)
mob = tf.keras.models.load_model(MOB_PATH)
eff = tf.keras.models.load_model(EFF_PATH)
xgb = joblib.load(XGB_PATH)

print("✅ All models loaded")

# ==============================
# CLASS NAMES
# ==============================
classes = ["cyst", "stone", "tumor", "normal"]

# ==============================
# PREDICT FOLDER
# ==============================
def predict_folder(folder):
    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"\n📂 Found {len(files)} images\n")

    for file in files:
        path = os.path.join(folder, file)

        # ---- CNN + MobileNet (224x224)
        img224 = load_img(path, target_size=(224,224))
        arr224 = img_to_array(img224) / 255.0
        arr224 = np.expand_dims(arr224, 0)

        p1 = cnn.predict(arr224, verbose=0)[0]
        p2 = mob.predict(arr224, verbose=0)[0]

        # ---- EfficientNet (160x160)
        img160 = load_img(path, target_size=(160,160))
        arr160 = img_to_array(img160) / 255.0
        arr160 = np.expand_dims(arr160, 0)

        p3 = eff.predict(arr160, verbose=0)[0]

        # ---- Stack predictions
        meta_features = np.hstack([p1, p2, p3]).reshape(1, -1)

        probs = xgb.predict_proba(meta_features)[0]
        pred_idx = np.argmax(probs)

        print(
            f"{file:30s} → "
            f"{classes[pred_idx].upper():7s} "
            f"(confidence: {probs[pred_idx]*100:.2f}%)"
        )

# ==============================
# RUN
# ==============================
predict_folder(IMAGE_FOLDER)
