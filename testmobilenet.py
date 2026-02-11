import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model_path = "/content/drive/MyDrive/Datasetcapstone/mobilenet_kidney_final.h5"
test_dir   = "/content/drive/MyDrive/Datasetcapstone/kidney-dataset/test"

classes = ["cyst", "stone", "tumor", "normal"]

print("📥 Loading model...")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded!\n")

files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]

print(f"📂 Found {len(files)} images to predict.\n")

for file in files:
    img_path = os.path.join(test_dir, file)

    try:

        img = load_img(img_path, target_size=(224, 224))
        arr = img_to_array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
    except Exception as e:
        print(f"⚠️ Could not read {file}: {e}")
        continue

    pred = model.predict(arr, verbose=0)
    pred_idx = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    predicted_class = classes[pred_idx]

    print(f"🖼️ {file} → {predicted_class} ({confidence:.2f}%)")

print("\n✅ Prediction complete!")
