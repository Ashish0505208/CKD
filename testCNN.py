import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = tf.keras.models.load_model("/content/drive/MyDrive/Datasetcapstone/ckd_cnn_model_v4.h5")
classes = ["cyst", "stone", "tumor", "normal"]

def get_label(name):
    name = name.lower()
    for i,c in enumerate(classes):
        if c in name:
            return i
    return None

test_dir = "/content/drive/MyDrive/Datasetcapstone/kidney-dataset/test"

correct = 0
total = 0

for f in os.listdir(test_dir):
    label = get_label(f)
    if label is None:
        continue

    img = load_img(os.path.join(test_dir,f), target_size=(224,224))
    arr = img_to_array(img)/255.0
    arr = np.expand_dims(arr,0)

    pred = np.argmax(model.predict(arr, verbose=0))
    total += 1
    correct += (pred == label)

    print(f"{f} → predicted:{classes[pred]} | actual:{classes[label]}")

print(f"\n🎯 Accuracy: {100*correct/total:.2f}%")

