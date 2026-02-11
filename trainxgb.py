import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ======================================================
# PATHS
# ======================================================
BASE = "/content/drive/MyDrive/Datasetcapstone"
STACK_DIR = BASE + "/stacking_data"
SAVE_MODEL = BASE + "/xgboost_meta_model.pkl"

classes = ["cyst", "stone", "tumor", "normal"]

# ======================================================
# LOAD STACKING FEATURES
# ======================================================
print("📥 Loading stacking data...")

p1 = np.load(STACK_DIR + "/p1_cnn_train.npy")
p2 = np.load(STACK_DIR + "/p2_mob_train.npy")
p3 = np.load(STACK_DIR + "/p3_eff_train.npy")
y  = np.load(STACK_DIR + "/y_train.npy")

print("Shapes:")
print("CNN:", p1.shape)
print("MobileNet:", p2.shape)
print("EfficientNet:", p3.shape)
print("Labels:", y.shape)

# ======================================================
# BUILD META FEATURES
# ======================================================
X_meta = np.hstack([p1, p2, p3])
print("Meta feature shape:", X_meta.shape)

# ======================================================
# TRAIN XGBOOST META-MODEL
# ======================================================
print("🚀 Training XGBoost meta-model...")

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist"   # FAST + LOW RAM
)

xgb.fit(X_meta, y)

# ======================================================
# SAVE MODEL
# ======================================================
joblib.dump(xgb, SAVE_MODEL)
print(f"💾 XGBoost model saved at {SAVE_MODEL}")

# ======================================================
# EVALUATION (TRAIN STACK)
# ======================================================
y_pred = np.argmax(xgb.predict_proba(X_meta), axis=1)

acc = accuracy_score(y, y_pred)
print(f"\n🎯 Train Stacking Accuracy: {acc*100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y, y_pred, target_names=classes))

cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)
plt.title("XGBoost Stacking — Train Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
