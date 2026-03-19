import tensorflow as tf
import numpy as np
import zipfile
from pathlib import Path
import os
from sklearn.metrics import f1_score

# Optional (for F1 + confusion matrix)
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. SETUP PATHS ---
ZIP_NAME = 'testset.zip'
BASE_EXTRACT_PATH = Path('/content')
DATA_DIR = BASE_EXTRACT_PATH / 'testset'

IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# --- 2. UNZIP TEST DATA ---
if Path(ZIP_NAME).exists():
    with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
        zip_ref.extractall(BASE_EXTRACT_PATH)
        print("Successfully unzipped test data")
else:
    print("Error: testset.zip not found")

# --- 3. LOAD TEST DATASET ---
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    shuffle=False,  # IMPORTANT for correct evaluation
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = test_ds.class_names
print("Classes:", class_names)

# --- 4. LOAD TRAINED MODEL ---
model = tf.keras.models.load_model('asl_model.h5')

# --- 5. TEST A SINGLE IMAGE ---
print("\n--- Single Image Test ---")

# Pick first image from dataset
for images, labels in test_ds.take(1):
    single_image = images[0]
    true_label = labels[0].numpy()

# Add batch dimension
img_array = tf.expand_dims(single_image, 0)

# Predict
pred = model.predict(img_array)
pred_label = np.argmax(pred)

print(f"Actual: {class_names[true_label]}")
print(f"Predicted: {class_names[pred_label]}")
print(f"Confidence: {100*np.max(pred):.2f}%")

# --- 6. FULL DATASET EVALUATION ---
print("\n--- Full Dataset Evaluation ---")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# --- Accuracy ---
accuracy = np.mean(y_true == y_pred)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# --- F1 Scores ---
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"\nF1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")

# --- Confusion Matrix ---
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# --- Classification Report (Precision, Recall, F1) ---
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
