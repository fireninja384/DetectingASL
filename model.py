import tensorflow as tf
from tensorflow.keras import layers, models
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt


# --- 1. SETUP PATHS ---
ZIP_NAME = 'trainingset.zip'
BASE_EXTRACT_PATH = Path('/content')
DATA_DIR = BASE_EXTRACT_PATH / 'trainingset'

IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# --- 2. UNZIP THE DATA ---
if Path(ZIP_NAME).exists():
    with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
        zip_ref.extractall(BASE_EXTRACT_PATH)
        print("Successfully unzipped to /content/trainingset")
else:
    print(f"Error: Could not find {ZIP_NAME} in the sidebar. Please upload it!")

# --- 3. LOAD DATA ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Check the class names found in the directory
class_names = train_ds.class_names
print("TensorFlow found these classes:", class_names)

# Check the mapping (e.g., is 'A' actually 0?)
for i, name in enumerate(class_names):
    print(f"Index {i} corresponds to Label: {name}")

# Display sample images from the training dataset
print("\nDisplaying sample images:")
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# --- DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# --- 4. THE CNN MODEL ---
model = models.Sequential([
    data_augmentation,  # 👈 NEW (applied only during training)
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

    #layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    #layers.MaxPooling2D((2, 2)),

    #layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    #layers.MaxPooling2D((2, 2)),

    #layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    #layers.MaxPooling2D((2, 2)),

    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    # Block 3
    layers.Conv2D(96, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'), #shivam: reduce neurons to potentially reduce parameter count.
    #layers.GlobalAveragePooling2D(),
    #layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax') # Use len(class_names) for the output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- 5. TRAIN ---
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # higher max
    callbacks=[early_stop]
)


# Save the final result
model.save('asl_model.h5')


