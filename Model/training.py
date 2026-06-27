import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# Dataset Path
# ==========================================
BASE_DIR = r"C:\R\Sistem Deteksi Ekspresi Wajah"

train_dir = os.path.join(BASE_DIR, "Dataset", "train")
test_dir = os.path.join(BASE_DIR, "Dataset", "test")

print(train_dir)
print(test_dir)

# ==========================================
# Data Augmentation
# ==========================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# ==========================================
# Data Generator
# ==========================================
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print("Class Indices:")
print(train_generator.class_indices)

# ==========================================
# CNN Model
# ==========================================
model = Sequential([
    # Block 1
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(48, 48, 1)
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Block 2
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Block 3
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Fully Connected Layer
    Flatten(),

    Dense(
        256,
        activation="relu"
    ),
    Dropout(0.5),

    # Output Layer (7 kelas)
    Dense(
        7,
        activation="softmax"
    )
])

model.summary()

# ==========================================
# Compile Model
# ==========================================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# Training
# ==========================================
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=30
)

# ==========================================
# Evaluation
# ==========================================
loss, accuracy = model.evaluate(test_generator)

print(f"Accuracy : {accuracy:.4f}")
print(f"Loss     : {loss:.4f}")

# ==========================================
# Save Model
# ==========================================
model.save(
    "emotion_cnn.h5"
)

print("Model berhasil disimpan.")