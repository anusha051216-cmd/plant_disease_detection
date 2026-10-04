import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# PLANT DISEASE DETECTION - CNN
# ==========================================

# Settings
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

DATASET_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "plant_disease_model.keras"
)

# Create model folder
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n🌱 Plant Disease Detection - Deep Learning")
print("=" * 55)


# ==========================================
# DATA AUGMENTATION
# ==========================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.20,
    rotation_range=20,
    width_shift_range=0.20,
    height_shift_range=0.20,
    zoom_range=0.20,
    shear_range=0.15,
    horizontal_flip=True
)


# ==========================================
# TRAINING DATA
# ==========================================

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=42
)


# ==========================================
# VALIDATION DATA
# ==========================================

validation_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=42
)


# ==========================================
# DISPLAY CLASS INFORMATION
# ==========================================

print("\nClass indices:")
print(train_data.class_indices)

print("\nTraining images:", train_data.samples)
print("Validation images:", validation_data.samples)


# ==========================================
# CNN MODEL
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    # Convolution Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 4
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    # Flatten
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout
    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# DISPLAY MODEL
# ==========================================

print("\nCNN Model:")
model.summary()


# ==========================================
# TRAIN MODEL
# ==========================================

print("\n🚀 Training started...\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)


# ==========================================
# EVALUATE MODEL
# ==========================================

print("\n📊 Evaluating model...")

validation_data.reset()

loss, accuracy = model.evaluate(
    validation_data,
    verbose=0
)

print("\nValidation Accuracy:",
      round(accuracy * 100, 2), "%")


# ==========================================
# PREDICTIONS
# ==========================================

validation_data.reset()

probabilities = model.predict(
    validation_data,
    verbose=0
)

predictions = (
    probabilities >= 0.5
).astype(int).flatten()

true_labels = validation_data.classes


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")
print()

print(
    classification_report(
        true_labels,
        predictions,
        target_names=[
            "Diseased",
            "Healthy"
        ],
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        true_labels,
        predictions
    )
)


# ==========================================
# SAVE KERAS MODEL
# ==========================================

model.save(MODEL_PATH)

print("\n✅ Model saved successfully!")
print("📁 Model path:", MODEL_PATH)

print("\n🎉 CNN training completed!")