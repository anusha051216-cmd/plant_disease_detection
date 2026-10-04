import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 🌱 PLANT DISEASE DETECTION - CNN
# ============================================================

print("\n🌱 Plant Disease Detection - Deep Learning")
print("=" * 60)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

DATASET_DIR = "dataset"
MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "plant_disease_model.keras"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

print("\nDataset folder:", DATASET_DIR)


# ============================================================
# TRAINING DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,

    validation_split=0.20,

    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True
)


# ============================================================
# TRAINING DATA
# ============================================================

train_generator = train_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training",

    shuffle=True,

    seed=42
)


# ============================================================
# VALIDATION DATA
# ============================================================

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.20
)


validation_generator = validation_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation",

    shuffle=False,

    seed=42
)


# ============================================================
# CLASS INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("CLASS INFORMATION")
print("=" * 60)

print("\nClass indices:")
print(train_generator.class_indices)

class_names = list(train_generator.class_indices.keys())

print("\nClass mapping:")

for name, index in train_generator.class_indices.items():
    print(f"{index} = {name}")

print("\nTraining images:", train_generator.samples)
print("Validation images:", validation_generator.samples)

print("=" * 60)


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

with open(CLASS_PATH, "w") as f:
    json.dump(
        train_generator.class_indices,
        f,
        indent=4
    )

print("\nClass mapping saved to:")
print(CLASS_PATH)


# ============================================================
# CNN MODEL
# ============================================================

print("\nBuilding CNN model...")


model = models.Sequential([

    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    # Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Block 4
    layers.Conv2D(
        256,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Classification
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nCNN Model:")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.000001
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)


history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)


print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("Model:")
print(MODEL_PATH)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


loss, accuracy = model.evaluate(
    validation_generator
)


print("\nValidation Loss:", round(loss, 4))

print(
    "Validation Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")


validation_generator.reset()


predictions = model.predict(
    validation_generator
)


predicted_classes = (
    predictions >= 0.5
).astype(int).flatten()


true_classes = validation_generator.classes


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)


cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print(cm)


# ============================================================
# FINAL MAPPING
# ============================================================

print("\n" + "=" * 60)
print("FINAL CLASS MAPPING")
print("=" * 60)


for name, index in train_generator.class_indices.items():

    print(
        f"Model class {index} = {name}"
    )


print("\nPrediction rule:")

print("Probability < 0.5  → DISEASED")

print("Probability >= 0.5 → HEALTHY")


print("\n✅ Training completed successfully!")

print("🌱 Plant Disease Detection model is ready!")