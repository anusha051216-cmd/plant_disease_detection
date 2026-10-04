import tensorflow as tf
import numpy as np
from tkinter import Tk, filedialog
from tensorflow.keras.preprocessing import image


# ============================================================
# 🌱 PLANT DISEASE DETECTION - PREDICTION
# ============================================================

print("\n🌱 Plant Disease Detection")
print("=" * 50)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "model/plant_disease_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded successfully")


# ============================================================
# SELECT IMAGE
# ============================================================

root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select a plant image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png")
    ]
)

root.destroy()


if not file_path:
    print("\n❌ No image selected.")
    exit()


print("\nSelected image:")
print(file_path)


# ============================================================
# LOAD IMAGE
# ============================================================

IMG_SIZE = 128

img = image.load_img(
    file_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

# Same preprocessing used during training
img_array = img_array / 255.0


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    img_array,
    verbose=0
)

# Sigmoid output = probability of class 1
healthy_probability = float(prediction[0][0])

healthy_percentage = healthy_probability * 100
diseased_percentage = (1 - healthy_probability) * 100


# ============================================================
# DISPLAY PROBABILITIES
# ============================================================

print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

print(
    f"Healthy probability : {healthy_percentage:.2f}%"
)

print(
    f"Diseased probability: {diseased_percentage:.2f}%"
)


# ============================================================
# CLASSIFICATION
# ============================================================

if healthy_probability >= 0.5:

    result = "HEALTHY"

    confidence = healthy_percentage

else:

    result = "DISEASED"

    confidence = diseased_percentage


print("\n🌱 Final Prediction:", result)

print(
    f"📊 Confidence: {confidence:.2f}%"
)


# ============================================================
# TREATMENT SUGGESTION
# ============================================================

if result == "DISEASED":

    print("\n" + "=" * 50)
    print("🩺 TREATMENT SUGGESTION")
    print("=" * 50)

    print("• Remove severely infected leaves.")
    print("• Keep the plant area clean and dry.")
    print("• Avoid excessive watering.")
    print("• Improve air circulation around the plant.")
    print("• Use an appropriate fungicide or pesticide")
    print("  based on the confirmed disease and crop.")
    print("• Consult a local agricultural expert for")
    print("  disease-specific treatment.")


else:

    print("\n" + "=" * 50)
    print("✅ PLANT STATUS")
    print("=" * 50)

    print("The plant appears healthy.")
    print("Continue proper watering, sunlight and care.")


print("\n" + "=" * 50)
print("✅ Prediction completed")
print("=" * 50)