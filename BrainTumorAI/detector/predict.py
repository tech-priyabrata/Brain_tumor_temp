from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "brain_tumor_model.keras"

print("Loading Brain Tumor AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Brain Tumor AI model loaded successfully!")

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

def predict_image(image_path):

    img = image.load_img(
        image_path,
        target_size=(160, 160)
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = float(
        predictions[0][predicted_index]
    ) * 100

    return predicted_class, confidence