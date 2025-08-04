import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# =============================
# Load models (update paths if needed)
# =============================
nationality_model = load_model("/content/nationality_model.h5", compile=False)
age_model = load_model("/content/age_model.h5", compile=False)
emotion_model = load_model("/content/emotion_model.h5", compile=False)

# Re-compile
age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nationality_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# =============================
# Helper functions
# =============================
def preprocess_img(img, target_size=(128, 128)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

def preprocess_emotion(img, target_size=(48, 48)):
    img_gray = img.convert("L")
    img_gray = img_gray.resize(target_size)
    img_array = np.array(img_gray) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

def predict_nationality(img_array):
    pred = nationality_model.predict(np.expand_dims(img_array, axis=0))[0]
    classes = ["Indian", "United States", "African", "Other"]
    return classes[np.argmax(pred)]

def predict_age(img_array):
    pred = age_model.predict(np.expand_dims(img_array, axis=0))[0][0]
    return int(pred)

def predict_emotion(img_array):
    pred = emotion_model.predict(np.expand_dims(img_array, axis=0))[0]
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    return classes[np.argmax(pred)]

# =============================
# Streamlit UI
# =============================
st.title("👤 Nationality, Age, Emotion & Dress Color Detector")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array_norm = preprocess_img(image)
    img_array_emotion = preprocess_emotion(image)

    # Predict nationality
    nationality = predict_nationality(img_array_norm)

    # Predict emotion
    emotion = predict_emotion(img_array_emotion)

    # Predict age if Indian or US
    if nationality in ["Indian", "United States"]:
        age = predict_age(img_array_norm)
    else:
        age = "Not predicted"

    # Dress color if Indian or African
    if nationality in ["Indian", "African"]:
        img_resized = np.array(image.resize((100, 100)))
        avg_color = tuple(np.mean(img_resized.reshape(-1, 3), axis=0).astype(int))
        dress_color = f"RGB{avg_color}"
    else:
        dress_color = "Not predicted"

    # ====================
    # Show results
    # ====================
    st.markdown("---")
    st.subheader("Prediction Results")
    st.write(f"**Nationality**: {nationality}")
    st.write(f"**Emotion**: {emotion}")

    if nationality in ["Indian", "United States"]:
        st.write(f"**Age**: {age}")

    if nationality in ["Indian", "African"]:
        st.write(f"**Dress Color**: {dress_color}")

    if nationality not in ["Indian", "United States", "African"]:
        st.write("(Age and Dress Color not predicted for this nationality)")
