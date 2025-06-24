import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageOps
import base64

# ---------- Helper: Encode Background Image ----------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode('utf-8')

# ---------- Set Paths ----------
background_image_path = (r"C:\Users\quant\Projects\DL_Projects\DL_Model\plexus-network-abstract-technology-technology-abstract-animation-business-network-concept-abstract-plexus-of-moving-glowing-dots-and-lines-technologic-network-background-free-video.jpg")
model_path = (r"C:\Users\quant\Projects\DL_Projects\DL_Model\mnist_model.h5")

# ---------- Page Config ----------
st.set_page_config(page_title="Digit Recognizer", layout="centered")

# ---------- Load Background CSS ----------
bg_img_base64 = get_base64_image(background_image_path)
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}

    .main {{
        background-color: rgba(255, 255, 255, 0.08);
        padding: 2rem;
        border-radius: 1rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        max-width: 800px;
        margin: auto;
        color: #ffffff;
    }}

    h1, h2, h3 {{
        color: #ffffff;
        text-align: center;
    }}

    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        padding: 0.5em 1em;
        font-size: 16px;
        border: none;
        border-radius: 0.5em;
    }}

    .stFileUploader label {{
        color: white;
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------- App Container ----------
st.title("🧠 Handwritten Digit Recognizer")
st.markdown("""
<div class="main">
    <h6 style="text-align:center; font-size:16px; color:#ccc;">
            🤖 This car price predictor is powered by a machine learning model developed by Rocket based Company Known as QubitSpace.
    </h6>
</div>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# ---------- Load MNIST Data ----------
@st.cache_data
def load_mnist_data():
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0
    return x_test, y_test

x_test, y_test = load_mnist_data()

# ---------- User Input Selection ----------
option = st.radio("Select input method:", ("Pick from MNIST test set", "Upload your own image"))

# ---------- Option 1: Pick from test set ----------
if option == "Pick from MNIST test set":
    index = st.slider("Select test image index", 0, len(x_test) - 1, 0)
    img = x_test[index]
    img_flat = img.reshape(1, 784)

    st.subheader("MNIST Test Image")
    st.image(img, width=150)

    prediction = model.predict(img_flat)[0]
    predicted_label = np.argmax(prediction)

    st.write(f"### Predicted Digit: **{predicted_label}** (True: {y_test[index]})")
    st.subheader("Prediction Confidence")
    st.bar_chart(prediction)

# ---------- Option 2: Upload Image ----------
elif option == "Upload your own image":
    uploaded_file = st.file_uploader("Upload an image (28x28 grayscale digit)", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))
        img = ImageOps.invert(img)  # Invert for MNIST style
        img_array = np.array(img) / 255.0
        img_flat = img_array.reshape(1, 784)

        st.subheader("Uploaded Image (Processed)")
        st.image(img, width=150)

        prediction = model.predict(img_flat)[0]
        predicted_label = np.argmax(prediction)

        st.write(f"### Predicted Digit: **{predicted_label}**")
        st.subheader("Prediction Confidence")
        st.bar_chart(prediction)
    else:
        st.info("Upload a digit image to predict.")

# ---------- Close App Container ----------
st.markdown('</div>', unsafe_allow_html=True)
