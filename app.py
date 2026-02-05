import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import pickle

# Load the trained model (MobileNetV2 version)
model = load_model("mobilenetv2_10k.h5")  # change filename if needed

# Path to validation folder (to get class labels)
val_dir = r"dataset_10k/val"  # your validation folder

# Read folder names to create class mapping
classes = sorted(os.listdir(val_dir))
class_labels = {i: cls for i, cls in enumerate(classes)}


st.title("🌱 Crop Disease Detection App")
st.write("Upload an image of your crop leaf and the app will tell you the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])


if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess for model
    img_array = img.resize((128,128))  # resize to your model input size
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)/255.0
    
    # Predict
    pred = model.predict(img_array)
    predicted_index = np.argmax(pred)
    confidence = np.max(pred)*100
    predicted_label = class_labels[predicted_index]
    
    # Split crop and disease name
    if "___" in predicted_label:
        crop_name, disease_name = predicted_label.split("___", 1)
    elif "_" in predicted_label:
        parts = predicted_label.split("_",1)
        crop_name = parts[0]
        disease_name = parts[1] if len(parts) > 1 else "Healthy / Not specified"
    else:
        crop_name = predicted_label
        disease_name = "Healthy / Not specified"
    
    # Show results
    st.success(f"🌱 Crop Detected: {crop_name}")
    if "healthy" in disease_name.lower():
        st.success(f"✅ Your {crop_name} crop is **healthy!** No disease detected.")
    else:
        st.error(f"🦠 Disease Detected: {disease_name}")
    st.info(f"🤖 Confidence: {confidence:.2f}%")

# Load history if available
if os.path.exists("history_10k.pkl"):
    with open("history_10k.pkl","rb") as f:
        hist = pickle.load(f)
    
    st.subheader("📊 Training Metrics")
    st.line_chart({
        "accuracy": hist["accuracy"],
        "val_accuracy": hist["val_accuracy"],
        "loss": hist["loss"],
        "val_loss": hist["val_loss"]
    })
