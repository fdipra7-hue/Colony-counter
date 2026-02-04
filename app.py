import streamlit as st
import cv2
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Farhan's Coliform Analyzer", layout="wide")
st.title("🔬 Fecal Coliform Colony Counter")

# --- SIDEBAR: CROP CONTROLS ---
st.sidebar.header("1. Crop Settings")
# These sliders will crop the image by percentage from each side
top_crop = st.sidebar.slider("Crop Top %", 0, 50, 0)
bottom_crop = st.sidebar.slider("Crop Bottom %", 0, 50, 0)
left_crop = st.sidebar.slider("Crop Left %", 0, 50, 0)
right_crop = st.sidebar.slider("Crop Right %", 0, 50, 0)

st.sidebar.header("2. Analysis Parameters")
min_area = st.sidebar.slider("Min Colony Area", 5, 500, 25)
max_area = st.sidebar.slider("Max Colony Area", 1000, 20000, 8000)
min_circ = st.sidebar.slider("Min Circularity", 0.1, 1.0, 0.45)
threshold_val = st.sidebar.slider("Watershed Sensitivity", 0.05, 0.9, 0.25)

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Upload Petri Dish Photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # --- CROP LOGIC ---
    h, w = img.shape[:2]
    # Calculate pixel boundaries based on percentage sliders
    y1, y2 = int(h * top_crop / 100), int(h * (100 - bottom_crop) / 100)
    x1, x2 = int(w * left_crop / 100), int(w * (100 - right_crop) / 100)
    
    # Slice the image
    img = img[y1:y2, x1:x2]
    output = img.copy()

    # --- CONTINUE PROCESSING (CLAHE, Masking, Watershed...) ---
    # [Insert the rest of the detection code here]
    
    # ... (After processing) ...
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Cropped & Analyzed Result", use_container_width=True)
