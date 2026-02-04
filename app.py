import streamlit as st
import cv2
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Farhan's Coliform Analyzer", layout="wide")
st.title("🔬 Fecal Coliform Colony Counter")
st.markdown("Developed by: **Farhan** | Civil Engineering Dept.")

# --- SIDEBAR ---
st.sidebar.header("1. Crop Settings")
top_c = st.sidebar.slider("Crop Top %", 0, 50, 5)
bottom_c = st.sidebar.slider("Crop Bottom %", 0, 50, 5)
left_c = st.sidebar.slider("Crop Left %", 0, 50, 5)
right_c = st.sidebar.slider("Crop Right %", 0, 50, 5)

st.sidebar.header("2. Deep Blue Detection (Black-Blue)")
b_hue = st.sidebar.slider("Blue Hue Range", 0, 180, (75, 135))
b_val_min = st.sidebar.slider("Blue Brightness Floor (Value)", 0, 255, 10)
b_sat_min = st.sidebar.slider("Blue Saturation Floor", 0, 255, 15)

st.sidebar.header("3. Purple Detection")
p_hue = st.sidebar.slider("Purple Hue Range", 0, 180, (136, 175))
p_val_min = st.sidebar.slider("Purple Brightness Floor", 0, 255, 20)

st.sidebar.header("4. Sensitivity")
# Updated Min Colony Size to 1
min_area = st.sidebar.slider("Min Colony Size", 1, 500, 1) 
max_area = st.sidebar.slider("Max Colony Size", 500, 50000, 8000)
ws_threshold = st.sidebar.slider("Cluster Separation", 0.05, 0.9, 0.25)

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Upload Petri Dish Photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_raw = cv2.imdecode(file_bytes, 1)
    
    # 1. CROP
    h, w = img_raw.shape[:2]
    y1, y2 = int(h * top_c / 100), int(h * (100 - bottom_c) / 100)
    x1, x2 = int(w * left_c / 100), int(w * (100 - right_c) / 100)
    img = img_raw[y1:y2, x1:x2]
    output = img.copy()

    # 2. ENHANCE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12,12))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b_chan))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # 3. MASKING
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    
    lower_b = np.array([b_hue[0], b_sat_min, b_val_min])
    upper_b = np.array([b_hue[1], 255, 255])
    mask_b = cv2.inRange(hsv, lower_b, upper_b)

    lower_p = np.array([p_hue[0], 20, p_val_min])
    upper_p = np.array([p_hue[1], 255, 255])
    mask_p = cv2.inRange(hsv, lower_p, upper_p)
    
    full_mask = cv2.bitwise_or(mask_b, mask_p)

    # 4. WATERSHED
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    dist_t = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_t, ws_threshold * dist_t.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    unknown = cv2.subtract(cv2.dilate(opening, kernel, iterations=3), sure_fg)
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # 5. COUNTING
    b_count, p_count = 0, 0
    for label in range(2, np.max(markers) + 1):
        colony_m = np.zeros(full_mask.shape, dtype="uint8")
        colony_m[markers == label] = 255
        cnts, _ = cv2.findContours(colony_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts:
            c = cnts[0]
            area = cv2.contourArea(c)
            if min_area <= area < max_area:
                b_pix = cv2.countNonZero(cv2.bitwise_and(mask_b, colony_m))
                p_pix =
