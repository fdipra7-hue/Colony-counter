import streamlit as st
import cv2
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Farhan's Coliform Analyzer", layout="wide")
st.title("🔬 Fecal Coliform Colony Counter")
st.markdown("Developed by: **Farhan** | Civil Engineering Dept.")

# --- SIDEBAR: CROP & PARAMETERS ---
st.sidebar.header("1. Crop Settings (Remove Edges)")
top_c = st.sidebar.slider("Crop Top %", 0, 50, 5)
bottom_c = st.sidebar.slider("Crop Bottom %", 0, 50, 5)
left_c = st.sidebar.slider("Crop Left %", 0, 50, 5)
right_c = st.sidebar.slider("Crop Right %", 0, 50, 5)

st.sidebar.header("2. Detection Sensitivity")
min_area = st.sidebar.slider("Min Colony Size", 5, 500, 30)
max_area = st.sidebar.slider("Max Colony Size", 1000, 20000, 8000)
min_circ = st.sidebar.slider("Circularity (0.1=Blurry, 0.9=Circle)", 0.1, 1.0, 0.45)
ws_threshold = st.sidebar.slider("Cluster Separation", 0.05, 0.9, 0.25)

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Upload Petri Dish Photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_raw = cv2.imdecode(file_bytes, 1)
    
    # 1. APPLY CROP
    h, w = img_raw.shape[:2]
    y1, y2 = int(h * top_c / 100), int(h * (100 - bottom_c) / 100)
    x1, x2 = int(w * left_c / 100), int(w * (100 - right_c) / 100)
    img = img_raw[y1:y2, x1:x2]
    output = img.copy()

    # 2. ENHANCE CONTRAST (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b_chan))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # 3. MASKING (Blue & Purple)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask_b = cv2.inRange(hsv, np.array([80, 25, 20]), np.array([135, 255, 255]))
    mask_p = cv2.inRange(hsv, np.array([136, 25, 20]), np.array([180, 255, 255]))
    full_mask = cv2.bitwise_or(mask_b, mask_p)

    # 4. WATERSHED (The 'Cluster-Breaker')
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

    # 5. COUNTING LOOP
    b_count, p_count = 0, 0
    for label in range(2, np.max(markers) + 1):
        colony_m = np.zeros(full_mask.shape, dtype="uint8")
        colony_m[markers == label] = 255
        cnts, _ = cv2.findContours(colony_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts:
            c = cnts[0]
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            if peri == 0: continue
            circ = (4 * np.pi * area) / (peri**2)
            
            if (min_area < area < max_area) and (circ > min_circ):
                b_pix = cv2.countNonZero(cv2.bitwise_and(mask_b, colony_m))
                p_pix = cv2.countNonZero(cv2.bitwise_and(mask_p, colony_m))
                
                if b_pix > p_pix:
                    b_count += 1
                    col = (255, 0, 0) # Blue tag
                else:
                    p_count += 1
                    col = (255, 0, 255) # Purple tag
                
                (x, y), rad = cv2.minEnclosingCircle(c)
                cv2.circle(output, (int(x), int(y)), int(rad) + 5, col, 3)

    # 6. DISPLAY DASHBOARD
    st.subheader("Results Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Blue Colonies", b_count)
    c2.metric("Purple Colonies", p_count)
    c3.metric("Total Coliform", b_count + p_count)

    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    st.info("Please upload an image to start the analysis.")
