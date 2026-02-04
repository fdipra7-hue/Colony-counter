import streamlit as st
import cv2
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Farhan's Coliform Analyzer", layout="wide")
st.title("🔬 Fecal Coliform Colony Counter")
st.markdown("Developed by: **Farhan** | Civil Engineering Dept.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Analysis Parameters")
# These sliders allow your Sir to tune the sensitivity live
min_area = st.sidebar.slider("Min Colony Area", 5, 500, 25)
max_area = st.sidebar.slider("Max Colony Area", 1000, 20000, 8000)
min_circ = st.sidebar.slider("Min Circularity", 0.1, 1.0, 0.45)
threshold_val = st.sidebar.slider("Watershed Sensitivity", 0.05, 0.9, 0.25)

# --- IMAGE UPLOAD (Fixed Function Name) ---
uploaded_file = st.file_uploader("Upload Petri Dish Photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    output = img.copy()

    # --- IMAGE PROCESSING PIPELINE ---
    
    # 1. CLAHE Enhancement (Contrast)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_img = cv2.merge((cl, a, b_chan))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

    # 2. Color Masking (HSV)
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array([80, 25, 20]), np.array([135, 255, 255]))
    mask_purple = cv2.inRange(hsv, np.array([136, 25, 20]), np.array([180, 255, 255]))
    full_mask = cv2.bitwise_or(mask_blue, mask_purple)

    # 3. Watershed (Separating touching colonies)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, threshold_val * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    # Area where we are unsure if it's background or foreground
    unknown = cv2.subtract(cv2.dilate(opening, kernel, iterations=3), sure_fg)
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # 4. Count & Label Logic
    blue_count, purple_count = 0, 0
    for label in range(2, np.max(markers) + 1):
        colony_mask = np.zeros(full_mask.shape, dtype="uint8")
        colony_mask[markers == label] = 255
        cnts, _ = cv2.findContours(colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts:
            cnt = cnts[0]
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circ = (4 * np.pi * area) / (peri**2)
            
            if (min_area < area < max_area) and (circ > min_circ):
                # Check which color dominates in this specific colony area
                b_pixels = cv2.countNonZero(cv2.bitwise_and(mask_blue, colony_mask))
                p_pixels = cv2.countNonZero(cv2.bitwise_and(mask_purple, colony_mask))
                
                if b_pixels > p_pixels:
                    blue_count += 1
                    color = (255, 0, 0) # Blue
                else:
                    purple_count += 1
                    color = (255, 0, 255) # Magenta/Purple
                
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(output, (int(x), int(y)), int(radius) + 5, color, 3)

    # --- DISPLAY RESULTS ---
    st.subheader("Results Dashboard")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Blue Colonies", blue_count)
    res_col2.metric("Purple Colonies", purple_count)
    res_col3.metric("Total Count", blue_count + purple_count)

    # Show processed image (Convert BGR to RGB for Streamlit)
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Analysis Result", use_container_width=True)

    # 5. Scientific Summary for Report
    with st.expander("See Mathematical Analysis Logic"):
        st.write(f"**Circular Filter:** Applied at {min_circ}")
        st.write(f"**Watershed Peak Detection:** {threshold_val}")
        st.latex(r"Circularity = \frac{4\pi \cdot Area}{Perimeter^2}")