import streamlit as st
import cv2
import numpy as np
import time

# --- 1. SET THEME & CUSTOM INTERFACE DESIGN ---
st.set_page_config(page_title="ColiScan Pro Dashboard", page_icon="🔬", layout="wide")
st.markdown("Developed by: **Farhan** | Civil And Environmental Engineering Dept.-SUST")

# Injecting Custom CSS for the Navy Blue & Glass UI
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metric Card Styling (Glassmorphism) */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    /* Metric Text Colors */
    [data-testid="stMetricValue"] {
        color: #00d2ff !important;
        font-weight: bold;
    }
    
    /* Titles and Text */
    h1, h2, h3, p {
        color: white !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Buttons */
    .stButton>button {
        background-color: #00d2ff;
        color: white;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
        box-shadow: 0 0 15px #00d2ff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR CONTROLS (EXPANDERS) ---

st.sidebar.title("Configuration")

with st.sidebar.expander("📐 Crop Control", expanded=True):
    top_c = st.slider("Top %", 0, 50, 5)
    bottom_c = st.slider("Bottom %", 0, 50, 5)
    left_c = st.slider("Left %", 0, 50, 5)
    right_c = st.slider("Right %", 0, 50, 5)

with st.sidebar.expander("🎨 Color Tuning", expanded=False):
    b_hue = st.slider("Blue Hue", 0, 180, (110, 135))
    b_val = st.slider("Blue Brightness Floor", 0, 255, 0)
    b_sat = st.slider("Blue Saturation Floor", 0, 255, 150)
    st.markdown("---")
    p_hue = st.slider("Purple Hue", 0, 180, (150, 180))
    p_val = st.slider("Purple Brightness Floor", 0, 255, 35)
    p_sat = st.slider("Purple Saturation Floor", 0, 255, 25)

with st.sidebar.expander("⚙️ Sensitivity Engine", expanded=True):
    min_area = st.slider("Min Colony Size", 1, 500, 1) 
    max_area = st.slider("Max Colony Size", 500, 50000, 1000)
    ws_threshold = st.slider("Watershed Separation", 0.05, 0.9, 0.07)

# --- 3. MAIN INTERFACE ---
st.write("# 🔬 Fecal Coliform Digital Analyzer")
st.write("### Environmental Lab Utility")

uploaded_file = st.file_uploader("Drop Image Here", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Scientific Progress Indicator
    with st.status("🔍 Analyzing Micro-Colonies...", expanded=False) as status:
        st.write("Extracting HSV Channels...")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, 1)
        st.write("Applying Watershed Segmentation (Kernel 1x1)...")
        #  PROCESSING LOGIC 
        h_orig, w_orig = img_raw.shape[:2]
        y1, y2 = int(h_orig * top_c / 100), int(h_orig * (100 - bottom_c) / 100)
        x1, x2 = int(w_orig * left_c / 100), int(w_orig * (100 - right_c) / 100)
        img = img_raw[y1:y2, x1:x2]
        output = img.copy()

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b_chan)), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        mask_b = cv2.inRange(hsv, np.array([b_hue[0], b_sat, b_val]), np.array([b_hue[1], 255, 255]))
        mask_p = cv2.inRange(hsv, np.array([p_hue[0], p_sat, p_val]), np.array([p_hue[1], 255, 255]))
        full_mask = cv2.bitwise_or(mask_b, mask_p)

        kernel = np.ones((1,1), np.uint8) # 1x1 Sensitivity
        opening = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        dist_t = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
        _, sure_fg = cv2.threshold(dist_t, ws_threshold * dist_t.max(), 255, 0)
        markers = cv2.connectedComponents(np.uint8(sure_fg))[1] + 1
        markers[cv2.subtract(cv2.dilate(opening, kernel, iterations=1), np.uint8(sure_fg)) == 255] = 0
        markers = cv2.watershed(img, markers)

        b_count, p_count = 0, 0
        for label in range(2, np.max(markers) + 1):
            colony_m = np.zeros(full_mask.shape, dtype="uint8")
            colony_m[markers == label] = 255
            cnts, _ = cv2.findContours(colony_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                area = cv2.contourArea(cnts[0])
                if min_area <= area < max_area:
                    b_pix = cv2.countNonZero(cv2.bitwise_and(mask_b, colony_m))
                    p_pix = cv2.countNonZero(cv2.bitwise_and(mask_p, colony_m))
                    col = (255, 0, 0) if b_pix > p_pix else (255, 0, 255)
                    if b_pix > p_pix: b_count += 1
                    else: p_count += 1
                    (x_c, y_c), rad = cv2.minEnclosingCircle(cnts[0])
                    cv2.circle(output, (int(x_c), int(y_c)), int(rad) + 2, col, 2)
        status.update(label="Analysis Optimized", state="complete")

    #  4. DISPLAY DASHBOARD 
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("🔹 Blue Colonies", b_count)
    m2.metric("🔮 Purple Colonies", p_count)
    m3.metric("✅ Total Count", b_count + p_count)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Analysis Result", width=500)

    # Technical Expander
    with st.expander("🛠️ System Engineering Specs"):
        st.write("Segmentation: Watershed Algorithm")

else:
    st.info("System Ready. Please upload lab sample to begin scanning.")
