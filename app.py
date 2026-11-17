import streamlit as st
import pandas as pd
import tempfile
import cv2

from pathlib import Path

from src.pothole_detector import PotholeDetector
from src.severity_estimator import estimate_severity
from src.database import save_detection, get_all_detections


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ROADWATCH AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. GLOBAL LIGHT THEME
# =====================================================

st.markdown("""
<style>

/* =========================================
   ROADWATCH AI - CLEAN LIGHT UI
========================================= */

/* PAGE */

.stApp {
    background: #F5F7FB !important;
    color: #172033 !important;
}

[data-testid="stHeader"] {
    background: #F5F7FB !important;
}

.main .block-container {
    max-width: 1450px;
    padding: 2rem 3rem 3rem;
}

/* HEADINGS */

h1, h2, h3, h4, h5, h6 {
    color: #172554 !important;
    font-weight: 800 !important;
}

/* TEXT */

p, label, li {
    color: #334155 !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #EDF2F9 !important;
    border-right: 1px solid #DCE5F0 !important;
}

[data-testid="stSidebar"] * {
    color: #172554 !important;
}

/* BRAND */

.brand {
    font-size: 42px;
    font-weight: 900;
    color: #172554 !important;
    letter-spacing: -1px;
}

.brand span {
    color: #2563EB !important;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 3px 12px rgba(15,23,42,0.04);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #172554 !important;
    font-size: 30px !important;
    font-weight: 850 !important;
}

/* =========================================
   UPLOAD PANEL
========================================= */

[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px dashed #93B4E8 !important;
    border-radius: 14px !important;
    padding: 12px !important;
}

/* Upload dropzone */

[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    border: none !important;
}

/* Upload instructions */

[data-testid="stFileUploaderDropzone"] * {
    color: #334155 !important;
}

/* Upload button */

[data-testid="stFileUploaderDropzone"] button {
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 8px !important;
}

/* Uploaded file display */

[data-testid="stFileUploaderFile"] {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

/* Filename */

[data-testid="stFileUploaderFile"] * {
    color: #172554 !important;
}

/* Remove button */

[data-testid="stFileUploaderFile"] button {
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
    border: none !important;
}

/* =========================================
   INPUTS
========================================= */

input, textarea {
    background: #FFFFFF !important;
    color: #172033 !important;
}

/* =========================================
   BUTTONS
========================================= */

.stButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    min-height: 46px;
    font-size: 15px !important;
    font-weight: 750 !important;
}

.stButton > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background: #1D4ED8 !important;
}

/* DOWNLOAD BUTTON */

.stDownloadButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    min-height: 45px;
    font-weight: 750 !important;
}

.stDownloadButton > button * {
    color: #FFFFFF !important;
}

/* =========================================
   INFORMATION MESSAGES
========================================= */

[data-testid="stAlert"] {
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"] * {
    color: #1E3A8A !important;
}

/* =========================================
   SEVERITY INDICATORS
========================================= */

.severity-high {
    background: #FEF2F2 !important;
    color: #991B1B !important;
    border: 1px solid #FECACA;
    border-radius: 9px;
    padding: 13px;
    font-weight: 750;
}

.severity-medium {
    background: #FFFBEB !important;
    color: #92400E !important;
    border: 1px solid #FDE68A;
    border-radius: 9px;
    padding: 13px;
    font-weight: 750;
}

.severity-low {
    background: #F0FDF4 !important;
    color: #166534 !important;
    border: 1px solid #BBF7D0;
    border-radius: 9px;
    padding: 13px;
    font-weight: 750;
}

/* SUCCESS MESSAGE */

div[style*="background:#F0FDF4"] {
    background: #F0FDF4 !important;
    color: #166534 !important;
    border: 1px solid #BBF7D0 !important;
}

/* RESULT CARDS */

.result-card {
    background: #FFFFFF !important;
    color: #172554 !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px;
    padding: 20px;
}

/* IMAGES */

[data-testid="stImage"] img {
    border-radius: 12px;
}

/* TABLES */

[data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
}

/* CHARTS */

[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
    background: #FFFFFF !important;
    border-radius: 12px !important;
}

/* DIVIDERS */

hr {
    border-color: #E2E8F0 !important;
}

/* CAPTIONS */

.stCaption {
    color: #64748B !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED YOLO MODEL
# =====================================================

@st.cache_resource
def load_model():
    return PotholeDetector()


try:
    detector = load_model()
    model_loaded = True
    model_error = None

except Exception as error:
    detector = None
    model_loaded = False
    model_error = str(error)


# =====================================================
# 4. NAVIGATION
# =====================================================

if "navigation" not in st.session_state:
    st.session_state.navigation = "Dashboard"


def go_to_analysis():
    st.session_state.navigation = "Image Analysis"


# =====================================================
# 5. HEADER
# =====================================================

st.markdown(
    '<p style="color:#2563EB;font-weight:800;'
    'letter-spacing:3px;font-size:12px;">'
    'INTELLIGENT ROAD MONITORING PLATFORM</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="brand">ROADWATCH <span>AI.</span></div>',
    unsafe_allow_html=True
)

st.caption(
    "Computer vision powered pothole detection and road damage analysis."
)

st.divider()


# =====================================================
# 6. SIDEBAR
# =====================================================

st.sidebar.markdown("## 🚧 ROADWATCH AI")
st.sidebar.caption("ROAD INTELLIGENCE SYSTEM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Image Analysis",
        "Detection History",
        "About Project"
    ],
    key="navigation"
)

st.sidebar.divider()

st.sidebar.subheader("SYSTEM STATUS")

if model_loaded:
    st.sidebar.markdown(
        "🟢 **YOLOv8 Model Loaded**"
    )
else:
    st.sidebar.markdown(
        "🔴 **Model Not Available**"
    )
    st.sidebar.caption(model_error)

st.sidebar.divider()

st.sidebar.markdown("**TECHNOLOGY STACK**")
st.sidebar.write("YOLOv8")
st.sidebar.write("OpenCV")
st.sidebar.write("Python")
st.sidebar.write("Streamlit")
st.sidebar.write("SQLite")

st.sidebar.divider()
st.sidebar.caption("ROADWATCH AI | VERSION 1.0")


# =====================================================
# 7. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("Road Intelligence Dashboard")

    st.caption(
        "Monitor pothole detection activity and inspection results."
    )

    records = get_all_detections()

    if records:

        df = pd.DataFrame(records)

        total = len(df)

        high = int((df["severity"] == "HIGH").sum())
        medium = int((df["severity"] == "MEDIUM").sum())
        low = int((df["severity"] == "LOW").sum())

        avg_confidence = df["confidence"].mean() * 100

    else:

        total = 0
        high = 0
        medium = 0
        low = 0
        avg_confidence = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("TOTAL RECORDS", total)
    c2.metric("HIGH SEVERITY", high)
    c3.metric("MEDIUM SEVERITY", medium)
    c4.metric("AVERAGE CONFIDENCE", f"{avg_confidence:.2f}%")

    st.divider()

    left, right = st.columns([1.2, 0.8])

    with left:

        st.subheader("Recent Detection Activity")

        if records:

            st.dataframe(
                pd.DataFrame(records).head(8),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No detection records available yet.")

    with right:

        st.subheader("Severity Distribution")

        if records:

            chart = pd.DataFrame({
                "Severity": ["HIGH", "MEDIUM", "LOW"],
                "Detections": [high, medium, low]
            })

            st.bar_chart(
                chart.set_index("Severity"),
                color="#2563EB"
            )

        else:

            st.info("Chart will appear after image analysis.")

    st.divider()

    st.subheader("Start a New Inspection")

    st.write(
        "Upload a road image and analyze it using the trained YOLOv8 model."
    )

    st.button(
        "OPEN IMAGE ANALYSIS",
        use_container_width=True,
        on_click=go_to_analysis
    )


# =====================================================
# 8. IMAGE ANALYSIS
# =====================================================

elif page == "Image Analysis":

    st.title("Image Analysis")

    st.caption(
        "Upload a road image to detect potholes and estimate their severity."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "UPLOAD ROAD IMAGE",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear road image.",
        key="road_image_upload"
    )

    if uploaded_file:

        left, right = st.columns([1.2, 0.8], gap="large")

        with left:

            st.subheader("Input Image")

            st.image(
                uploaded_file.getvalue(),
                caption=uploaded_file.name,
                use_container_width=True
            )

        with right:

            st.subheader("Analysis Settings")

            st.metric("DETECTION ENGINE", "YOLOv8n")
            st.metric("TASK TYPE", "Object Detection")

            st.info(
                "The trained model identifies potholes using bounding boxes."
            )

        st.write("")

        analyze = st.button(
            "ANALYZE ROAD IMAGE",
            use_container_width=True,
            disabled=not model_loaded
        )

        if analyze:

            temp_path = None

            try:

                with st.spinner(
                    "ROADWATCH AI is analyzing your image..."
                ):

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_file.name).suffix
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    detections, annotated_image = detector.detect(
                        temp_path
                    )

                st.divider()

                st.title("Detection Results")

                total = len(detections)

                if total > 0:

                    average_confidence = (
                        sum(d["confidence"] for d in detections)
                        / total
                    ) * 100

                    largest_area = max(
                        d["area_percentage"] for d in detections
                    )

                    overall_severity = estimate_severity(largest_area)

                    # SUMMARY METRICS

                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "POTHOLES DETECTED",
                        total
                    )

                    m2.metric(
                        "AVERAGE CONFIDENCE",
                        f"{average_confidence:.2f}%"
                    )

                    m3.metric(
                        "HIGHEST SEVERITY",
                        overall_severity
                    )

                    st.divider()

                    # ANNOTATED IMAGE

                    st.subheader("YOLOv8 Detection Output")

                    st.image(
                        annotated_image,
                        channels="BGR",
                        use_container_width=True,
                        caption="Detected potholes with bounding boxes"
                    )

                    # INDIVIDUAL ANALYSIS

                    st.divider()

                    st.subheader("Individual Pothole Analysis")

                    for index, detection in enumerate(
                        detections,
                        start=1
                    ):

                        severity = estimate_severity(
                            detection["area_percentage"]
                        )

                        st.markdown(f"### Pothole {index}")

                        a, b, c = st.columns(3)

                        a.metric(
                            "CONFIDENCE SCORE",
                            f"{detection['confidence'] * 100:.2f}%"
                        )

                        b.metric(
                            "IMAGE AREA",
                            f"{detection['area_percentage']:.2f}%"
                        )

                        c.metric(
                            "SEVERITY",
                            severity
                        )

                        # LIGHT SEVERITY INDICATORS

                        if severity == "HIGH":

                            st.markdown(
                                '<div class="severity-high">'
                                '🔴 HIGH SEVERITY — Large image-area estimate'
                                '</div>',
                                unsafe_allow_html=True
                            )

                        elif severity == "MEDIUM":

                            st.markdown(
                                '<div class="severity-medium">'
                                '🟠 MEDIUM SEVERITY — Moderate image-area estimate'
                                '</div>',
                                unsafe_allow_html=True
                            )

                        else:

                            st.markdown(
                                '<div class="severity-low">'
                                '🟢 LOW SEVERITY — Small image-area estimate'
                                '</div>',
                                unsafe_allow_html=True
                            )

                        # SAVE RECORD

                        save_detection(
                            image_name=uploaded_file.name,
                            confidence=detection["confidence"],
                            area_percentage=detection["area_percentage"],
                            severity=severity
                        )

                        st.divider()

                    # LIGHT SUCCESS MESSAGE

                    st.markdown(
                        """
                        <div style="
                            background:#F0FDF4;
                            color:#166534;
                            border:1px solid #BBF7D0;
                            border-radius:12px;
                            padding:16px;
                            font-weight:700;
                        ">
                            ✓ Analysis completed successfully.
                            Results saved to database.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # DOWNLOAD IMAGE

                    success, image_buffer = cv2.imencode(
                        ".jpg",
                        annotated_image
                    )

                    if success:

                        st.download_button(
                            "⬇ DOWNLOAD ANNOTATED IMAGE",
                            data=image_buffer.tobytes(),
                            file_name="roadwatch_detection.jpg",
                            mime="image/jpeg"
                        )

                else:

                    st.warning(
                        "No potholes detected in this image."
                    )

                    st.info(
                        "Try a clear image containing visible road potholes."
                    )

            except Exception as error:

                st.error(f"Analysis failed: {error}")

            finally:

                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)

    else:

        st.info(
            "Upload a JPG or PNG road image to begin your inspection."
        )


# =====================================================
# 9. DETECTION HISTORY
# =====================================================

elif page == "Detection History":

    st.title("Detection History")

    st.caption(
        "Review previous inspections and export detection records."
    )

    records = get_all_detections()

    if records:

        df = pd.DataFrame(records)

        c1, c2, c3 = st.columns(3)

        c1.metric("TOTAL RECORDS", len(df))

        c2.metric(
            "HIGH SEVERITY",
            int((df["severity"] == "HIGH").sum())
        )

        c3.metric(
            "LOW SEVERITY",
            int((df["severity"] == "LOW").sum())
        )

        st.divider()

        st.subheader("Stored Detection Records")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ DOWNLOAD CSV REPORT",
            data=csv_data,
            file_name="roadwatch_detection_history.csv",
            mime="text/csv"
        )

    else:

        st.info("No detection records available.")


# =====================================================
# 10. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.title("About ROADWATCH AI")

    st.caption(
        "AI-powered road damage identification and analysis."
    )

    st.subheader("Project Overview")

    st.write(
        """
        ROADWATCH AI is a computer vision application that uses a
        trained YOLOv8 model to identify potholes in uploaded road images.
        """
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("01. Pothole Detection")

        st.write(
            "Identifies potholes and localizes them using bounding boxes."
        )

        st.subheader("02. Confidence Analysis")

        st.write(
            "Displays the model confidence score for each detection."
        )

    with c2:

        st.subheader("03. Severity Estimation")

        st.write(
            "Uses bounding-box image area to estimate LOW, MEDIUM and HIGH."
        )

        st.subheader("04. Database Storage")

        st.write(
            "Stores detection results using SQLite."
        )

    st.divider()

    st.warning(
        "Severity is an image-area-based heuristic. "
        "It does not measure actual pothole depth or physical dimensions."
    )


# =====================================================
# 11. FOOTER
# =====================================================

st.divider()

st.caption(
    "ROADWATCH AI | COMPUTER VISION ROAD INSPECTION | YOLOv8"
)