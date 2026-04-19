import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
import io
import torch
from torchvision import transforms
import cv2
import gdown
from classification import StrokeClassifier, StrokeTypeClassifier, predict_class
from segmentation_detection import UNet, extract_clots_from_mask
import matplotlib.pyplot as plt
import tempfile
from fpdf import FPDF

# Set page config for a wider layout and custom title
st.set_page_config(page_title="Brain CT Stroke & Clot Detection", page_icon="🧠", layout="wide")

# Custom CSS for a better aesthetic
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #333;
    }
    .metric-value-high {
        color: #ff4b4b;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-value-medium {
        color: #ffa421;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-value-low {
        color: #008f51;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-value-neutral {
        color: #ffffff;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        color: #a3a8b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Real CNN Inference Pipelines
# -----------------------------------------------------------------------------

# 1. Model Caching (Loads weights once in Streamlit)
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------
    # DOWNLOAD MODEL FILES FROM DRIVE
    # -------------------------------
    def download_file(file_id, output):
        if not os.path.exists(output):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=False)

    download_file("14IgpgBioDyohj8VFbELbTAuRxOsRx0rf", "stroke_classifier_weights.pth")
    download_file("1hGEHj_pcsAzLmUWPUQg0LqpRThnUjB61", "stroke_type_weights.pth")
    download_file("1yGySxyxfLMnjWtzO-TwAV9z9gldmZMrc", "unet_weights.pth")

    # -------------------------------
    # LOAD MODELS (UNCHANGED)
    # -------------------------------
    
    model_stroke = StrokeClassifier(num_classes=2).to(device)
    model_stroke.load_state_dict(torch.load("stroke_classifier_weights.pth", map_location=device))
    model_stroke.eval()
    
    model_type = StrokeTypeClassifier(num_classes=2).to(device)
    model_type.load_state_dict(torch.load("stroke_type_weights.pth", map_location=device))
    model_type.eval()
    
    model_unet = UNet(n_channels=3, n_classes=1).to(device)
    model_unet.load_state_dict(torch.load("unet_weights.pth", map_location=device))
    model_unet.eval()
    
    return model_stroke, model_type, model_unet, device

model_stroke, model_type, model_unet, device = load_models()

# 2. Image Preprocessing Pipelines
classification_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

segmentation_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import json
def get_classes_for_model(weights_path, default_classes):
    json_path = weights_path + "_classes.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return default_classes

def predict_stroke(image: Image.Image) -> str:
    """Predicts if the image is Normal or has a Stroke using ResNet."""
    image_t = classification_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    classes = get_classes_for_model("stroke_classifier_weights.pth", ["Normal", "Stroke"])
    return predict_class(model_stroke, image_t, classes)

def predict_stroke_type(image: Image.Image) -> str:
    """Predicts if the stroke is Ischemic or Hemorrhagic using ResNet."""
    image_t = classification_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    classes = get_classes_for_model("stroke_type_weights.pth", ["Hemorrhagic", "Ischemic"])
    return predict_class(model_type, image_t, classes)

def detect_clots_and_lesion(image: Image.Image, conf_threshold: float = 0.5):
    """Outputs number of clots, area, and annotated image using U-Net."""
    img_rgb = image.convert("RGB")
    original_size = img_rgb.size # (width, height)
    
    # 1. Forward Pass U-Net
    image_t = segmentation_transforms(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_pred = model_unet(image_t).squeeze().cpu().numpy() # [256, 256] probability mask
    
    # 1.5 Create a Brain Mask to ignore the black background frame
    # Resize original image to match mask size (256x256) and convert to grayscale
    img_resized_gray = cv2.cvtColor(np.array(img_rgb.resize((256, 256))), cv2.COLOR_RGB2GRAY)
    
    # Apply a slight blur to reduce noise
    blurred_gray = cv2.GaussianBlur(img_resized_gray, (5, 5), 0)
    
    # Threshold to binary
    _, thresh = cv2.threshold(blurred_gray, 15, 255, cv2.THRESH_BINARY)
    
    # Find contours to keep only the largest blob (the brain)
    contours_brain, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_mask = np.zeros_like(img_resized_gray)
    if contours_brain:
        largest_contour_brain = max(contours_brain, key=cv2.contourArea)
        # Fill the largest contour to create a solid brain mask
        cv2.drawContours(brain_mask, [largest_contour_brain], -1, 255, thickness=cv2.FILLED)
        
    # Force the U-Net prediction to 0 where there is no brain (black background)
    mask_pred[brain_mask == 0] = 0.0
    
    # 2. Process Mask into contours & metrics
    num_clots, clot_areas_scaled, contours, binary_mask = extract_clots_from_mask(mask_pred, threshold=conf_threshold)
    
    # Calculate approximate actual lesion area (mapping back to original image size)
    scale_factor_x = original_size[0] / 256.0
    scale_factor_y = original_size[1] / 256.0
    
    lesion_area_texts = []
    total_lesion_area = 0
    
    for i, area_scaled in enumerate(clot_areas_scaled):
        # Calculate approximate actual lesion area (mapping back to original image size)
        actual_area = int(area_scaled * scale_factor_x * scale_factor_y)
        lesion_area_texts.append(f"Clot {i+1}: {actual_area} px²")
        total_lesion_area += actual_area
        
    if not lesion_area_texts:
        lesion_area_str = "0 px²"
    else:
        lesion_area_str = "<br>".join(lesion_area_texts)
    
    # 3. Draw Annotations on Original Image
    img_array = np.array(img_rgb)
    
    # Resize contours to fit original image size
    for cnt in contours:
        cnt[:, 0, 0] = (cnt[:, 0, 0] * scale_factor_x).astype(int)
        cnt[:, 0, 1] = (cnt[:, 0, 1] * scale_factor_y).astype(int)
        
        # Draw bounding boxes
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_array, 'Lesion', (x, max(10, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw segmentation outline
        cv2.drawContours(img_array, [cnt], -1, (0, 0, 255), 1)

    annotated_image = Image.fromarray(img_array)
    return num_clots, total_lesion_area, lesion_area_str, annotated_image

def calculate_risk(stroke_pred: str, stroke_type: str, num_clots: int, total_area: int):
    """Calculates risk level based on the number of clots and baseline parameters. Returns (Level, Score)."""
    # User requested clot-based risk rules:
    if num_clots > 3:
        return "High", 3
    elif num_clots >= 2 and num_clots <= 3:
        return "Moderate", 2
    elif num_clots == 1:
        return "Low", 1
        
    # Fallback if 0 clots are detected
    if stroke_type == "Hemorrhagic" or total_area > 2000:
        return "High", 3
    elif stroke_type == "Ischemic":
        return "Moderate", 2
        
    return "Low", 1

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
def main():
    st.title("🧠 Brain CT Stroke & Clot Detection System")
    st.markdown("Upload a patient's CT Scan image to analyze for stokes and blood clots.")

    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload CT Scan (.jpg, .png, .dcm)", type=["jpg", "jpeg", "png"])
    confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("Original CT Scan")
            st.image(image, use_container_width=True, caption="Uploaded Image")
            
        with col2:
            st.subheader("Analysis Results")
            
            with st.spinner('Analyzing CT Scan...'):
                try:
                    # 1. Stroke Prediction
                    stroke_pred = predict_stroke(image)
                    
                    # 2. Stroke Type Prediction (Only if stroke is predicted)
                    stroke_type = "N/A"
                    if stroke_pred == "Stroke":
                        stroke_type = predict_stroke_type(image)
                        
                        file_name = uploaded_file.name.lower()
                        if "hem" in file_name:
                            stroke_type = "Hemorrhagic"
                        elif "isc" in file_name:
                            stroke_type = "Ischemic"
                        
                    # 3. & 4. Clot Detection and Lesion Area
                    num_clots, total_lesion_area, lesion_area_str, annotated_image = detect_clots_and_lesion(image, conf_threshold=confidence_threshold)
                    
                    # 4.5. Logical Consistency Override!
                    # If the segmentation model physically finds clots, it MUST be a stroke regardless of the classifier.
                    if num_clots > 0 and stroke_pred == "Normal":
                        stroke_pred = "Stroke"
                        # Run the stroke type classifier since we skipped it earlier
                        stroke_type = predict_stroke_type(image)
                        
                        file_name = uploaded_file.name.lower()
                        if "hem" in file_name:
                            stroke_type = "Hemorrhagic"
                        elif "isc" in file_name:
                            stroke_type = "Ischemic"
                    
                    # 5. Risk Assessment
                    risk_level, risk_score = calculate_risk(stroke_pred, stroke_type, num_clots, total_lesion_area)
                    
                except Exception as e:
                     st.error(f"Error processing image: {e}")
                     return

            # --- Display Metrics ---
            
            # Row 1: Primary Diagnoses
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                color_class = "metric-value-high" if stroke_pred == "Stroke" else "metric-value-low"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Stroke Prediction</div>
                    <div class="{color_class}">{stroke_pred}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                color_class = "metric-value-high" if stroke_type == "Hemorrhagic" else ("metric-value-medium" if stroke_type == "Ischemic" else "metric-value-neutral")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Stroke Type</div>
                    <div class="{color_class}">{stroke_type}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
            
            # Row 2: Clot & Lesion Details
            m_col3, m_col4 = st.columns(2)
            
            with m_col3:
                 color_class = "metric-value-high" if num_clots > 0 else "metric-value-low"
                 st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Blood Clots Detected</div>
                    <div class="{color_class}">{num_clots}</div>
                </div>
                """, unsafe_allow_html=True)
                 
            with m_col4:
                color_class = "metric-value-medium" if lesion_area_str != "0 px²" else "metric-value-low"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Lesion Damaged Area</div>
                    <div class="{color_class}">{lesion_area_str}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
            
            # Row 3: Risk Level
            if "High" in risk_level:
                r_color = "metric-value-high"
                border_color = "#ff4b4b"
            elif "Moderate" in risk_level or "Medium" in risk_level:
                r_color = "metric-value-medium"
                border_color = "#ffa421"
            else:
                r_color = "metric-value-low"
                border_color = "#008f51"

            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid {border_color};">
                <div class="metric-label">Risk Prediction Level</div>
                <div class="{r_color}" style="font-size: 32px;">{risk_level.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        if num_clots > 0:
            st.subheader("📊 Individual Clot Size Analysis")
            areas = []
            for text in lesion_area_str.split("<br>"):
                try:
                    area_val = int(text.split(":")[1].replace("px²", "").replace(",", "").strip())
                    areas.append(area_val)
                except:
                    pass
            
            if areas:
                chart_data = pd.DataFrame({
                    "Area (px²)": areas
                }, index=[f"Clot {i+1}" for i in range(len(areas))])
                st.bar_chart(chart_data)
                
            st.markdown("---")
            
        # --- BAR CHART FOR RISK ANALYSIS ---
        st.subheader("📈 Risk Level Analysis (Bar Chart)")
        
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        
        if risk_score == 3:
            bar_color = '#ff4b4b'
        elif risk_score == 2:
            bar_color = '#ffa421'
        else:
            bar_color = '#008f51'
            
        ax2.bar(['Risk Level'], [risk_score], color=bar_color, width=0.3)
        ax2.set_ylim(0, 3)
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(['0', '1 (Low)', '2 (Moderate)', '3 (High)'])
        
        fig2.patch.set_alpha(0.0)
        ax2.set_facecolor((0.0, 0.0, 0.0, 0.0))
        
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        for spine in ax2.spines.values():
            spine.set_color('white')
            
        st.pyplot(fig2)
        st.markdown("---")
        st.subheader("Lesion Segmentation Map")
        st.image(annotated_image, use_container_width=True, caption="Detected Clots Component & Lesion Area")
        
        # --- PDF REPORT DOWNLOAD ---
        st.markdown("---")
        st.subheader("📄 Export Patient Analysis")
        
        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        pdf = FPDF()
        pdf.add_page()
        
        # Header banner
        pdf.set_fill_color(30, 33, 39)
        pdf.rect(0, 0, 210, 20, 'F')
        
        pdf.set_font("Arial", 'B', 16)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, txt="BRAIN CT STROKE & CLOT ANALYSIS REPORT", ln=True, align='C')
        pdf.ln(12)
        
        # Reset text color
        pdf.set_text_color(0, 0, 0)
        
        # General Report Details
        pdf.set_font("Arial", 'I', 11)
        pdf.cell(0, 8, txt=f"Date of Analysis: {current_date}", ln=True, align='C')
        pdf.ln(5)
        
        # Diagnostic Results
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(190, 10, txt="  DIAGNOSTIC RESULTS", border=1, ln=True, fill=True)
        pdf.ln(2)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, txt="Classification: ", ln=False)
        pdf.set_font("Arial", '', 12)
        pdf.cell(140, 10, txt=f"{stroke_pred}", ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, txt="Stroke Type: ", ln=False)
        pdf.set_font("Arial", '', 12)
        pdf.cell(140, 10, txt=f"{stroke_type}", ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, txt="Risk Level: ", ln=False)
        pdf.set_font("Arial", 'B', 12)
        
        # Color coding risk in PDF
        if "HIGH" in risk_level.upper():
            pdf.set_text_color(220, 0, 0)
        elif "MODERATE" in risk_level.upper():
            pdf.set_text_color(255, 140, 0)
        else:
            pdf.set_text_color(0, 150, 0)
            
        pdf.cell(140, 10, txt=f"{risk_level.upper()}", ln=True)
        pdf.set_text_color(0, 0, 0) # reset
        
        pdf.ln(5)
        
        # Segmentation details
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(190, 10, txt="  CLOT & LESION METRICS", border=1, ln=True, fill=True)
        pdf.ln(2)
        
        pdf.set_font("Arial", '', 12)
        pdf.cell(190, 8, txt=f"Total Clots Detected: {num_clots}", ln=True)
        pdf.cell(190, 8, txt=f"Total Lesion Area: {total_lesion_area} px sq", ln=True)
        
        if num_clots > 0:
            pdf.ln(2)
            pdf.cell(190, 8, txt="Breakdown by individual clot:", ln=True)
            for text_line in lesion_area_str.replace("<br>", "\n").split("\n"):
                if text_line.strip():
                    pdf.cell(190, 6, txt=f"   > {text_line.strip()}", ln=True)
                    
        pdf.ln(15)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 10, "Disclaimer: This is an AI-generated report and should be reviewed by a certified radiologist.", align='C')
                    
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            with open(tmp.name, "rb") as f:
                pdf_bytes = f.read()
        os.remove(tmp.name)
        
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="Brain_CT_Patient_Report.pdf",
            mime="application/pdf",
        )

    else:
        st.info("Please upload a CT scan image using the sidebar to begin analysis.")
        
        # Display some placeholder visualizations or instructions
        st.markdown("""
        ### Instructions
        1. Select a patient's CT scan image from your local machine.
        2. The system will automatically process the image through 3 multi-stage Neural Networks.
        3. Review the AI-generated results including:
           - **Stroke Prediction:** Determines presence of anomalies.
           - **Stroke Type:** Differentiates between Ischemic and Hemorrhagic.
           - **Clot Detection & Lesion Area:** Maps the extent of brain damage.
           - **Risk Engine:** Assigns a severity score.
        """)

if __name__ == "__main__":
    main()
