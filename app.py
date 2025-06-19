import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.title("🎨 Advanced Video to Cartoon Converter")
st.markdown("""
Converts videos into **real cartoon-style animation** with adjustable effects!
""")

def advanced_cartoon(frame, color_levels=8, edge_thickness=9):
    # Color quantization
    data = np.float32(frame).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, palette = cv2.kmeans(data, color_levels, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    quantized = palette[labels.flatten()].reshape(frame.shape)
    quantized = np.uint8(quantized)

    # Edge detection
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, edge_thickness, 9)

    # Smoothing
    smoothed = cv2.bilateralFilter(quantized, d=9, sigmaColor=200, sigmaSpace=200)
    
    # Combine edges with colors
    cartoon = cv2.bitwise_and(smoothed, smoothed, mask=edges)
    return cartoon

# File uploader
uploaded_file = st.file_uploader("Upload video (MP4/AVI)", type=["mp4", "avi"])

if uploaded_file is not None:
    # Save uploaded file to temporary location
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Read video
    cap = cv2.VideoCapture(temp_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # User adjustments
    st.sidebar.header("Cartoon Settings")
    color_levels = st.sidebar.slider("Color Levels", 4, 32, 8)
    edge_thickness = st.sidebar.slider("Edge Thickness", 3, 15, 9, step=2)
    
    if st.button("Convert to Cartoon"):
        # Prepare output
        output_path = os.path.join(temp_dir, "cartoon_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            cartoon_frame = advanced_cartoon(frame, color_levels, edge_thickness)
            out.write(cartoon_frame)
            progress = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/total_frames*100)
            progress_bar.progress(progress)
        
        # Clean up
        cap.release()
        out.release()
        
        # Show sample frame
        st.image(cv2.cvtColor(cartoon_frame, cv2.COLOR_BGR2RGB), 
                caption="Cartoon Effect Preview", use_column_width=True)
        
        # Download button
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Cartoon Video",
                data=f,
                file_name="cartoon_video.mp4",
                mime="video/mp4"
            )
        
        # Clean temporary files
        try:
            os.remove(temp_path)
            os.remove(output_path)
            os.rmdir(temp_dir)
        except:
            pass