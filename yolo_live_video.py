import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import math
from PIL import Image

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Streamlit app
st.title("YOLO Live Detection")
st.write("Use the sidebar to control the webcam feed and detection parameters.")

# Sidebar options
st.sidebar.title("Options")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
run_webcam = st.sidebar.checkbox("Run Webcam Feed", value=True)

# Session state for control
if "stop" not in st.session_state:
    st.session_state.stop = False

if run_webcam:
    st.write("Webcam is running...")
    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()

    # Start the webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Stop button
    if st.button('Stop'):
        st.session_state.stop = True

    # Display the webcam feed
    while not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        # Perform detection
        results = model(frame, stream=True)

        # Draw bounding boxes and labels
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                if confidence >= conf_threshold:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cls = int(box.cls[0])
                    label = f"{classNames[cls]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to PIL Image format
        image = Image.fromarray(frame_rgb)

        # Display the frame in Streamlit
        frame_placeholder.image(image, channels="RGB", use_column_width=True)

    cap.release()
    st.write("Webcam stopped.")
else:
    st.write("Check the checkbox to run the webcam feed.")
