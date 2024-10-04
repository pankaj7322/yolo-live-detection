# Yolo Live Detection
### Overview
This project implementes real-time object detection using the YOLOv8 model with a webcam feed. Built using Python and Streamlit,it demostrates how to perform live object detection and visualize the results in a web interface.

### Features
- Real-time object detection using YOLOv8
- confidence threshold control for object detection
- Bounding boxes and labels over detected objects
- Option to upload video files for processing
### Installation
#### Prerequisites
    Ensure you have Python 3.7 or higher installed. You will also need the following Python Pakcages:
    `streamlit`
    `opencv-python`
    `ultralytics`
    `pillow`
#### Setup
    1. Clone the Repository:
        ```
            https://github.com/pankaj7322/yolo-live-detection.git
            cd yolo-live-detection
        ```
            
    2. Install Dependencies:
        ```
            pip install -r requirements.txt
        ```
### Usage
1. Start the Streamlit App
     ```
        streamlit run yolo_live_video.py
     ```
2. Interact with the Web Interface
    - Use the sidebar to set the confidence threshold and control the webcam feed.
    - Upload video files if webcam access is not available or preferred
### contact
For questions or support, please contact [pankajkumar732298@gmail.com](mailto:pankajkumar732298@gmail.com).
