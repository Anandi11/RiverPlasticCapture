# River Plastic Capture System
A tech-based solution for detecting and capturing plastic waste in rivers using AI and simulated drone navigation. Built for a hackathon, this project uses YOLOv5 for plastic detection and Pygame for navigation simulation, with plans for future drone hardware integration.

## Setup
1. Clone the repository: `git clone https://github.com/Anandi11/RiverPlasticCapture.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python river_plastic_capture.py

## Current Features
- Detects objects in a video feed, webcam, or image using YOLOv5 and labels them as 'plastic_object'.
- Displays green bounding boxes with confidence scores in a separate CV2 window.
- Simulates drone movement toward detected plastic using Pygame, with a red dot marking the plastic and a green dot for the drone (refinement pending).
- Logs detection details (timestamp, type, confidence, coordinates) to plastic_detections.csv.
- Tested with image input, with debug logging for all detections.

## Future Work
- Integrate with drone SDKs (e.g., DJI, Ardupilot).
- Deploy on IoT devices for real-time river cleanup.