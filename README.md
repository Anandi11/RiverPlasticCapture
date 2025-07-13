# River Plastic Capture System
A tech-based solution for detecting and capturing plastic waste in rivers using AI and simulated drone navigation. Built for a hackathon, this project uses YOLOv5 for plastic detection and Pygame for navigation simulation, with plans for future drone hardware integration.

## Setup
1. Clone the repository: `git clone https://github.com/Anandi11/RiverPlasticCapture.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python river_plastic_capture.py

## Current Features
- Detects plastic objects (bottles, cups) in a video feed, webcam, or image using YOLOv5.
- Displays green bounding boxes for confirmed plastic (e.g., bottle) and blue for fallback objects.
- Simulates drone movement toward detected plastic using Pygame.
- Tested with image input, with debug logging for all detections.

## Future Work
- Integrate with drone SDKs (e.g., DJI, Ardupilot).
- Deploy on IoT devices for real-time river cleanup.