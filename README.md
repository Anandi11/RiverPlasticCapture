# River Plastic Capture System
A tech-based solution for detecting and capturing plastic waste in rivers using AI and simulated drone navigation. Built for the DigiGreen Recycling Pvt. Ltd. hackathon, this project leverages YOLOv5 for object detection, Pygame for drone simulation, and CSV logging for data analysis.

## Features
- Detects objects in images (e.g., plastic bottles) using YOLOv5 and labels them as 'plastic_object'.
- Displays green bounding boxes with confidence scores in a separate CV2 window.
- Simulates drone movement toward detected plastic using Pygame, with a green dot for the drone (red dot alignment pending refinement).
- Logs detection details (timestamp, type, confidence, coordinates) to plastic_detections.csv for analysis.
- Tested with a sample image input, with debug logging for all detections.

## Setup
1. Clone the repository: git clone https://github.com/Anandi11/RiverPlasticCapture.git
2. Install dependencies: pip install -r requirements.txt
3. Place a test image (e.g., test_image.jpg) in the project folder.
4. Run the script: python river_plastic_capture.py
- Requires a virtual environment (optional): python -m venv venv and activate with venv\Scripts\activate (Windows) or source venv/bin/activate (Mac/Linux).

## Demo
Watch the [demo video](test_video.mp4) (local file, not uploaded) to see the system in action:
- Drone simulation moving toward detected plastic.
- CV2 window showing bounding box detection.
- CSV logging of detection data.

## Future Work
- Refine Pygame simulation to accurately align the red dot with the detected plastic object.
- Integrate with drone hardware (e.g., DJI SDK) for real-world testing.
- Fine-tune YOLOv5 with a custom dataset for better plastic detection in water.
- Expand to video/webcam input for dynamic river monitoring.

## Acknowledgments
Built with guidance from xAI’s Grok 3 for the DigiGreen Recycling Pvt. Ltd. hackathon.