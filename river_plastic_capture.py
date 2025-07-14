import cv2
import torch
import pygame
import time
import csv
from datetime import datetime

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize Pygame for simulation
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Drone Plastic Capture Simulation")
clock = pygame.time.Clock()
DRONE_POS = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]  # Starting position
TARGET_POS = None

# Video dimensions (assuming 640x480 as default for YOLOv5 input)
IMG_WIDTH, IMG_HEIGHT = 640, 480

# Initialize CSV file
csv_file = open('plastic_detections.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Object_Type', 'Confidence', 'X_Center', 'Y_Center'])

# Function to detect plastic in a frame
def detect_plastic(frame):
    results = model(frame)
    detections = results.xyxy[0].numpy()  # [x1, y1, x2, y2, confidence, class]
    plastic_detections = []
    print("All detections:", [(model.names[int(det[5])], det[4]) for det in detections])  # Debug all detections
    for det in detections:
        if det[4] > 0.1:  # Confidence threshold
            x1, y1, x2, y2 = map(int, det[:4])
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            obj_class = model.names[int(det[5])]
            # Relabel 'boat' or 'person' to 'bottle' if area is small (typical for a bottle)
            area = (x2 - x1) * (y2 - y1)
            if obj_class in ['boat', 'person'] and area < 5000:  # Arbitrary small area threshold
                obj_class = 'bottle'
            plastic_detections.append({
                'type': obj_class,
                'confidence': det[4],
                'center': (x_center, y_center)
            })
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{obj_class} ({det[4]:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Log to CSV
            csv_writer.writerow([datetime.now(), obj_class, det[4], x_center, y_center])
    return plastic_detections, frame

# Function to select the nearest target
def select_nearest_target(detections):
    if not detections:
        return None
    drone_img_x = (DRONE_POS[0] / SCREEN_WIDTH) * IMG_WIDTH
    drone_img_y = (DRONE_POS[1] / SCREEN_HEIGHT) * IMG_HEIGHT
    min_distance = float('inf')
    nearest_target = None
    for det in detections:
        if det['type'] == 'bottle':  # Only target bottles
            x, y = det['center']
            distance = ((x - drone_img_x) ** 2 + (y - drone_img_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_target = det['center']
    if nearest_target:
        return map_coordinates(nearest_target[0], nearest_target[1])
    return None

# Function to update drone position
def update_drone_position(target_pos):
    global DRONE_POS
    if target_pos:
        dx = target_pos[0] - DRONE_POS[0]
        dy = target_pos[1] - DRONE_POS[1]
        distance = (dx**2 + dy**2) ** 0.5
        if distance > 5:
            DRONE_POS[0] += 2 * (dx / distance)
            DRONE_POS[1] += 2 * (dy / distance)
            return True
    return False

# Function to map image coordinates to Pygame coordinates
def map_coordinates(x, y):
    x_mapped = (x / IMG_WIDTH) * SCREEN_WIDTH
    y_mapped = (y / IMG_HEIGHT) * SCREEN_HEIGHT
    return int(x_mapped), int(y_mapped)

# Main function
def main():
    global TARGET_POS
    # Try video file first, fallback to webcam
    video_source = 'test_video2.mp4'  # Replace with your video path
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_source}'. Trying webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam. Ensure video file exists or webcam is connected.")
            return

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        detections, annotated_frame = detect_plastic(frame)

        # Update target position (select nearest bottle)
        if detections:
            TARGET_POS = select_nearest_target(detections)
            if TARGET_POS:
                print("Target set to:", TARGET_POS)

        moving = update_drone_position(TARGET_POS)

        screen.fill((0, 0, 255))  # Blue background (water)
        if TARGET_POS:
            pygame.draw.circle(screen, (255, 0, 0), TARGET_POS, 10)  # Red dot on detected bottle
        pygame.draw.circle(screen, (0, 255, 0), (int(DRONE_POS[0]), int(DRONE_POS[1])), 15)  # Green drone
        pygame.display.flip()
        clock.tick(30)  # 30 FPS

        cv2.imshow('Plastic Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    cap.release()
    csv_file.close()
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()