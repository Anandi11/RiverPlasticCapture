import cv2
import torch
import pygame
import time
import csv
from datetime import datetime

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Drone Plastic Capture Simulation")
clock = pygame.time.Clock()
DRONE_POS = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2] 
TARGET_POS = None

IMG_WIDTH, IMG_HEIGHT = None, None 

csv_file = open('plastic_detections.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Object_Type', 'Confidence', 'X_Center', 'Y_Center'])

def detect_plastic(frame):
    results = model(frame)
    detections = results.xyxy[0].numpy() 
    plastic_detections = []
    print("All detections:", [(model.names[int(det[5])], det[4]) for det in detections])
    for det in detections:
        if det[4] > 0.05:
            x1, y1, x2, y2 = map(int, det[:4])
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
    
            obj_class = 'plastic_object'
            plastic_detections.append({
                'type': obj_class,
                'confidence': det[4],
                'center': (x_center, y_center)
            })
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{obj_class} ({det[4]:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            csv_writer.writerow([datetime.now(), obj_class, det[4], x_center, y_center])
    return plastic_detections, frame

def select_nearest_target(detections):
    if not detections:
        return None
    drone_img_x = (DRONE_POS[0] / SCREEN_WIDTH) * IMG_WIDTH if IMG_WIDTH else DRONE_POS[0]
    drone_img_y = (DRONE_POS[1] / SCREEN_HEIGHT) * IMG_HEIGHT if IMG_HEIGHT else DRONE_POS[1]
    min_distance = float('inf')
    nearest_target = None
    for det in detections:
        x, y = det['center']
        distance = ((x - drone_img_x) ** 2 + (y - drone_img_y) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest_target = det['center']
    if nearest_target:
        return map_coordinates(nearest_target[0], nearest_target[1])
    return None

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

def map_coordinates(x, y):
    x_mapped = (x / IMG_WIDTH) * SCREEN_WIDTH if IMG_WIDTH else (x / 640) * SCREEN_WIDTH
    y_mapped = (y / IMG_HEIGHT) * SCREEN_HEIGHT if IMG_HEIGHT else (y / 480) * SCREEN_HEIGHT
    return int(x_mapped), int(y_mapped)

def main():
    global TARGET_POS, IMG_WIDTH, IMG_HEIGHT
    video_source = 'test_video2.mp4'
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_source}'. Trying webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam. Ensure video file exists or webcam is connected.")
            return

    IMG_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    IMG_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {IMG_WIDTH}x{IMG_HEIGHT}")

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        detections, annotated_frame = detect_plastic(frame)
        annotated_frame = cv2.resize(annotated_frame, (640, 480))

        if detections:
            TARGET_POS = select_nearest_target(detections)
            if TARGET_POS:
                print("Target set to:", TARGET_POS)

        moving = update_drone_position(TARGET_POS)

        screen.fill((0, 0, 255))
        if TARGET_POS:
            pygame.draw.circle(screen, (255, 0, 0), TARGET_POS, 10) 
        pygame.draw.circle(screen, (0, 255, 0), (int(DRONE_POS[0]), int(DRONE_POS[1])), 15)
        pygame.display.flip()
        clock.tick(30)

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