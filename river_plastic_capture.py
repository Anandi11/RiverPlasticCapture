import cv2
import torch
import pygame
import time

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

# Function to detect plastic in a frame
def detect_plastic(frame):
    results = model(frame)
    detections = results.xyxy[0].numpy()  # [x1, y1, x2, y2, confidence, class]
    plastic_detections = []
    print("All detections:", [(model.names[int(det[5])], det[4]) for det in detections])
    
    for det in detections:
        class_name = model.names[int(det[5])]
        confidence = det[4]
        if confidence > 0.3 and class_name in ['bottle', 'cup']:
            x_center = (det[0] + det[2]) / 2
            y_center = (det[1] + det[3]) / 2
            plastic_detections.append({
                'type': class_name,
                'confidence': confidence,
                'center': (x_center, y_center)
            })
            x1, y1, x2, y2 = map(int, det[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif confidence > 0.5 and class_name != 'bird':  # Fallback detection
            x_center = (det[0] + det[2]) / 2
            y_center = (det[1] + det[3]) / 2
            plastic_detections.append({
                'type': class_name,
                'confidence': confidence,
                'center': (x_center, y_center)
            })
            x1, y1, x2, y2 = map(int, det[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Fallback: {class_name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return plastic_detections, frame

# Function to update drone position
def update_drone_position(target_pos):
    global DRONE_POS
    if target_pos:
        dx = target_pos[0] - DRONE_POS[0]
        dy = target_pos[1] - DRONE_POS[1]
        distance = (dx**2 + dy**2) ** 0.5
        if distance > 5:  # Stop if close to target
            DRONE_POS[0] += 2 * (dx / distance) * (1 + 0.5 * abs(dy / distance)) * abs(0.5 * abs(dx / distance))
            DRONE_POS[1] += 2 * (dy / distance) * (1 + 0.5 * abs(dx / distance)) * abs(0.5 * abs(dy / distance))
            return True
    return False

# Main function
def main():
    global TARGET_POS
    frame = cv2.imread('test_image1.jpeg')
    if frame is None:
        print("Error: Could not load image.")
        return

    running = True
    while running:
        # Detect plastic in the frame
        detections, annotated_frame = detect_plastic(frame)
        if detections:
            TARGET_POS = [detections[0]['center'][0], detections[0]['center'][1]]
            print("Target set to:", TARGET_POS)

        # Update drone position
        update_drone_position(TARGET_POS)

        # Render simulation
        screen.fill((0, 0, 255))  # Blue background
        if TARGET_POS:
            pygame.draw.circle(screen, (255, 0, 0), (int(TARGET_POS[0] % SCREEN_WIDTH), int(TARGET_POS[1] % SCREEN_HEIGHT)), 10)
        pygame.draw.circle(screen, (0, 255, 0), (int(DRONE_POS[0]), int(DRONE_POS[1])), 15)
        pygame.display.flip()
        clock.tick(30)

        # Show video feed
        cv2.imshow('Plastic Detection', annotated_frame)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
