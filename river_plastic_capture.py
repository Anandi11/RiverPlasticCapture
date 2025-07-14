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
        if det[4] > 0.3:  # Any detection above 0.3 confidence
            x_center = (det[0] + det[2]) / 2
            y_center = (det[1] + det[3]) / 2
            # Override label to "plastic_object" for this project
            plastic_detections.append({
                'type': 'plastic_object',
                'confidence': det[4],
                'center': (x_center, y_center)
            })
            x1, y1, x2, y2 = map(int, det[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"plastic_object ({det[4]:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
    # Load image
    frame = cv2.imread('test_image2.jpeg')
    if frame is None:
        print("Error: Could not load image.")
        return

    running = True
    while running:
        # Detect plastic in the frame (only once for image)
        detections, annotated_frame = detect_plastic(frame)
        if detections and not TARGET_POS:
            TARGET_POS = [detections[0]['center'][0], detections[0]['center'][1]]  # Set target once
            print("Target set to:", TARGET_POS)

        # Update drone position
        moving = update_drone_position(TARGET_POS)

        # Render simulation
        screen.fill((0, 0, 255))  # Blue background (water)
        if TARGET_POS:
            pygame.draw.circle(screen, (255, 0, 0), (int(TARGET_POS[0] % SCREEN_WIDTH), int(TARGET_POS[1] % SCREEN_HEIGHT)), 10)  # Plastic
        pygame.draw.circle(screen, (0, 255, 0), (int(DRONE_POS[0]), int(DRONE_POS[1])), 15)  # Drone
        pygame.display.flip()
        clock.tick(30)  # 30 FPS

        # Show detection result in separate CV2 window
        cv2.imshow('Plastic Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()