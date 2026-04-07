import cv2
import numpy as np
from ultralytics import YOLO
import pygame

# Toggle this for harsh lighting/shadow testing
USE_ADAPTIVE_THRESH = True


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

# --- NEW: AUDIO --- Initialize the audio mixer and load the horn
pygame.mixer.init()
try:
    horn_sound = pygame.mixer.Sound('horn.wav.mp3')
    print("Horn sound loaded successfully!")
except:
    print("WARNING: 'horn.wav' not found in your folder. No sound will play.")
    horn_sound = None

cap = cv2.VideoCapture(0)
pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
model = YOLO('yolov8n.pt') 

ret, frame = cap.read()
height, width = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('round1_demo.avi', fourcc, 20.0, (width, height))

print("Recording started... Press 'q' to save and exit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. OBSTACLE DETECTION (YOLO)
    results = model(frame, verbose=False) 
    combined_image = results[0].plot() 

    danger = False
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > 50000: 
            danger = True
            cv2.putText(combined_image, "EMERGENCY BRAKE!", (50, height // 2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
            
            # --- NEW: AUDIO --- Play the horn if it isn't already playing
            if horn_sound and not pygame.mixer.get_busy():
                horn_sound.play()

    # 2. LANE DETECTION
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if USE_ADAPTIVE_THRESH:
        # Create a dynamically thresholded image resistant to harsh shadows
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(thresh, 50, 150)
    else:
        edges = cv2.Canny(blur, 50, 150)

    polygons = np.array([[(0, height), (width, height), (width // 2, height // 2)]])
    cropped_edges = region_of_interest(edges, polygons)

    lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
    
    left_x, right_x = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if x1 < width // 2 and x2 < width // 2: left_x.extend([x1, x2])
            if x1 > width // 2 and x2 > width // 2: right_x.extend([x1, x2])

    # 3. STEERING LOGIC (PID)
    left_center = int(np.mean(left_x)) if left_x else 0
    right_center = int(np.mean(right_x)) if right_x else width
    lane_center = (left_center + right_center) // 2
    car_center = width // 2
    
    error = lane_center - car_center
    steering_angle = pid.compute(error)

    if danger:
        action_text = "Action: BRAKE (Throttle: 0)"
    else:
        action_text = f"Action: DRIVE (Throttle: 100) | Steering: {int(steering_angle)}"

    cv2.putText(combined_image, action_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.circle(combined_image, (lane_center, height // 2), 10, (0, 0, 255) if danger else (0, 255, 0), -1)

    out.write(combined_image)

    cv2.imshow('Autonomous Brain (Lanes + Obstacles)', combined_image)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()