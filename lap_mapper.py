import cv2
import numpy as np
import csv
import time

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

def main():
    cap = cv2.VideoCapture(0)
    pid = PIDController(kp=0.5, ki=0.01, kd=0.1)

    print("Starting Mapping Lap. Press 'q' to stop early.")

    # Open CSV for writing
    with open("lap_map.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "frame", "lane_center", "steering_angle"])
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            height, width = frame.shape[:2]
            
            # LANE DETECTION (Standard Canny is generally good enough for mapping lap without harsh conditions)
            # You can also integrate adaptive thresholding here if required.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            polygons = np.array([[(0, height), (width, height), (width // 2, height // 2)]])
            cropped_edges = region_of_interest(edges, polygons)

            lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
            
            left_x, right_x = [], []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x1 < width // 2 and x2 < width // 2: left_x.extend([x1, x2])
                    if x1 > width // 2 and x2 > width // 2: right_x.extend([x1, x2])

            left_center = int(np.mean(left_x)) if left_x else 0
            right_center = int(np.mean(right_x)) if right_x else width
            lane_center = (left_center + right_center) // 2
            car_center = width // 2
            
            error = lane_center - car_center
            steering_angle = pid.compute(error)

            # Log into CSV
            current_time = time.time() - start_time
            writer.writerow([round(current_time, 3), frame_count, lane_center, round(steering_angle, 2)])
            frame_count += 1
            
            # Display
            cv2.circle(frame, (lane_center, height // 2), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"Mapping Lap: Frame {frame_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Mapper", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("Mapping Lap complete. Data saved to lap_map.csv")

if __name__ == "__main__":
    main()
