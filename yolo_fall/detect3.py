import time
import cv2
import numpy as np
import math
from ultralytics import YOLO

# Custom class labels
custom_class_names = ['Empty Chair', 'playing Phone', 'Working', 'Standing', 'Sleeping']

# Load YOLO model
model = YOLO('code/code/best2.pt')

# Initialize video capture
cap = cv2.VideoCapture('code/code/f1.mp4')  # Replace 'uas.mp4' with your video file

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Timer variables
timer_started = False
start_time = 0
total_time = 0
detected_activity = None

# Region point variables
region_top_left = (0, 100)
region_bottom_right = (700, 820)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    current_timer_started = False
    region_contains_empty_chair = False
    detected_activities = []

    # Draw region point
    region_color = (0, 0, 255)  # Default region color is green
    cv2.rectangle(img, region_top_left, region_bottom_right, region_color, 2)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = custom_class_names[cls]

            # Draw bounding box and label for all classes
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, f"{currentClass} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Check if the bounding box is within the region point
            if (region_top_left[0] < x1 < region_bottom_right[0] and region_top_left[1] < y1 < region_bottom_right[1]) or \
               (region_top_left[0] < x2 < region_bottom_right[0] and region_top_left[1] < y2 < region_bottom_right[1]):
                if currentClass in ['playing Phone', 'Sleeping'] and conf > 0.3:
                    if not timer_started:
                        start_time = time.time()
                        timer_started = True
                        detected_activity = currentClass
                        print(f"Started timer for {currentClass} in region")
                    detected_activities.append(currentClass)
                    current_timer_started = True
                elif currentClass == 'Empty Chair' and conf > 0.3:
                    region_contains_empty_chair = True

    # If no relevant class was detected in this frame, stop the timer
    if timer_started and not current_timer_started:
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        timer_started = False
        print(f"Stopped timer. Total time for {detected_activity}: {total_time:.2f} seconds")
        detected_activity = None

    # Display the timer if it's running
    if timer_started:
        elapsed_time = time.time() - start_time
        y_offset = 30
        for activity in detected_activities:
            activity_text = f"Worker 1 is {activity}: {elapsed_time:.2f}s"
            cv2.putText(img, activity_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 30

    # If the region contains only an empty chair, update the region text and color
    if region_contains_empty_chair and not current_timer_started:
        cv2.putText(img, "Worker 1 is Empty", (region_top_left[0], region_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(img, region_top_left, region_bottom_right, (0, 0, 255), 2)
    else:
        cv2.putText(img, "Worker 1", (region_top_left[0], region_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, region_color, 2)

    # Write the frame into the file
    out.write(img)

    # Display the frame
    cv2.imshow('YOLO Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total time tracked: {total_time:.2f} seconds")
