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
cap = cv2.VideoCapture('code/code/f2.mp4')  # Replace 'uas.mp4' with your video file

# Get the video width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer
output_filename = 'output2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Timer variables for Worker 1
timer_started_1 = False
start_time_1 = 0
total_time_1 = 0
detected_activity_1 = None

# Timer variables for Worker 2
timer_started_2 = False
start_time_2 = 0
total_time_2 = 0
detected_activity_2 = None

# Region point variables
region1_top_left = (100, 60)
region1_bottom_right = (400, 530)
region2_top_left = (520, 60)
region2_bottom_right = (850, 530)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    current_timer_started_1 = False
    region1_contains_empty_chair = False
    detected_activities_1 = []

    current_timer_started_2 = False
    region2_contains_empty_chair = False
    detected_activities_2 = []

    # Draw region points
    region_color = (0, 255, 0)  # Default region color is green
    cv2.rectangle(img, region1_top_left, region1_bottom_right, region_color, 2)
    cv2.rectangle(img, region2_top_left, region2_bottom_right, region_color, 2)

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

            # Check if the bounding box is within region 1
            if (region1_top_left[0] < x1 < region1_bottom_right[0] and region1_top_left[1] < y1 < region1_bottom_right[1]) or \
               (region1_top_left[0] < x2 < region1_bottom_right[0] and region1_top_left[1] < y2 < region1_bottom_right[1]):
                if currentClass in ['playing Phone', 'Sleeping'] and conf > 0.3:
                    if not timer_started_1:
                        start_time_1 = time.time()
                        timer_started_1 = True
                        detected_activity_1 = currentClass
                        print(f"Started timer for {currentClass} in region 1")
                    detected_activities_1.append(currentClass)
                    current_timer_started_1 = True
                elif currentClass == 'Empty Chair' and conf > 0.3:
                    region1_contains_empty_chair = True

            # Check if the bounding box is within region 2
            if (region2_top_left[0] < x1 < region2_bottom_right[0] and region2_top_left[1] < y1 < region2_bottom_right[1]) or \
               (region2_top_left[0] < x2 < region2_bottom_right[0] and region2_top_left[1] < y2 < region2_bottom_right[1]):
                if currentClass in ['playing Phone', 'Sleeping'] and conf > 0.3:
                    if not timer_started_2:
                        start_time_2 = time.time()
                        timer_started_2 = True
                        detected_activity_2 = currentClass
                        print(f"Started timer for {currentClass} in region 2")
                    detected_activities_2.append(currentClass)
                    current_timer_started_2 = True
                elif currentClass == 'Empty Chair' and conf > 0.3:
                    region2_contains_empty_chair = True

    # If no relevant class was detected in region 1 in this frame, stop the timer
    if timer_started_1 and not current_timer_started_1:
        end_time_1 = time.time()
        elapsed_time_1 = end_time_1 - start_time_1
        total_time_1 += elapsed_time_1
        timer_started_1 = False
        print(f"Stopped timer. Total time for {detected_activity_1} in region 1: {total_time_1:.2f} seconds")
        detected_activity_1 = None

    # If no relevant class was detected in region 2 in this frame, stop the timer
    if timer_started_2 and not current_timer_started_2:
        end_time_2 = time.time()
        elapsed_time_2 = end_time_2 - start_time_2
        total_time_2 += elapsed_time_2
        timer_started_2 = False
        print(f"Stopped timer. Total time for {detected_activity_2} in region 2: {total_time_2:.2f} seconds")
        detected_activity_2 = None

    # Display the timer if it's running for region 1
    if timer_started_1:
        elapsed_time_1 = time.time() - start_time_1
        y_offset = region1_top_left[1] - 40
        if 'playing Phone' in detected_activities_1:
            activity_text = f"Worker 1 is playing Phone: {elapsed_time_1:.2f}s"
            cv2.putText(img, activity_text, (region1_top_left[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 30
        if 'Sleeping' in detected_activities_1:
            activity_text = f"Worker 1 is Sleeping: {elapsed_time_1:.2f}s"
            cv2.putText(img, activity_text, (region1_top_left[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # If the region 1 contains only an empty chair, update the region text and color
    if region1_contains_empty_chair and not current_timer_started_1:
        cv2.putText(img, "Worker 1 is Empty", (region1_top_left[0], region1_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(img, region1_top_left, region1_bottom_right, (0, 0, 255), 2)
    else:
        cv2.putText(img, "Worker 1", (region1_top_left[0], region1_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, region_color, 1)

    # Display the timer if it's running for region 2
    if timer_started_2:
        elapsed_time_2 = time.time() - start_time_2
        y_offset = region2_top_left[1] - 40
        if 'playing Phone' in detected_activities_2:
            activity_text = f"Worker 2 is playing Phone: {elapsed_time_2:.2f}s"
            cv2.putText(img, activity_text, (region2_top_left[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 30
        if 'Sleeping' in detected_activities_2:
            activity_text = f"Worker 2 is Sleeping: {elapsed_time_2:.2f}s"
            cv2.putText(img, activity_text, (region2_top_left[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # If the region 2 contains only an empty chair, update the region text and color
    if region2_contains_empty_chair and not current_timer_started_2:
        cv2.putText(img, "Worker 2 is Empty", (region2_top_left[0], region2_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(img, region2_top_left, region2_bottom_right, (0, 0, 255), 2)
    else:
        cv2.putText(img, "Worker 2", (region2_top_left[0], region2_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, region_color, 1)

    # Write the frame to the output video
    out.write(img)

    # Display the frame
    cv2.imshow('YOLO Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total time tracked for Worker 1: {total_time_1:.2f} seconds")
print(f"Total time tracked for Worker 2: {total_time_2:.2f} seconds")
