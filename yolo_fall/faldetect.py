import cv2
import cvzone
import math
from ultralytics import YOLO

# Initialize video capture and writer
cap = cv2.VideoCapture('fall.mp4')

# Get the width and height of the video frames
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_fall_detection.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

model = YOLO('yolov8s-pose.pt')

classnames = []
with open('Fall-Detection-main/classes.txt', 'r') as f:
    classnames = f.read().splitlines()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (980, 740))

    results = model(frame)
    # Check if results are being obtained
    if not results:
        print("No results detected")
        continue

    for info in results:
        parameters = info.boxes
        if not parameters:
            print("No boxes detected")
            continue

        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            # Implement fall detection using the coordinates x1, y1, x2
            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

            if threshold < 0:
                cvzone.putTextRect(frame, 'Fall Detected', [height, width], thickness=2, scale=2)

    # Write the frame into the file 'output_fall_detection.avi'
    out.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
