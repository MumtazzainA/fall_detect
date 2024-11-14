from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

assert cap.isOpened(), "Error reading video file"

# Define the points of the rectangle (A and B)
# Assuming point A as the top left corner and point B as the bottom right corner
region_points = [(100, 100), (540, 100), (540, 360), (100, 360)]
classes_to_count = [0]  # Person class for count

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

def is_close(detection1, detection2, threshold=50):
    # Calculate the Euclidean distance between the centers of two detections
    center1 = np.array([detection1[0] + detection1[2] / 2, detection1[1] + detection1[3] / 2])
    center2 = np.array([detection2[0] + detection2[2] / 2, detection2[1] + detection2[3] / 2])
    distance = np.linalg.norm(center1 - center2)
    return distance < threshold

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Track all classes (persons and badges)
    tracks = model.track(im0, persist=True, show=False)

    # Check if tracks are not empty
    if tracks:
        # Filter out person detections that are close to badge detections
        filtered_tracks = []
        for track in tracks:
            if track.boxes.cls[0] == 0:  # Check if it's a person
                exclude = False
                for other_track in tracks:
                    if other_track.boxes.cls[0] != 0:  # Check if it's not a person (i.e., a badge)
                        if is_close(track.boxes.xyxy[0], other_track.boxes.xyxy[0]):
                            exclude = True
                            break
                if not exclude:
                    filtered_tracks.append(track)

        # Update the image with filtered tracks
        if filtered_tracks:  # Ensure filtered_tracks is not empty
            im0 = counter.start_counting(im0, filtered_tracks)

    # Display the resulting frame
    cv2.imshow('Frame', im0)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
