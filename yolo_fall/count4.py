from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open the video file
cap = cv2.VideoCapture('uash.mp4')
assert cap.isOpened(), "Error reading video file"

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output3.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Define the points of the rectangle (A and B)
region_points = np.array([(100, 100), (840, 100), (840, 460), (100, 460)], dtype=np.int32)

def is_inside_region(center, region_points):
    return cv2.pointPolygonTest(region_points, (float(center[0]), float(center[1])), False) >= 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Track all classes (persons and badges)
    results = model.track(im0, persist=True, show=False)

    person_inside_count = 0
    filtered_tracks = []

    if results:
        for result in results:
            for box in result.boxes:
                if len(box.cls) > 0 and box.cls[0] == 0:  # Check if it's a person and array is not empty
                    bbox = box.xyxy[0]
                    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    if is_inside_region(center, region_points):
                        filtered_tracks.append(box)
                        person_inside_count += 1

        # Draw bounding boxes on the image for filtered tracks
        for box in filtered_tracks:
            bbox = box.xyxy[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]}"
            cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(im0, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Draw the region on the frame
    cv2.polylines(im0, [region_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the count of people inside the region
    cv2.putText(im0, f"People inside region: {person_inside_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the frame to the output video
    out.write(im0)

    # Display the resulting frame
    cv2.imshow('Frame', im0)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
