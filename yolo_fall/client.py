import os
import cv2
import cvzone
import math
import requests
from datetime import datetime
from ultralytics import YOLO
import json

# Initialize video capture and writer
cap = cv2.VideoCapture('code/code/fall.mp4')

# Get the width and height of the video frames
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_fall_detection.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Load YOLO model
model = YOLO('yolov8s-pose.pt')

# Load class names
classnames = []
with open('code/code/classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Server URL for fall detection
url = "http://127.0.0.1:5000/fall-detected"

# File path for storing fall history
HISTORY_FILE = 'fall_history.json'

# Function to load fall history from a JSON file
def load_fall_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

# Function to save fall history to a JSON file
def save_fall_history(fall_history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(fall_history, f)

# Load existing fall history
fall_history = load_fall_history()

# Function to send POST request with image to the server
def send_fall_signal(cropped_image):
    # Set the time for fall detection
    fall_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert image to memory file for sending
    _, image_encoded = cv2.imencode('.jpg', cropped_image)
    files = {'image': ('fall_image.jpg', image_encoded.tobytes(), 'image/jpeg')}
    data = {
        "fall_detected": True,
        "fall_time": fall_time
    }
    
    try:
        # Send POST request to the server
        response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            print(f"Fall detected! Data and image sent: {response.json()}")
        else:
            print(f"Failed to send fall detection. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (980, 740))

    # Run the frame through the model
    results = model(frame)
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

            # Calculate dimensions for fall detection logic
            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

            # If a fall is detected based on threshold logic
            if threshold < 0:
                # Display "Fall Detected" on the frame
                cvzone.putTextRect(frame, 'Fall Detected', [height, width], thickness=2, scale=2)
                
                # Crop the image to the detected bounding box
                cropped_image = frame[y1:y2, x1:x2]
                
                # Save fall detection information in fall history
                fall_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fall_entry = {
                    "fall_detected": True,
                    "fall_time": fall_time,
                    "image_filename": f"{fall_time.replace(':', '-').replace(' ', '_')}.jpg"
                }

                # Save the cropped image to disk
                image_path = os.path.join('uploaded_images', fall_entry["image_filename"])
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                cv2.imwrite(image_path, cropped_image)

                # Append the fall entry to fall history
                fall_history.append(fall_entry)

                # Save the updated fall history to the JSON file
                save_fall_history(fall_history)

                # Send fall detection signal and image to the server
                send_fall_signal(cropped_image)

    # Write the frame into the output video file
    out.write(frame)

    # Show the frame in a window
    cv2.imshow('frame', frame)
    
    # Break the loop if 't' is pressed
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
