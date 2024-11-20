# import time
# import cv2
# import numpy as np
# import math
# import os
# import json
# from ultralytics import YOLO
# import requests  # Untuk mengirim data ke server

# # Custom class labels
# custom_class_names = ['Empty Chair', 'playing Phone', 'Working', 'Standing', 'Sleeping']

# # Load YOLO model
# model = YOLO('code/code/best2.pt')

# # Initialize video capture
# cap = cv2.VideoCapture('code/code/f2.mp4')  # Ganti dengan video file Anda

# # Get the video width, height, and frames per second (fps)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# # Initialize video writer
# output_filename = 'output2.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# # Timer variables for Worker 1
# timer_started_1 = False
# start_time_1 = 0
# end_time_1 = 0
# total_time_1 = 0
# detected_activity_1 = None

# # Timer variables for Worker 2
# timer_started_2 = False
# start_time_2 = 0
# end_time_2 = 0
# total_time_2 = 0
# detected_activity_2 = None

# # Region point variables
# region1_top_left = (100, 60)
# region1_bottom_right = (400, 530)
# region2_top_left = (520, 60)
# region2_bottom_right = (850, 530)

# # Server URL
# server_url = "http://127.0.0.1:5000/notDj1"  # Ganti dengan URL endpoint server

# # Folder untuk menyimpan gambar
# UPLOAD_FOLDER = 'upload_images2'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # JSON file for history
# HISTORY_FILE = 'notDj_history.json'


# def load_history():
#     """Membaca riwayat dari file JSON."""
#     if os.path.exists(HISTORY_FILE):
#         with open(HISTORY_FILE, 'r') as f:
#             return json.load(f)
#     return []


# def save_history(history):
#     """Menyimpan riwayat ke file JSON."""
#     with open(HISTORY_FILE, 'w') as f:
#         json.dump(history, f, indent=4)


# def save_and_send_to_server(worker_id, activity, duration, frame, bbox):
#     """Menyimpan gambar ke folder lokal, menyimpan riwayat ke JSON, dan mengirim data ke server."""
#     # Crop frame berdasarkan bounding box
#     x1, y1, x2, y2 = bbox
#     cropped_frame = frame[y1:y2, x1:x2]

#     # Simpan gambar di folder lokal
#     timestamp = time.strftime('%Y%m%d_%H%M%S')
#     filename = f"worker{worker_id}_{activity}_{timestamp}.jpg"
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     cv2.imwrite(filepath, cropped_frame)
#     print(f"Cropped image saved: {filepath}")

#     # Menambahkan data ke riwayat JSON
#     history = load_history()
#     history_entry = {
#         'worker_id': worker_id,
#         'activity': activity,
#         'duration': round(duration, 2),
#         'timestamp': timestamp,
#         'image_path': filepath
#     }
#     history.append(history_entry)
#     save_history(history)
#     print(f"History updated: {history_entry}")

#     # Kirim gambar dan data ke server
#     try:
#         with open(filepath, 'rb') as f:
#             files = {'image': (filename, f, 'image/jpeg')}
#             data = {
#                 'worker_id': worker_id,
#                 'activity': activity,
#                 'duration': duration,
#             }
#             response = requests.post(server_url, data=data, files=files)
#             print(f"Data sent to server: {response.status_code}, {response.json()}")
#     except Exception as e:
#         print(f"Failed to send data to server: {e}")


# # Process video frames
# while cap.isOpened():
#     success, img = cap.read()
#     if not success:
#         break

#     results = model(img, stream=True)
#     current_timer_started_1 = False
#     current_timer_started_2 = False

#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100

#             # Class Name
#             cls = int(box.cls[0])
#             currentClass = custom_class_names[cls]

#             # Check if the bounding box is within region 1
#             if (region1_top_left[0] < x1 < region1_bottom_right[0] and region1_top_left[1] < y1 < region1_bottom_right[1]):
#                 if currentClass in ['playing Phone', 'Sleeping'] and conf > 0.3:
#                     if not timer_started_1:
#                         start_time_1 = time.time()
#                         timer_started_1 = True
#                         detected_activity_1 = currentClass
#                         print(f"Started timer for {currentClass} in region 1")

#             # Check if the bounding box is within region 2
#             if (region2_top_left[0] < x1 < region2_bottom_right[0] and region2_top_left[1] < y1 < region2_bottom_right[1]):
#                 if currentClass in ['playing Phone', 'Sleeping'] and conf > 0.3:
#                     if not timer_started_2:
#                         start_time_2 = time.time()
#                         timer_started_2 = True
#                         detected_activity_2 = currentClass
#                         print(f"Started timer for {currentClass} in region 2")

#     # If no relevant class was detected in region 1 in this frame, stop the timer
#     if timer_started_1 and not current_timer_started_1:
#         end_time_1 = time.time()
#         elapsed_time_1 = end_time_1 - start_time_1
#         total_time_1 += elapsed_time_1
#         timer_started_1 = False
#         print(f"Timer stopped for Worker 1: {detected_activity_1} - Duration: {elapsed_time_1:.2f} seconds")
#         save_and_send_to_server(1, detected_activity_1, elapsed_time_1, img, (x1, y1, x2, y2))
#         detected_activity_1 = None

#     # If no relevant class was detected in region 2 in this frame, stop the timer
#     if timer_started_2 and not current_timer_started_2:
#         end_time_2 = time.time()
#         elapsed_time_2 = end_time_2 - start_time_2
#         total_time_2 += elapsed_time_2
#         timer_started_2 = False
#         print(f"Timer stopped for Worker 2: {detected_activity_2} - Duration: {elapsed_time_2:.2f} seconds")
#         save_and_send_to_server(2, detected_activity_2, elapsed_time_2, img, (x1, y1, x2, y2))
#         detected_activity_2 = None

#     # Write the frame to the output video
#     out.write(img)

#     # Display the frame
#     cv2.imshow('YOLO Detection', img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print(f"Total time tracked for Worker 1: {total_time_1:.2f} seconds")
# print(f"Total time tracked for Worker 2: {total_time_2:.2f} seconds")
import json
import os
import time
import cv2
import numpy as np
import math
from ultralytics import YOLO
import requests  # Untuk mengirim data ke server
import base64    # Untuk mengonversi foto ke format yang dapat dikirim

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
saved_frame_1 = None  # Untuk menyimpan frame terakhir

# Timer variables for Worker 2
timer_started_2 = False
start_time_2 = 0
total_time_2 = 0
detected_activity_2 = None
saved_frame_2 = None  # Untuk menyimpan frame terakhir

# Region point variables
region1_top_left = (100, 60)
region1_bottom_right = (400, 530)
region2_top_left = (520, 60)
region2_bottom_right = (850, 530)

# Server URL
server_url = "http://127.0.0.1:5000/notDj1"  # Ganti dengan URL endpoint server

# Folder untuk menyimpan gambar
UPLOAD_FOLDER = 'upload_images2'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# JSON file for history
HISTORY_FILE = 'notDj_history.json'


def load_history():
    """Membaca riwayat dari file JSON."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []


def save_history(history):
    """Menyimpan riwayat ke file JSON."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)


def save_and_send_to_server(worker_id, activity, duration, frame, bbox):
    """Menyimpan gambar ke folder lokal, menyimpan riwayat ke JSON, dan mengirim data ke server."""
    # Crop frame berdasarkan bounding box
    x1, y1, x2, y2 = bbox
    cropped_frame = frame[y1:y2, x1:x2]

    # Simpan gambar di folder lokal
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"worker{worker_id}_{activity}_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(filepath, cropped_frame)
    print(f"Cropped image saved: {filepath}")

    # Menambahkan data ke riwayat JSON
    history = load_history()
    history_entry = {
        'worker_id': worker_id,
        'activity': activity,
        'duration': round(duration, 2),
        'timestamp': timestamp,
        'image_filename': filename
    }
    history.append(history_entry)
    save_history(history)
    print(f"History updated: {history_entry}")

    # Kirim gambar dan data ke server
    try:
        with open(filepath, 'rb') as f:
            files = {'image': (filename, f, 'image/jpeg')}
            data = {
                'worker_id': worker_id,
                'activity': activity,
                'duration': duration,
            }
            response = requests.post(server_url, data=data, files=files)
            print(f"Data sent to server: {response.status_code}, {response.json()}")
    except Exception as e:
        print(f"Failed to send data to server: {e}")

# Draw region points
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
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, f"{currentClass} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if (region1_top_left[0] < x1 < region1_bottom_right[0] and region1_top_left[1] < y1 < region1_bottom_right[1]) or \
               (region1_top_left[0] < x2 < region1_bottom_right[0] and region1_top_left[1] < y2 < region1_bottom_right[1]):
                if currentClass in ['playing Phone', 'Sleeping'] and conf > 0.3:
                    if not timer_started_1:
                        start_time_1 = time.time()
                        timer_started_1 = True
                        detected_activity_1 = currentClass
                        print(f"Started timer for {currentClass} in region 1")
                    current_timer_started_1 = True
                    saved_frame_1 = img.copy()  # Simpan frame saat aktivitas terdeteksi

            # Check if the bounding box is within region 2
            if (region2_top_left[0] < x1 < region2_bottom_right[0] and region2_top_left[1] < y1 < region2_bottom_right[1]) or \
               (region2_top_left[0] < x2 < region2_bottom_right[0] and region2_top_left[1] < y2 < region2_bottom_right[1]):
                if currentClass in ['playing Phone', 'Sleeping'] and conf > 0.3:
                    if not timer_started_2:
                        start_time_2 = time.time()
                        timer_started_2 = True
                        detected_activity_2 = currentClass
                        print(f"Started timer for {currentClass} in region 2")
                    current_timer_started_2 = True
                    saved_frame_2 = img.copy()  # Simpan frame saat aktivitas terdeteksi

    # Stop timer and send data for Worker 1
    if timer_started_1 and not current_timer_started_1:
        end_time_1 = time.time()
        elapsed_time_1 = end_time_1 - start_time_1
        total_time_1 += elapsed_time_1
        timer_started_1 = False
        print(f"Stopped timer for {detected_activity_1} in region 1: {elapsed_time_1:.2f}s")
        if saved_frame_1 is not None:
            save_and_send_to_server(1, detected_activity_1, elapsed_time_1,img,(x1, y1, x2, y2))
        detected_activity_1 = None

    # Stop timer and send data for Worker 2
    if timer_started_2 and not current_timer_started_2:
        end_time_2 = time.time()
        elapsed_time_2 = end_time_2 - start_time_2
        total_time_2 += elapsed_time_2
        timer_started_2 = False
        print(f"Stopped timer for {detected_activity_2} in region 2: {elapsed_time_2:.2f}s")
        if saved_frame_2 is not None:
            save_and_send_to_server(2, detected_activity_2, elapsed_time_2, img ,(x1, y1, x2, y2))
        detected_activity_2 = None

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
