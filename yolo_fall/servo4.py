import cv2
from ultralytics import YOLO
from adafruit_servokit import ServoKit
import time

# Inisialisasi model YOLOv8
model = YOLO("best.pt")  # Gantilah dengan model YOLOv8 Anda

# Inisialisasi ServoKit
kit = ServoKit(channels=16)
servo_channel = 0  # Sesuaikan dengan channel servo yang Anda gunakan
servo_angle = 0  # Sudut awal servo
servo_speed = 10  # Kecepatan pergerakan servo, lebih cepat dari sebelumnya

# Fungsi untuk mencoba beberapa indeks kamera
def initialize_camera():
    for i in range(5):  # Mencoba 5 indeks kamera yang berbeda
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Kamera terbuka pada indeks {i}")
            return cap
    raise ValueError("Tidak dapat membuka kamera pada indeks mana pun")

# Inisialisasi kamera
cap = initialize_camera()

def move_servo_to_find_object():
    global servo_angle
    while True:
        servo_angle += servo_speed
        if servo_angle >= 360:
            servo_angle = 0
        kit.servo[servo_channel].angle = servo_angle % 180  # Servo hanya mendukung rotasi hingga 180 derajat
        time.sleep(0.05)  # Mengurangi waktu jeda untuk mempercepat pencarian
        ret, frame = cap.read()
        if not ret:
            continue
        results = model(frame)
        if len(results) > 0 and len(results[0].boxes) > 0:
            return frame, results
    return None, None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)
    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = box.cls[0]
            if conf > 0.5:  # Threshold confidence
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    else:
        frame, results = move_servo_to_find_object()
        if frame is not None and len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                if conf > 0.5:  # Threshold confidence
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
