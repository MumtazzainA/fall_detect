import os
import base64
import json
from flask import Flask, render_template, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Direktori untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Nama file untuk menyimpan riwayat deteksi
FALL_HISTORY_FILE = 'fall_history.json'
NOTDJ_HISTORY_FILE = 'notDj_history.json'

# Membaca riwayat deteksi dari file JSON jika ada
def load_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
            else:
                return []
    return []

# Menyimpan riwayat deteksi ke dalam file JSON
def save_history(history, file_path):
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4)

# Inisialisasi riwayat deteksi dari file
fall_history = load_history(FALL_HISTORY_FILE)
notDj_history = load_history(NOTDJ_HISTORY_FILE)


@app.route('/')
def home_page():
    notification_count = len(fall_history) + len(notDj_history)  # Menghitung total riwayat
    return render_template('homePage.html', notification_count=notification_count)


# Routes untuk Fall Detection
@app.route('/fall-detected')
def fall_detected_page():
    global fall_history
    history_with_images = []
    for entry in fall_history:
        entry_with_image = entry.copy()
        if entry["image_filename"]:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], entry["image_filename"])
            if os.path.exists(image_path):
                # Konversi gambar ke Base64 jika file ada
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    entry_with_image["image_base64"] = f"data:image/jpeg;base64,{encoded_string}"
            else:
                # Tambahkan placeholder jika file tidak ada
                entry_with_image["image_base64"] = "data:image/jpeg;base64,"  # Kosongkan
                entry_with_image["file_missing"] = True
        else:
            entry_with_image["image_base64"] = None
        history_with_images.append(entry_with_image)

    return render_template('index.html', fall_history=history_with_images)


@app.route('/fall-detected2', methods=['POST'])
def fall_detected_route():
    global fall_history

    # Menerima data JSON dan file gambar (jika ada)
    fall_detected = request.form.get('fall_detected', 'false').lower() == 'true'
    fall_time = request.form.get('fall_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Cek apakah ada file gambar yang diunggah
    image = request.files.get('image')
    image_filename = None
    if image:
        # Simpan file gambar
        image_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(image_path)

    # Simpan data deteksi ke dalam riwayat
    fall_entry = {
        "fall_detected": fall_detected,
        "fall_time": fall_time,
        "image_filename": image_filename
    }
    fall_history.append(fall_entry)

    # Simpan riwayat deteksi ke file JSON
    save_history(fall_history, FALL_HISTORY_FILE)

    print(f"Fall detected! Data received: {fall_entry}")

    return "Fall detection data received.", 200


# Routes untuk NotDJ Detection
@app.route('/notDj')
def not_dj_page():
    global notDj_history
    history_with_images = []
    for entry in notDj_history:
        entry_with_image = entry.copy()
        if entry["image_filename"]:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], entry["image_filename"])
            if os.path.exists(image_path):
                # Konversi gambar ke Base64 jika file ada
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    entry_with_image["image_base64"] = f"data:image/jpeg;base64,{encoded_string}"
            else:
                # Tambahkan placeholder jika file tidak ada
                entry_with_image["image_base64"] = "data:image/jpeg;base64,"  # Kosongkan
                entry_with_image["file_missing"] = True
        else:
            entry_with_image["image_base64"] = None
        history_with_images.append(entry_with_image)

    return render_template('notDj.html', history=history_with_images)


@app.route('/notDj1', methods=['POST'])
def not_dj_data():
    global notDj_history

    # Menerima data JSON dan file gambar (jika ada)
    worker_id = request.form.get('worker_id')
    activity = request.form.get('activity')
    duration = request.form.get('duration')
    detection_time = request.form.get('detection_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Cek apakah ada file gambar yang diunggah
    image = request.files.get('image')
    image_filename = None
    if image:
        # Simpan file gambar
        image_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(image_path)

    # Simpan data deteksi ke dalam riwayat
    entry = {
        "worker_id": worker_id,
        "activity": activity,
        "duration": duration,
        "detection_time": detection_time,
        "image_filename": image_filename
    }
    notDj_history.append(entry)

    # Simpan riwayat deteksi ke file JSON
    save_history(notDj_history, NOTDJ_HISTORY_FILE)

    print(f"Data received for notDj: {entry}")

    return jsonify({"message": "Data received successfully.", "entry": entry}), 200

@app.route('/reset-notDj', methods=['POST'])
def reset_not_dj():
    global notDj_history

    # Reset data riwayat notDj
    notDj_history = []

    # Kosongkan file JSON
    save_history(notDj_history, NOTDJ_HISTORY_FILE)

    print("notDj data has been reset.")
    return jsonify({"message": "notDj data has been reset."}), 200

@app.route('/purge-history', methods=['POST'])
def purge_history():
    try:
        global fall_history, notDj_history

        # Clear the in-memory history lists
        fall_history = []
        notDj_history = []

        # Clear the contents of the history files
        open(FALL_HISTORY_FILE, 'w').close()
        open(NOTDJ_HISTORY_FILE, 'w').close()

        print("All history data has been purged.")
        return jsonify({"message": "All history data has been purged."}), 200
    except Exception as e:
        print(f"Error purging history data: {e}")
        return "Internal Server Error", 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
