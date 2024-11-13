import os
import base64
import json
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

# Direktori untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Nama file untuk menyimpan riwayat deteksi
HISTORY_FILE = 'fall_history.json'

# Membaca riwayat deteksi dari file JSON jika ada
def load_fall_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

# Menyimpan riwayat deteksi ke dalam file JSON
def save_fall_history(fall_history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(fall_history, f)

# Inisialisasi riwayat deteksi dari file
fall_history = load_fall_history()

@app.route('/')
def index():
    # Mendapatkan riwayat deteksi dan mengonversi gambar menjadi Base64
    history_with_images = []
    for entry in fall_history:
        entry_with_image = entry.copy()
        if entry["image_filename"]:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], entry["image_filename"])
            
            # Konversi gambar ke Base64
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                entry_with_image["image_base64"] = f"data:image/jpeg;base64,{encoded_string}"
        history_with_images.append(entry_with_image)
    
    # Render template dengan data riwayat deteksi
    return render_template('index.html', fall_history=history_with_images)

@app.route('/fall-detected', methods=['POST'])
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

    # Filter riwayat deteksi untuk hanya menyimpan deteksi terakhir per waktu
    fall_history = [entry for entry in fall_history if entry["fall_time"][:16] != fall_time[:16]]

    # Simpan data deteksi ke dalam riwayat
    fall_entry = {
        "fall_detected": fall_detected,
        "fall_time": fall_time,
        "image_filename": image_filename
    }
    fall_history.append(fall_entry)

    # Simpan riwayat deteksi ke file JSON
    save_fall_history(fall_history)

    print(f"Fall detected! Data received: {fall_entry}")

    return "Fall detection data received.", 200

if __name__ == "__main__":
    # Jalankan server Flask di host 0.0.0.0 untuk dapat diakses oleh perangkat lain dalam jaringan
    app.run(host='0.0.0.0', port=5000, debug=True)
