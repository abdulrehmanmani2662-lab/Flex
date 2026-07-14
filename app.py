import io
import os
import sys
import tempfile
import traceback
import zipfile
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, request, render_template_string, send_file
from PIL import Image, ImageFilter, ImageDraw

app = Flask(__name__)

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "u2netp.onnx")

_ort_session = None

def get_ort_session():
    global _ort_session
    if _ort_session is None:
        if not os.path.exists(MODEL_PATH):
            resp = requests.get(MODEL_URL, timeout=120)
            resp.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(resp.content)
        _ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _ort_session

def remove_background(img_rgb):
    session = get_ort_session()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    orig_w, orig_h = img_rgb.size
    small = img_rgb.resize((320, 320), Image.Resampling.LANCZOS)
    arr = np.asarray(small).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    outputs = session.run([output_name], {input_name: arr})
    mask = np.squeeze(outputs[0])
    if mask.ndim == 3: mask = mask[0]
    mask = mask - mask.min()
    denom = mask.max()
    if denom > 0: mask = mask / denom
    mask = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask, mode="L").resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    rgba = img_rgb.convert("RGBA")
    rgba.putalpha(mask_img)
    return rgba

MAX_PHOTOS = 50
OUTPUT_SIZE = 700
HD_SCALE = 1.5

PAGE = """
<!DOCTYPE html>
<html lang="ur"><head><meta charset="UTF-8"><title>Batch Background Remover</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
<style>
body{background:#0a1f13;color:#fff;font-family:sans-serif;padding:20px;}
.card{background:#fff;color:#000;padding:20px;border-radius:10px;margin-bottom:20px;}
.btn{background:#3fae5c;color:white;border:none;padding:10px;width:100%;cursor:pointer;}
</style></head><body>
<div class="card">
    <input type="file" id="fileInput" multiple accept="image/*">
    <button class="btn" id="processBtn">Sab Photos Process Karein</button>
    <div id="status"></div>
</div>
<script>
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const statusEl = document.getElementById('status');
processBtn.addEventListener('click', async () => {
    const files = Array.from(fileInput.files);
    const form = new FormData();
    files.forEach(f => form.append('photos', f));
    statusEl.textContent = "Processing...";
    const res = await fetch('/process', { method:'POST', body: form });
    if(res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = 'results.zip'; a.click();
        statusEl.textContent = "Done!";
    } else { statusEl.textContent = "Error: " + res.status; }
});
</script></body></html>
"""

@app.route("/")
def index():
    return render_template_string(PAGE, max_photos=MAX_PHOTOS)

@app.route("/process", methods=["POST"])
def process():
    photos = request.files.getlist("photos")
    results = []
    for photo in photos:
        try:
            input_img = Image.open(photo).convert("RGB")
            cutout = remove_background(input_img)
            buf = io.BytesIO()
            cutout.save(buf, format="PNG")
            buf.seek(0)
            results.append((photo.filename, buf))
            input_img.close()
        except Exception as e:
            print(f"Error: {e}")
            
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for name, buf in results:
            zf.writestr(name, buf.getvalue())
    zip_buf.seek(0)
    return send_file(zip_buf, mimetype="application/zip", as_attachment=True, download_name="results.zip")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
