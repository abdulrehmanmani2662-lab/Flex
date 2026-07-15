"""
Batch Background Remover — single-file Flask app.

Run:
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000 in your browser.

Everything (backend + frontend) lives in this one file on purpose,
so the whole project is just this folder: app.py + requirements.txt.
"""

import gc
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

# --- Lightweight background removal (no rembg) -----------------------------
# rembg pulls in scipy, numba, scikit-image, pymatting, networkx etc — that
# whole stack alone can eat most of a 512MB free-tier RAM limit before a
# single photo is even processed. Here we talk to the same u2netp ONNX model
# directly through onnxruntime, skipping all of that, which uses far less
# memory. Loaded lazily (first request, not import time) so a download/load
# failure shows up as a normal error instead of crashing the whole app.
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "u2netp.onnx")
MODEL_MIN_SIZE_BYTES = 1_000_000  # real file is ~4.5MB; anything smaller means a corrupt/partial download

_ort_session = None


def get_ort_session():
    global _ort_session
    if _ort_session is None:
        need_download = True
        if os.path.exists(MODEL_PATH):
            # Agar cached file chhoti/corrupt hai (adhoori download hui thi)
            # to usko delete kr ke dobara download karte hain.
            if os.path.getsize(MODEL_PATH) > MODEL_MIN_SIZE_BYTES:
                need_download = False
            else:
                print("[startup] cached model file corrupt lag rahi hai, delete kr rahe hain...", file=sys.stderr, flush=True)
                os.remove(MODEL_PATH)

        if need_download:
            print("[startup] downloading u2netp model...", file=sys.stderr, flush=True)
            resp = requests.get(MODEL_URL, timeout=300)
            resp.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(resp.content)
            print("[startup] model downloaded.", file=sys.stderr, flush=True)

        print("[startup] loading onnxruntime session...", file=sys.stderr, flush=True)
        try:
            _ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        except Exception:
            # Model file corrupt nikla — delete kr do taake agli try pe fresh download ho.
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            raise
        print("[startup] onnxruntime session ready.", file=sys.stderr, flush=True)
    return _ort_session


def estimate_background_color(img_rgb, soft_mask_arr):
    """Median color of pixels the model is confident are background —
    used to strip that color out of semi-transparent edge pixels."""
    arr = np.asarray(img_rgb).astype(np.float32)
    bg_pixels = arr[soft_mask_arr < 10]
    if len(bg_pixels) < 50:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.median(bg_pixels, axis=0)


def decontaminate_edges(rgba_img, bg_color):
    """Remove background-color bleed from semi-transparent edge pixels so
    no colored halo/glow shows when the cutout is placed on a new background."""
    arr = np.asarray(rgba_img).astype(np.float32)
    rgb = arr[..., :3]
    a = arr[..., 3:4] / 255.0
    edge_mask = (a > 0.01) & (a < 0.99)
    a_safe = np.clip(a, 0.15, 1.0)
    decontam = (rgb - (1 - a_safe) * bg_color) / a_safe
    decontam = np.clip(decontam, 0, 255)
    rgb_out = np.where(edge_mask, decontam, rgb)
    out = np.concatenate([rgb_out, arr[..., 3:4]], axis=-1).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def remove_background(img_rgb):
    """Take a PIL RGB image, return an RGBA PIL image with background removed."""
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
    if mask.ndim == 3:
        mask = mask[0]

    mask = mask - mask.min()
    denom = mask.max()
    if denom > 0:
        mask = mask / denom
    mask = (mask * 255).astype(np.uint8)

    mask_img = Image.fromarray(mask, mode="L").resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    mask_arr = np.array(mask_img)

    # u2netp ka mask fuzzy/low-res hota hai — isko "hard" bana ke zyada
    # andar se erode karte hain taake purani background ka rim/glow poori
    # tarah kat jaye, phir sirf halka sa blur anti-aliasing ke liye.
    binary = (mask_arr > 150).astype(np.uint8) * 255
    clean_mask_img = Image.fromarray(binary, mode="L")
    clean_mask_img = clean_mask_img.filter(ImageFilter.MinFilter(7))   # ~3px erode
    clean_mask_img = clean_mask_img.filter(ImageFilter.GaussianBlur(1.2))  # gentle anti-alias

    rgba = img_rgb.convert("RGBA")
    rgba.putalpha(clean_mask_img)

    bg_color = estimate_background_color(img_rgb, mask_arr)
    rgba = decontaminate_edges(rgba, bg_color)

    return rgba

MAX_PHOTOS = 50
OUTPUT_SIZE = 700
HD_SCALE = 1.5

# ---------------------------------------------------------------------------
# Frontend (HTML + CSS + JS embedded as a single template string)
# ---------------------------------------------------------------------------

PAGE = """
<!DOCTYPE html>
<html lang="ur">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Batch Background Remover</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
<style>
:root{--ink:#0f1a12;--paper:#f4f7f2;--green-deep:#12331f;--green:#1f6b3a;
--green-bright:#3fae5c;--gold:#d9a441;--line:#c9d6c6;--card:#ffffff;}
*{box-sizing:border-box;}
body{margin:0;font-family:'Segoe UI',system-ui,sans-serif;
background:radial-gradient(circle at 20% -10%,#235c34 0%,#12331f 45%,#0a1f13 100%);
color:var(--ink);min-height:100vh;}
header{padding:36px 24px 24px;text-align:center;color:var(--paper);}
header h1{font-size:clamp(22px,4vw,32px);margin:0 0 8px;}
header p{margin:0;color:#cfe3d3;font-size:14px;}
main{max-width:1000px;margin:0 auto;padding:24px;display:grid;
grid-template-columns:1fr 1fr;gap:20px;}
@media (max-width:760px){main{grid-template-columns:1fr;}}
.card{background:var(--card);border-radius:14px;padding:20px;
box-shadow:0 10px 30px rgba(0,0,0,.25);border:1px solid var(--line);}
.card.wide{grid-column:1/-1;}
.card h2{font-size:15px;text-transform:uppercase;letter-spacing:1px;
color:var(--green-deep);margin:0 0 14px;display:flex;align-items:center;gap:8px;}
.card h2 small{font-size:11px;text-transform:none;color:#889;font-weight:400;}
.badge{background:var(--green);color:#fff;width:22px;height:22px;border-radius:50%;
display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;}
.drop{border:2px dashed var(--line);border-radius:10px;padding:22px;text-align:center;
cursor:pointer;background:#fafcf9;}
.drop:hover,.drop.drag{border-color:var(--green-bright);background:#f0f8f1;}
.drop small{color:#688;display:block;margin-top:4px;font-size:12px;}
.bg-options{display:flex;flex-wrap:wrap;gap:10px;}
.swatch{width:40px;height:40px;border-radius:8px;cursor:pointer;border:2px solid transparent;}
.swatch.selected{border-color:var(--gold);box-shadow:0 0 0 2px var(--gold) inset;}
.swatch.upload-swatch{border:2px dashed var(--line);display:flex;align-items:center;
justify-content:center;font-size:16px;color:#789;background:#fafcf9;}
.row{display:flex;align-items:center;gap:10px;margin-top:14px;}
.row label{font-size:13px;color:#456;flex:1;}
.checkbox-row label{flex:none;}
input[type=range]{flex:2;accent-color:var(--green);}
.btn{padding:12px;border:none;border-radius:8px;font-size:14px;font-weight:600;
cursor:pointer;width:100%;}
.btn.primary{background:linear-gradient(135deg,var(--green-bright),var(--green));
color:#fff;margin-bottom:14px;}
.btn.primary:disabled{opacity:.5;cursor:not-allowed;}
.btn.ghost{margin-top:10px;border:1px solid var(--line);background:#fff;color:var(--green-deep);}
.btn.download{background:var(--gold);color:#28210a;margin-top:14px;}
.btn.download:disabled{opacity:.4;cursor:not-allowed;}
.status{text-align:center;font-size:13px;color:#456;min-height:18px;margin-bottom:10px;}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid var(--line);
border-top-color:var(--green);border-radius:50%;animation:spin .8s linear infinite;
vertical-align:middle;margin-inline-end:6px;}
@keyframes spin{to{transform:rotate(360deg);}}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));
gap:10px;margin-bottom:6px;}
.thumb{position:relative;border-radius:8px;overflow:hidden;aspect-ratio:1/1;
border:1px solid var(--line);}
.thumb img{width:100%;height:100%;object-fit:cover;display:block;}
footer{text-align:center;color:#9fb8a4;font-size:12px;padding:20px;}
</style>
</head>
<body>

<header>
  <h1>Batch Background Remover</h1>
  <p>Ek sath 50 photos tak background remove &amp; blend karein — HD quality ke sath</p>
</header>

<main>
  <div class="card">
    <h2><span class="badge">1</span> Photos Upload Karein <small>(max {{ max_photos }})</small></h2>
    <div class="drop" id="dropZone">
      <input type="file" id="fileInput" accept="image/*" multiple hidden>
      <div>📁 Photos yahan drop karein ya click karein</div>
      <small id="countInfo">0 / {{ max_photos }} photos selected</small>
    </div>
    <button class="btn ghost" id="clearBtn" type="button">Sab Clear Karein</button>
  </div>

  <div class="card">
    <h2><span class="badge">2</span> Background Chunein</h2>
    <div class="bg-options" id="bgOptions">
      <div class="swatch selected" data-bg="transparent" title="Background nahi, sirf cutout" style="background-image:linear-gradient(45deg,#ccc 25%,transparent 25%),linear-gradient(-45deg,#ccc 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ccc 75%),linear-gradient(-45deg,transparent 75%,#ccc 75%);background-size:12px 12px;background-position:0 0,0 6px,6px -6px,-6px 0px;background-color:#fff;"></div>
      <div class="swatch" data-bg="green-poster" style="background:linear-gradient(135deg,#3fae5c,#0a1f13)"></div>
      <div class="swatch" data-bg="#0a1f13" style="background:#0a1f13"></div>
      <div class="swatch" data-bg="#ffffff" style="background:#ffffff"></div>
      <div class="swatch" data-bg="#0b1e3d" style="background:#0b1e3d"></div>
      <div class="swatch" data-bg="#7a1f1f" style="background:#7a1f1f"></div>
      <div class="swatch upload-swatch" id="bgUploadSwatch">+</div>
      <input type="file" id="bgInput" accept="image/*" hidden>
    </div>
    <div class="row"><label>Edge Blend</label><input type="range" id="featherRange" min="0" max="20" value="2"></div>
    <div class="row"><label>Photo Size</label><input type="range" id="scaleRange" min="40" max="150" value="90"></div>
    <div class="row checkbox-row"><label>HD Enhance</label><input type="checkbox" id="hdToggle"></div>
    <div class="row checkbox-row"><label>Bottom Fade (background mein blend)</label><input type="checkbox" id="fadeToggle" checked></div>
  </div>

  <div class="card wide">
    <h2><span class="badge">3</span> Process &amp; Download</h2>
    <button class="btn primary" id="processBtn" disabled>Sab Photos Process Karein</button>
    <div class="progress-bar" id="progressBar" style="display:none;height:8px;background:#e6ede6;border-radius:6px;overflow:hidden;margin-bottom:10px;">
      <div id="progressFill" style="height:100%;width:0%;background:var(--green-bright);transition:width .2s;"></div>
    </div>
    <div class="status" id="status"></div>
    <div class="gallery" id="gallery"></div>
  </div>
</main>

<footer>Processing server par hoti hai (Python / rembg) — is machine par jo app.py chal raha hai.</footer>

<script>
const MAX_PHOTOS = {{ max_photos }};
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const countInfo = document.getElementById('countInfo');
const clearBtn = document.getElementById('clearBtn');
const bgOptions = document.getElementById('bgOptions');
const bgUploadSwatch = document.getElementById('bgUploadSwatch');
const bgInput = document.getElementById('bgInput');
const featherRange = document.getElementById('featherRange');
const scaleRange = document.getElementById('scaleRange');
const hdToggle = document.getElementById('hdToggle');
const fadeToggle = document.getElementById('fadeToggle');
const processBtn = document.getElementById('processBtn');
const statusEl = document.getElementById('status');
const gallery = document.getElementById('gallery');

let files = [];
let bgMode = 'transparent';
let bgFile = null;

function setStatus(msg, spinning=false){
  statusEl.innerHTML = spinning ? `<span class="spinner"></span>${msg}` : msg;
}

dropZone.addEventListener('click', () => fileInput.click());
['dragover','dragleave','drop'].forEach(evt=>{
  dropZone.addEventListener(evt, e=>{ e.preventDefault(); dropZone.classList.toggle('drag', evt==='dragover'); });
});
dropZone.addEventListener('drop', e => addFiles([...e.dataTransfer.files]));
fileInput.addEventListener('change', e => addFiles([...e.target.files]));

function addFiles(list){
  const imgs = list.filter(f => f.type.startsWith('image/'));
  const room = MAX_PHOTOS - files.length;
  if(room <= 0){ setStatus(`Limit ${MAX_PHOTOS} photos hai.`); return; }
  imgs.slice(0, room).forEach(f=>{
    files.push(f);
    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    thumb.innerHTML = `<img src="${URL.createObjectURL(f)}">`;
    gallery.appendChild(thumb);
  });
  countInfo.textContent = `${files.length} / ${MAX_PHOTOS} photos selected`;
  processBtn.disabled = files.length === 0;
  fileInput.value = '';
}

clearBtn.addEventListener('click', ()=>{
  files = []; gallery.innerHTML=''; setStatus('');
  countInfo.textContent = `0 / ${MAX_PHOTOS} photos selected`;
  processBtn.disabled = true;
});

bgOptions.addEventListener('click', e=>{
  const sw = e.target.closest('.swatch');
  if(!sw || sw === bgUploadSwatch) return;
  document.querySelectorAll('.swatch').forEach(s=>s.classList.remove('selected'));
  sw.classList.add('selected');
  bgMode = sw.dataset.bg; bgFile = null;
});
bgUploadSwatch.addEventListener('click', ()=>bgInput.click());
bgInput.addEventListener('change', e=>{
  bgFile = e.target.files[0];
  if(!bgFile) return;
  bgMode = 'custom-image';
  document.querySelectorAll('.swatch').forEach(s=>s.classList.remove('selected'));
  bgUploadSwatch.classList.add('selected');
});

// Photos ko chhote batches me bhejte hain taake ek request bhaari hokar
// server par timeout/crash (502) na ho. Har batch ka result (PNG ya ZIP)
// browser me hi ek final ZIP me jama hota hai.
const CHUNK_SIZE = 4;
const progressBar = document.getElementById('progressBar');
const progressFill = document.getElementById('progressFill');

function chunkArray(arr, size){
  const out = [];
  for(let i=0;i<arr.length;i+=size) out.push(arr.slice(i,i+size));
  return out;
}

async function processChunk(chunkFiles){
  const form = new FormData();
  chunkFiles.forEach(f => form.append('photos', f));
  form.append('bg_mode', bgMode);
  if(bgFile) form.append('bg_image', bgFile);
  form.append('feather', featherRange.value);
  form.append('scale', scaleRange.value);
  form.append('hd', hdToggle.checked ? '1' : '0');
  form.append('fade_bottom', fadeToggle.checked ? '1' : '0');

  const res = await fetch('/process', { method:'POST', body: form });
  if(!res.ok){
    let detail = `status ${res.status}`;
    try{
      const j = await res.clone().json();
      if(j.details && j.details.length) detail = j.details.join(' | ');
      else if(j.error) detail = j.error;
    }catch(e){ /* response wasn't JSON, keep generic status */ }
    throw new Error(detail);
  }
  const blob = await res.blob();
  const contentType = res.headers.get('Content-Type') || '';
  return { blob, isZip: contentType.includes('zip') };
}

processBtn.addEventListener('click', async ()=>{
  if(files.length === 0) return;
  processBtn.disabled = true;
  progressBar.style.display = 'block';
  progressFill.style.width = '0%';

  const chunks = chunkArray(files, CHUNK_SIZE);
  const finalZip = new JSZip();
  let photoCounter = 0;
  let failedCount = 0;
  let singlePngBlob = null;
  let lastError = '';

  for(let i=0; i<chunks.length; i++){
    setStatus(`Batch ${i+1} / ${chunks.length} process ho raha hai... (${photoCounter}/${files.length} mukammal)`, true);
    try{
      const { blob, isZip } = await processChunk(chunks[i]);
      if(isZip){
        const loaded = await JSZip.loadAsync(blob);
        const entries = Object.values(loaded.files);
        for(const entry of entries){
          const content = await entry.async('blob');
          photoCounter++;
          finalZip.file(`photo-${String(photoCounter).padStart(3,'0')}.png`, content);
        }
      } else {
        photoCounter++;
        singlePngBlob = blob;
        finalZip.file(`photo-${String(photoCounter).padStart(3,'0')}.png`, blob);
      }
    }catch(err){
      console.error(err);
      lastError = err.message;
      failedCount += chunks[i].length;
    }
    progressFill.style.width = `${Math.round(((i+1)/chunks.length)*100)}%`;
  }

  if(photoCounter === 0){
    setStatus(`Masla: ${lastError || 'Wajah nahi mili, dobara koshish karein.'}`);
    processBtn.disabled = false;
    return;
  }

  if(photoCounter === 1 && singlePngBlob){
    const link = document.createElement('a');
    link.href = URL.createObjectURL(singlePngBlob);
    link.download = 'blended-photo.png';
    link.click();
  } else {
    setStatus('ZIP taiyar ho raha hai...', true);
    const zipBlob = await finalZip.generateAsync({ type:'blob' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(zipBlob);
    link.download = 'blended-photos.zip';
    link.click();
  }

  const failMsg = failedCount > 0 ? ` (${failedCount} photos fail ho gayin)` : '';
  setStatus(`Mukammal ✅ — ${photoCounter} photos download ho gayin${failMsg}.`);
  processBtn.disabled = false;
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------


def paint_background(size, bg_mode, bg_image_bytes):
    """Return an RGBA background image of (size, size)."""
    w = h = size
    if bg_mode == "custom-image" and bg_image_bytes:
        bg = Image.open(io.BytesIO(bg_image_bytes)).convert("RGB")
        bw, bh = bg.size
        ratio = max(w / bw, h / bh)
        bg = bg.resize((int(bw * ratio) + 1, int(bh * ratio) + 1))
        left = (bg.width - w) // 2
        top = (bg.height - h) // 2
        bg = bg.crop((left, top, left + w, top + h))
        return bg.convert("RGBA")

    if bg_mode == "green-poster":
        canvas = Image.new("RGBA", (w, h), "#0a1f13")
        draw = ImageDraw.Draw(canvas)
        cx, cy = w * 0.5, h * 0.45
        max_r = int(w * 0.9)
        stops = [(0.0, (63, 174, 92)), (0.45, (31, 107, 58)), (1.0, (10, 31, 19))]
        for r in range(max_r, 0, -2):
            t = r / max_r
            for i in range(len(stops) - 1):
                t0, c0 = stops[i]
                t1, c1 = stops[i + 1]
                if t0 <= t <= t1:
                    f = (t - t0) / (t1 - t0) if t1 > t0 else 0
                    color = tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))
                    break
            else:
                color = stops[-1][1]
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (255,))
        return canvas

    # plain hex color, e.g. "#ffffff"
    color = bg_mode if bg_mode.startswith("#") else "#ffffff"
    return Image.new("RGBA", (w, h), color)


def fade_bottom_edge(rgba_img, fade_ratio=0.22):
    """Gradually fade alpha to 0 over the bottom portion of the image so the
    subject blends into whatever background it's placed on, instead of a
    hard rectangular cutoff at the bottom."""
    w, h = rgba_img.size
    fade_h = int(h * fade_ratio)
    if fade_h <= 0:
        return rgba_img
    arr = np.array(rgba_img).astype(np.float32)
    grad = np.ones(h, dtype=np.float32)
    grad[h - fade_h:] = np.linspace(1.0, 0.0, fade_h)
    grad = grad[:, None]
    arr[..., 3] = arr[..., 3] * grad
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")


def trim_faint_alpha(rgba_img, threshold=20):
    """After fading, drop rows/cols that are almost fully transparent so the
    crop box stays tight — only a short visible fade band remains, not a
    big empty transparent area."""
    arr = np.array(rgba_img)
    alpha = arr[..., 3]
    rows = np.where(alpha.max(axis=1) > threshold)[0]
    cols = np.where(alpha.max(axis=0) > threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        return rgba_img
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols.min()), int(cols.max())
    return rgba_img.crop((x0, y0, x1 + 1, y1 + 1))


def crop_to_subject(cutout_rgba, padding_ratio=0.08):
    """Tight-crop an RGBA cutout to its non-transparent bounding box, with a
    small padding margin, instead of leaving it centered on a big square."""
    alpha = np.array(cutout_rgba.split()[-1])
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        return cutout_rgba  # nothing detected, return unchanged
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = x1 - x0, y1 - y0
    pad_x, pad_y = int(w * padding_ratio), int(h * padding_ratio)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(cutout_rgba.width, x1 + pad_x)
    y1 = min(cutout_rgba.height, y1 + pad_y)
    return cutout_rgba.crop((x0, y0, x1, y1))


def compose_on_background(cutout_rgba, bg_mode, bg_image_bytes, feather, scale_pct, hd, fade_bottom=True):
    if feather > 0:
        alpha = cutout_rgba.split()[-1]
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
        cutout_rgba.putalpha(alpha)

    cropped = crop_to_subject(cutout_rgba)
    if fade_bottom:
        cropped = fade_bottom_edge(cropped, fade_ratio=0.15)
        cropped = trim_faint_alpha(cropped, threshold=20)

    if bg_mode == "transparent":
        # No background painted — khud transparent PNG, tight-cropped around
        # the subject. Koi background select karne ki zaroorat nahi.
        result = cropped
        if hd:
            new_w = int(result.width * HD_SCALE)
            new_h = int(result.height * HD_SCALE)
            result = result.resize((new_w, new_h), Image.Resampling.LANCZOS)
            r, g, b, a = result.split()
            rgb = Image.merge("RGB", (r, g, b)).filter(
                ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2)
            )
            r2, g2, b2 = rgb.split()
            result = Image.merge("RGBA", (r2, g2, b2, a))
        return result

    size = OUTPUT_SIZE
    background = paint_background(size, bg_mode, bg_image_bytes)

    target_h = size * scale_pct
    target_w = target_h * (cropped.width / cropped.height)
    subject = cropped.resize((int(target_w), int(target_h)), Image.Resampling.LANCZOS)

    dx = int((size - target_w) / 2)
    dy = int(size - target_h - size * 0.02)

    result = background.copy()
    result.alpha_composite(subject, (dx, dy))
    result = result.convert("RGB")

    if hd:
        result = result.resize((int(size * HD_SCALE), int(size * HD_SCALE)), Image.Resampling.LANCZOS)
        result = result.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template_string(PAGE, max_photos=MAX_PHOTOS)


@app.route("/process", methods=["POST"])
def process():
    photos = request.files.getlist("photos")[:MAX_PHOTOS]
    if not photos:
        return {"error": "No photos received"}, 400

    bg_mode = request.form.get("bg_mode", "green-poster")
    feather = int(request.form.get("feather", 6))
    scale_pct = int(request.form.get("scale", 90)) / 100
    hd = request.form.get("hd", "1") == "1"
    fade_bottom = request.form.get("fade_bottom", "1") == "1"

    bg_image_bytes = None
    bg_file = request.files.get("bg_image")
    if bg_file and bg_file.filename:
        bg_image_bytes = bg_file.read()

    results = []
    errors = []
    for idx, photo in enumerate(photos, start=1):
        print(f"[process] photo {idx}/{len(photos)}: {photo.filename}", file=sys.stderr, flush=True)
        try:
            raw = photo.read()
            input_img = Image.open(io.BytesIO(raw)).convert("RGB")
            cutout_img = remove_background(input_img)
            final_img = compose_on_background(
                cutout_img, bg_mode, bg_image_bytes, feather, scale_pct, hd, fade_bottom
            )
            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            buf.seek(0)
            results.append(buf)
            del raw, input_img, cutout_img, final_img
        except Exception as e:
            err_text = f"{photo.filename}: {type(e).__name__}: {e}"
            errors.append(err_text)
            print(f"[process] FAILED on {err_text}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
        finally:
            gc.collect()

    if not results:
        return {"error": "All photos failed to process", "details": errors}, 500

    if len(results) == 1:
        return send_file(
            results[0],
            mimetype="image/png",
            as_attachment=True,
            download_name="blended-photo.png",
        )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, buf in enumerate(results, start=1):
            zf.writestr(f"photo-{i:02d}.png", buf.getvalue())
    zip_buf.seek(0)

    return send_file(
        zip_buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="blended-photos.zip",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
