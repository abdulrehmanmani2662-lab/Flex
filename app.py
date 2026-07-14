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
import zipfile

from flask import Flask, request, render_template_string, send_file
from PIL import Image, ImageFilter, ImageDraw

from rembg import new_session, remove

app = Flask(__name__)

# u2netp is the "lite" rembg model (~4MB vs ~170MB for the default u2net) —
# uses far less RAM, which matters on free-tier hosting (512MB limit).
_session = new_session("u2netp")

MAX_PHOTOS = 50
OUTPUT_SIZE = 1000
HD_SCALE = 2

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
      <div class="swatch selected" data-bg="green-poster" style="background:linear-gradient(135deg,#3fae5c,#0a1f13)"></div>
      <div class="swatch" data-bg="#0a1f13" style="background:#0a1f13"></div>
      <div class="swatch" data-bg="#ffffff" style="background:#ffffff"></div>
      <div class="swatch" data-bg="#0b1e3d" style="background:#0b1e3d"></div>
      <div class="swatch" data-bg="#7a1f1f" style="background:#7a1f1f"></div>
      <div class="swatch upload-swatch" id="bgUploadSwatch">+</div>
      <input type="file" id="bgInput" accept="image/*" hidden>
    </div>
    <div class="row"><label>Edge Blend</label><input type="range" id="featherRange" min="0" max="20" value="6"></div>
    <div class="row"><label>Photo Size</label><input type="range" id="scaleRange" min="40" max="150" value="90"></div>
    <div class="row checkbox-row"><label>HD Enhance</label><input type="checkbox" id="hdToggle" checked></div>
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
const processBtn = document.getElementById('processBtn');
const statusEl = document.getElementById('status');
const gallery = document.getElementById('gallery');

let files = [];
let bgMode = 'green-poster';
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

  const res = await fetch('/process', { method:'POST', body: form });
  if(!res.ok) throw new Error(`Batch failed (status ${res.status})`);
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
      failedCount += chunks[i].length;
    }
    progressFill.style.width = `${Math.round(((i+1)/chunks.length)*100)}%`;
  }

  if(photoCounter === 0){
    setStatus('Kuch masla hua — koi photo process nahi ho saki. Dobara koshish karein.');
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


def compose_on_background(cutout_rgba, bg_mode, bg_image_bytes, feather, scale_pct, hd):
    size = OUTPUT_SIZE
    background = paint_background(size, bg_mode, bg_image_bytes)

    target_h = size * scale_pct
    target_w = target_h * (cutout_rgba.width / cutout_rgba.height)
    subject = cutout_rgba.resize((int(target_w), int(target_h)), Image.LANCZOS)

    dx = int((size - target_w) / 2)
    dy = int(size - target_h - size * 0.02)

    if feather > 0:
        alpha = subject.split()[-1]
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
        subject.putalpha(alpha)

    result = background.copy()
    result.alpha_composite(subject, (dx, dy))
    result = result.convert("RGB")

    if hd:
        result = result.resize((size * HD_SCALE, size * HD_SCALE), Image.LANCZOS)
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

    bg_image_bytes = None
    bg_file = request.files.get("bg_image")
    if bg_file and bg_file.filename:
        bg_image_bytes = bg_file.read()

    results = []
    for photo in photos:
        raw = photo.read()
        cutout = remove(raw, session=_session)  # bytes in -> bytes out (RGBA PNG), via rembg
        cutout_img = Image.open(io.BytesIO(cutout)).convert("RGBA")
        final_img = compose_on_background(
            cutout_img, bg_mode, bg_image_bytes, feather, scale_pct, hd
        )
        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        buf.seek(0)
        results.append(buf)

        del raw, cutout, cutout_img, final_img
        gc.collect()

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
