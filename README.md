# Batch Background Remover (Python / Flask, single folder)

Sab kuch is ek folder me hai — `app.py` (backend + frontend dono is file ke andar) aur
`requirements.txt`.

## Chalane ka tareeqa

```bash
pip install -r requirements.txt
python app.py
```

Phir browser me kholein: **http://127.0.0.1:5000**

Pehli baar chalane par `rembg` apna AI model (~170MB) download karega — thoda time lagega,
uske baad fast chalega.

## Kya karta hai

- Ek sath **50 photos** tak upload
- Har photo ka background AI se remove (Python `rembg` library, server par process hota hai)
- Naye background (color ya apni image) ke sath naturally blend (feather edges)
- HD Enhance — 2x upscale + sharpening
- 1 photo → seedha PNG download, zyada photos → automatic **ZIP** (extract karne par sab
  ek folder me, gallery jaisi)

## Design badalna

`app.py` ke andar `PAGE` variable me poora HTML/CSS/JS hai — Urdu text, rangon, layout
sab wahin se badal sakte ho. Ek hi file hai, dhoondna asaan hai.

## Deploy karna (agar internet par live karna ho)

Ye Flask app hai, GitHub Pages (static hosting) par nahi chalegi — kisi Python hosting
chahiye, jaise Render, Railway, ya apna VPS. Bata dena agar deploy me madad chahiye ho.
