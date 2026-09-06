from __future__ import annotations

import os
import glob
import json
import re
import time
import subprocess
from typing import List, Dict, Any, Optional

import cv2
import easyocr
import torch
from ultralytics import YOLO

from function import process_bubble, add_text
from api_llm import LLM, OLLAMA_MODEL

# ── Yapılandırma ──────────────────────────────────────────────
MODEL_PATH = "comic-speech-bubble-detector.pt"
INPUT_FOLDER = "/home/ugo/Documents/Python/manga/test"
OUTPUT_FOLDER = "output_images"
FONT_PATH = "/home/ugo/Documents/Python/manga/ComicRelief.ttf"

CONF_THRESHOLD = 0.15      # Düşük tut: kaçan balon, yanlış alarmdan daha kötü
IMG_SIZE = 1280            # YOLO çıkarım boyutu (640 default'u küçük balonları kaçırır)
TALL_RATIO = 2.5           # h/w bu oranı aşarsa (webtoon şeridi) dikey dilimle
TILE_OVERLAP_RATIO = 0.2   # Dilimler arası bindirme (dilim sınırındaki balonlar için)
NMS_IOU = 0.5              # Bu IoU üstü çakışan kutular tek sayılır
CONTAIN_RATIO = 0.8        # Küçük kutunun bu oranı büyüğün içindeyse kopyadır
MIN_BOX_AREA = 400         # px^2 altı gürültü kutuları ele
SAVE_DEBUG_BOXES = True    # Tespit kutuları çizilmiş kopyayı kaydet
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "tiff", "webp")

# Crop kaydı
CROPS_SUBDIR = "crops"

# Seri çeviri notları: her serinin kendi .md dosyası
SERIES_MD_DIR = "series_md"
SERIES_NAME = os.getenv("SERIES_NAME") or os.path.basename(os.path.normpath(INPUT_FOLDER))

# ── Global nesneler ───────────────────────────────────────────
llm = LLM()
model_yolo = YOLO(MODEL_PATH)
reader = easyocr.Reader(["en", "tr"])
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SERIES_MD_DIR, exist_ok=True)


# ── Yardımcılar ───────────────────────────────────────────────
def find_images(folder: str) -> List[str]:
    """Klasördeki tüm desteklenen görüntü dosyalarını bulur."""
    files: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*.{ext.upper()}")))
    return sorted(set(files))


def _select_ultralytics_device() -> Any:
    """
    Ultralytics device:
      - GPU varsa: 0
      - yoksa: 'cpu'
    """
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 0
    except Exception:
        pass
    return "cpu"


def ollama_stop(model_name: str) -> None:
    subprocess.run(
        ["ollama", "stop", model_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def free_vram() -> None:
    """PyTorch tarafında VRAM cache temizliği."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def easy_ocr_text(crop) -> str:
    """EasyOCR ile crop görüntüsünden (numpy BGR) metni okur."""
    if crop is None or crop.size == 0:
        return ""
    try:
        results = reader.readtext(crop)
    except Exception as e:
        print(f"  OCR hatası (easyocr): {e}")
        return ""
    return " ".join(str(r[1]) for r in results).strip()


def _safe_imwrite(path: str, img) -> bool:
    try:
        return bool(cv2.imwrite(path, img))
    except Exception:
        return False


# ── Seri .md dosyası ──────────────────────────────────────────
def series_md_path() -> str:
    return os.path.join(SERIES_MD_DIR, f"{SERIES_NAME}.md")


def load_series_md() -> str:
    path = series_md_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def save_series_md(content: str) -> None:
    with open(series_md_path(), "w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")


# ── Faz 1: Tespit + Crop + OCR ────────────────────────────────
def _yolo_boxes(img, conf: float) -> List[List[float]]:
    """Tek YOLO geçişi; [x1, y1, x2, y2, conf] listesi döndürür."""
    device = _select_ultralytics_device()
    results = model_yolo(img, save=False, conf=conf, imgsz=IMG_SIZE,
                         device=device, verbose=False)
    out: List[List[float]] = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].cpu().numpy())
            out.append([x1, y1, x2, y2, float(b.conf[0])])
    return out


def _merge_boxes(boxes: List[List[float]]) -> List[List[float]]:
    """Çakışan kutuları eler: IoU > NMS_IOU veya biri diğerinin içindeyse
    yüksek conf'lu kalır (tam görüntü + dilim geçişleri aynı balonu iki kez bulur)."""
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: List[List[float]] = []
    for b in boxes:
        bx1, by1, bx2, by2, _ = b
        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if b_area <= 0:
            continue
        dup = False
        for k in kept:
            kx1, ky1, kx2, ky2, _ = k
            iw = min(bx2, kx2) - max(bx1, kx1)
            ih = min(by2, ky2) - max(by1, ky1)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            k_area = (kx2 - kx1) * (ky2 - ky1)
            iou = inter / (b_area + k_area - inter)
            if iou > NMS_IOU or inter / min(b_area, k_area) > CONTAIN_RATIO:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def detect_bubbles(img) -> List[List[float]]:
    """Tam görüntü geçişi + (uzun sayfalarda) bindirmeli dikey dilim geçişleri.

    Webtoon şeritleri tek geçişte 640-1280 piksele ezilince küçük balonlar
    kaybolur; dilimler gerçek çözünürlüğe yakın çalışıp recall'u kurtarır.
    """
    h, w = img.shape[:2]
    boxes = _yolo_boxes(img, CONF_THRESHOLD)

    if h / max(w, 1) > TALL_RATIO:
        tile_h = max(int(w * 1.5), 640)
        step = max(int(tile_h * (1 - TILE_OVERLAP_RATIO)), 1)
        y = 0
        while y < h:
            y2 = min(y + tile_h, h)
            for bx1, by1, bx2, by2, c in _yolo_boxes(img[y:y2], CONF_THRESHOLD):
                boxes.append([bx1, by1 + y, bx2, by2 + y, c])
            if y2 >= h:
                break
            y += step

    return _merge_boxes(boxes)


def detect_and_ocr(img_path: str) -> Optional[Dict[str, Any]]:
    """YOLO ile balonları bulur, crop'ları kaydeder, OCR yapar, meta yazar."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Hata: Resim yüklenemedi - {img_path}")
        return None

    img_h, img_w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    crops_dir = os.path.join(OUTPUT_FOLDER, CROPS_SUBDIR, base_name)
    os.makedirs(crops_dir, exist_ok=True)
    meta_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_meta.json")

    raw_boxes = detect_bubbles(img)
    print(f"  {len(raw_boxes)} balon bulundu")

    debug_img = img.copy() if SAVE_DEBUG_BOXES else None

    items: List[Dict[str, Any]] = []
    for j, (fx1, fy1, fx2, fy2, conf) in enumerate(raw_boxes):
        x1, y1 = max(0, int(fx1)), max(0, int(fy1))
        x2, y2 = min(img_w, int(fx2)), min(img_h, int(fy2))

        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue

        crop = img[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue

        crop_path = os.path.join(crops_dir, f"bubble_{j:03d}.png")
        if not _safe_imwrite(crop_path, crop):
            continue

        if debug_img is not None:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(debug_img, f"{j} {conf:.2f}", (x1 + 2, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        items.append(
            {
                "id": j,
                "box": [x1, y1, x2, y2],
                "conf": round(conf, 3),
                "crop_path": crop_path,
                "ocr_text": "",
                "tr_text": "",
            }
        )

    if debug_img is not None:
        debug_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_boxes.jpg")
        _safe_imwrite(debug_path, debug_img)
        print(f"  Tespit görseli: {debug_path}")

    if not items:
        print("  Balon bulunamadı, geçiliyor.")
        return None

    # Okuma sırası: yukarıdan aşağı, soldan sağa (webtoon/manga sayfası)
    items.sort(key=lambda it: (it["box"][1], it["box"][0]))

    print("  OCR başlıyor (easyocr)...")
    for it in items:
        x1, y1, x2, y2 = it["box"]
        ocr = easy_ocr_text(img[y1:y2, x1:x2])
        it["ocr_text"] = ocr
        if ocr:
            print(f"  OCR {it['id']}: '{ocr}'")

    page = {"image": img_path, "base_name": base_name, "items": items}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(page, f, ensure_ascii=False, indent=2)
    print(f"  Meta kaydedildi: {meta_path}")
    return page


# ── Faz 3: Balonlara yazma ────────────────────────────────────
def render_page(page: Dict[str, Any]) -> None:
    """Çevrilmiş metinleri balonlara yazar, çıktıyı ve meta'yı kaydeder."""
    img_out = cv2.imread(page["image"])
    if img_out is None:
        print(f"  Hata: Resim yüklenemedi - {page['image']}")
        return

    for it in page["items"]:
        tr = (it["tr_text"] or "").strip()
        if not tr:
            continue

        x1, y1, x2, y2 = it["box"]
        roi = img_out[y1:y2, x1:x2].copy()

        roi_clean, contour, mask = process_bubble(roi)
        if contour is None:
            continue

        roi_final = add_text(roi_clean, tr, FONT_PATH, contour, mask)
        img_out[y1:y2, x1:x2] = roi_final

    output_path = os.path.join(OUTPUT_FOLDER, f"{page['base_name']}_translated.jpg")
    _safe_imwrite(output_path, img_out)
    print(f"  Kaydedildi: {output_path}")

    meta_path = os.path.join(OUTPUT_FOLDER, f"{page['base_name']}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(page, f, ensure_ascii=False, indent=2)


# ── Ana akış ──────────────────────────────────────────────────
def main() -> None:
    image_files = find_images(INPUT_FOLDER)
    total = len(image_files)
    print(f"Seri: {SERIES_NAME}")
    print(f"Toplam {total} görüntü bulundu\n")

    # Faz 1: bölümün tüm sayfalarında tespit + OCR
    pages: List[Dict[str, Any]] = []
    for i, path in enumerate(image_files, 1):
        print(f"[{i}/{total}] {os.path.basename(path)}")
        page = detect_and_ocr(path)
        if page:
            pages.append(page)
        print()

    if not pages:
        print("İşlenecek balon yok, çıkılıyor.")
        return

    # Çeviri modeline yer aç
    print("VRAM temizleniyor (torch cache)...")
    free_vram()
    time.sleep(0.2)

    # Faz 2: bölümün TAMAMINI tek seferde çevir (tutarlılık için)
    series_md = load_series_md()
    if series_md:
        print(f"Seri notları yüklendi: {series_md_path()}")

    todo = [it for p in pages for it in p["items"] if (it["ocr_text"] or "").strip()]
    texts = [it["ocr_text"].strip() for it in todo]
    print(f"\nBölüm çevirisi başlıyor: {len(texts)} balon, tek istek (gerekirse parçalı)...")

    translations = llm.translate_batch(texts, series_md)
    for it, tr in zip(todo, translations):
        it["tr_text"] = tr or it["ocr_text"]
        print(f"  TR: '{it['ocr_text']}' -> '{it['tr_text']}'")

    # Faz 3: sayfaları yaz
    print("\nBalonlara yazılıyor...")
    for page in pages:
        render_page(page)

    # Faz 4: seri .md dosyasını güncelle (gelecek bölümler tutarlı kalsın)
    pairs = [(it["ocr_text"].strip(), it["tr_text"].strip())
             for it in todo if (it["tr_text"] or "").strip()]
    print("\nSeri notları güncelleniyor...")
    new_md = llm.update_series_md(series_md, pairs, SERIES_NAME)
    if new_md:
        save_series_md(new_md)
        print(f"Seri notları kaydedildi: {series_md_path()}")
    else:
        print("Seri notları güncellenemedi, eski dosya korundu.")

    ollama_stop(OLLAMA_MODEL)
    free_vram()

    print("\nTüm görüntüler işlendi!")
    print(f"Çıkış klasörü: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
