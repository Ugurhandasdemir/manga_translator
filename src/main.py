from __future__ import annotations

import os
import glob
import json
import re
import time
import subprocess
from typing import List, Dict, Any, Tuple, Optional

import cv2
import torch
from ultralytics import YOLO

from function import process_bubble, add_text
from api_llm import LLM

# ── Yapılandırma ──────────────────────────────────────────────
MODEL_PATH = "comic-speech-bubble-detector.pt"
INPUT_FOLDER = "/home/ugo/Documents/Python/manga/test"
OUTPUT_FOLDER = "output_images"
FONT_PATH = "/home/ugo/Documents/Python/manga/ComicRelief.ttf"

CONF_THRESHOLD = 0.25
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "tiff", "webp")
MIN_BOX_XY = (10, 5)  # Kenar kutuları filtreleme eşiği

# OCR / LLM
OCR_MODEL_NAME = "glm-ocr"
OCR_TIMEOUT_S = 90

# Crop kaydı
CROPS_SUBDIR = "crops"

# ── Global nesneler ───────────────────────────────────────────
llm = LLM()
model_yolo = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


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


def clean_ocr_text(s: str) -> str:
    """glm-ocr çıktısındaki markdown/code-fence gibi artıkları temizler."""
    if not s:
        return ""
    s = s.strip()

    # Tam fence bloğu ise içini al: ```...```
    m = re.match(r"^```(?:\w+)?\s*(.*?)\s*```$", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()

    # Tek satırlık fence artıkları
    s = s.replace("```markdown", "").replace("```", "").strip()

    # whitespace normalize
    s = " ".join(s.split())
    return s


def glm_ocr_text(image_path: str, timeout_s: int = OCR_TIMEOUT_S) -> str:
    """
    Ollama glm-ocr ile OCR:
      ollama run glm-ocr "Text Recognition: /path/img.png"
    """
    cmd = ["ollama", "run", OCR_MODEL_NAME, f"Text Recognition: {image_path}"]
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print("  OCR timeout (glm-ocr)")
        return ""

    if p.returncode != 0:
        err = (p.stderr or "").strip()
        if err:
            print(f"  OCR hatası (glm-ocr): {err}")
        return ""

    out = clean_ocr_text((p.stdout or "").strip())
    return out


def translate_text(text: str) -> str:
    """
    Metni Türkçeye çevirir.
    Önce Ollama, patlarsa Gemini, patlarsa OpenAI fallback.
    """
    if not text:
        return ""

    # 1) Ollama
    try:
        tr = llm.ollama(text)
        if tr:
            return tr.strip()
    except Exception as e:
        print(f"  Çeviri hatası (ollama): {e}")

    # 2) Gemini
    try:
        tr = llm.gemini(text)
        if tr:
            return tr.strip()
    except Exception as e:
        print(f"  Çeviri hatası (gemini): {e}")

    # 3) OpenAI
    try:
        tr = llm.openai(text)
        if tr:
            return tr.strip()
    except Exception as e:
        print(f"  Çeviri hatası (openai): {e}")

    return text


def _safe_imwrite(path: str, img) -> bool:
    try:
        return bool(cv2.imwrite(path, img))
    except Exception:
        return False


# ── Ana iş ────────────────────────────────────────────────────
def process_image(img_path: str) -> None:
    """
    2-faz pipeline:
      Faz-1: YOLO -> box -> crop kaydet -> OCR (glm-ocr) -> meta.json
      Faz-2: glm-ocr stop + VRAM temizle -> translate -> bubble'a yaz -> çıktı kaydet
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Hata: Resim yüklenemedi - {img_path}")
        return

    img_h, img_w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Çıktı klasörleri
    crops_dir = os.path.join(OUTPUT_FOLDER, CROPS_SUBDIR, base_name)
    os.makedirs(crops_dir, exist_ok=True)

    meta_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_meta.json")

    # ---------------- FAZ 1: DETECT + CROP + OCR ----------------
    device = _select_ultralytics_device()
    results = model_yolo(img, save=False, conf=CONF_THRESHOLD, device=device)

    items: List[Dict[str, Any]] = []
    img_out = img.copy()

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        print(f"  {len(result.boxes)} balon bulundu")

        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            # Kenara çok yakın olanları ele
            if x1 < MIN_BOX_XY[0] or y1 < MIN_BOX_XY[1]:
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            crop_path = os.path.join(crops_dir, f"bubble_{j:03d}.png")
            if not _safe_imwrite(crop_path, crop):
                continue

            items.append(
                {
                    "id": j,
                    "box": [x1, y1, x2, y2],
                    "crop_path": crop_path,
                    "ocr_text": "",
                    "tr_text": "",
                }
            )

    if not items:
        print("  Balon bulunamadı, geçiliyor.")
        return

    print("  OCR başlıyor (glm-ocr)...")
    for it in items:
        ocr = glm_ocr_text(it["crop_path"], timeout_s=OCR_TIMEOUT_S)
        it["ocr_text"] = ocr
        if ocr:
            print(f"  OCR {it['id']}: '{ocr}'")

    # Meta kaydet (OCR sonrası)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"image": img_path, "base_name": base_name, "items": items},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Meta kaydedildi: {meta_path}")

    # ---------------- VRAM TEMİZLE ----------------
    print("  VRAM temizleniyor (glm-ocr stop + torch cache)...")
    ollama_stop(OCR_MODEL_NAME)
    free_vram()
    time.sleep(0.2)

    # ---------------- FAZ 2: TRANSLATE + WRITE BACK ----------------
    print("  Çeviri başlıyor...")
    for it in items:
        src = (it["ocr_text"] or "").strip()
        if not src:
            continue

        tr = translate_text(src)
        it["tr_text"] = tr
        if tr:
            print(f"  TR  {it['id']}: '{tr}'")

    # Balonlara yaz
    for it in items:
        tr = (it["tr_text"] or "").strip()
        if not tr:
            continue

        x1, y1, x2, y2 = it["box"]
        roi = img_out[y1:y2, x1:x2].copy()

        roi_clean, contour = process_bubble(roi)
        if contour is None:
            continue

        roi_final = add_text(roi_clean, tr, FONT_PATH, contour)
        img_out[y1:y2, x1:x2] = roi_final

    # Çıktı kaydet
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_translated.jpg")
    _safe_imwrite(output_path, img_out)
    print(f"  Kaydedildi: {output_path}")

    # Meta güncelle (TR sonrası)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"image": img_path, "base_name": base_name, "items": items},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Meta güncellendi: {meta_path}")


def main() -> None:
    image_files = find_images(INPUT_FOLDER)
    total = len(image_files)
    print(f"Toplam {total} görüntü bulundu\n")

    for i, path in enumerate(image_files, 1):
        print(f"[{i}/{total}] {os.path.basename(path)}")
        process_image(path)
        print()

    print("Tüm görüntüler işlendi!")
    print(f"Çıkış klasörü: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()