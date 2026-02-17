from ultralytics import YOLO
import cv2
import easyocr
import os
import glob
from function import process_bubble, add_text
from api_llm import LLM

# ── Yapılandırma ──────────────────────────────────────────────
MODEL_PATH = "comic-speech-bubble-detector.pt"
INPUT_FOLDER = "/home/ugo/Downloads"
OUTPUT_FOLDER = "output_images"
FONT_PATH = "/home/ugo/Documents/Python/manga/ComicRelief.ttf"
CONF_THRESHOLD = 0.25
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "tiff", "webp")
MIN_BOX_XY = (10, 5)  # Kenar kutuları filtreleme eşiği

# ── Global nesneler ───────────────────────────────────────────
reader = easyocr.Reader(["en", "tr"])
llm = LLM()
model_yolo = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def find_images(folder: str) -> list[str]:
    """Klasördeki tüm desteklenen görüntü dosyalarını bulur."""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*.{ext.upper()}")))
    return sorted(set(files))


def extract_text(cropped_img) -> str:
    """OCR ile kırpılmış görüntüden metin çıkarır."""
    if cropped_img.shape[0] <= 0 or cropped_img.shape[1] <= 0:
        return ""
    results = reader.readtext(cropped_img)
    return " ".join(r[1] for r in results).strip()


def translate_text(text: str) -> str:
    """Metni LLM ile Türkçeye çevirir."""
    if not text:
        return ""
    try:
        return llm.ollama(text)
    except Exception as e:
        print(f"  Çeviri hatası: {e}")
        return text


def process_image(img_path: str) -> None:
    """Tek bir görüntüyü işle: algıla → OCR → çevir → yerleştir → kaydet."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Hata: Resim yüklenemedi - {img_path}")
        return

    results = model_yolo(img, save=False, conf=CONF_THRESHOLD, device="gpu")
    img_h, img_w = img.shape[:2]
    img_out = img.copy()

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        print(f"  {len(result.boxes)} balon bulundu")

        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            # Çok kenarda kalan balonları atla
            if x1 < MIN_BOX_XY[0] or y1 < MIN_BOX_XY[1]:
                continue

            cropped = img_out[y1:y2, x1:x2]
            text = extract_text(cropped)
            if not text:
                continue

            print(f"  Balon {j}: '{text}'")
            translated = translate_text(text)
            print(f"  Çeviri {j}: '{translated}'")

            if not translated:
                continue

            roi = img_out[y1:y2, x1:x2].copy()
            roi_clean, contour = process_bubble(roi)
            if contour is not None:
                roi_final = add_text(roi_clean, translated, FONT_PATH, contour)
                img_out[y1:y2, x1:x2] = roi_final

    # Kaydet
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_translated.jpg")
    cv2.imwrite(output_path, img_out)
    print(f"  Kaydedildi: {output_path}")


def main():
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