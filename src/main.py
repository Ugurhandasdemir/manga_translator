from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from deep_translator import GoogleTranslator
import os
import glob
from function import process_bubble, add_text
from api_llm import LLM

translator = GoogleTranslator(source='auto', target='tr')
reader = easyocr.Reader(['en', 'tr'])
llm = LLM()

# Model ve klasör yolu
model_yolo = YOLO("/home/ugo/Documents/manga/comic-speech-bubble-detector.pt")
input_folder = "/home/ugo/Downloads/Enoch_ Shining Tree - Chapter 16 - Manhwa Clan"  # Giriş klasörü
output_folder = "/home/ugo/Documents/manga/output_images"  # Çıkış klasörü

# Çıkış klasörünü oluştur
os.makedirs(output_folder, exist_ok=True)

# Desteklenen görüntü formatları
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']

# Klasördeki tüm görüntü dosyalarını bul
image_files = []
for extension in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_folder, extension)))
    image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))

print(f"Toplam {len(image_files)} görüntü dosyası bulundu")

# Türkçe karakterleri destekleyen font
FONT_PATH = "/home/ugo/Documents/manga/ComicRelief.ttf"
FONT_SIZE = 40
TEXT_COLOR = (0, 0, 0) # Siyah



for img_index, img_path in enumerate(image_files):
    print(f"\n{'='*50}")
    print(f"İşleniyor: {os.path.basename(img_path)} ({img_index+1}/{len(image_files)})")
    print(f"{'='*50}")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Hata: Resim yüklenemedi - {img_path}")
        continue

    results = model_yolo(img, save=False, conf=0.25, device="cpu")

    print(f"Toplam {len(results)} sonuç bulundu")

    for i, result in enumerate(results):
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"Sonuç {i}: {len(result.boxes)} kutu bulundu")

            img_with_text = img.copy()

            for j, box in enumerate(result.boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                cropped_img = img_with_text[y1:y2, x1:x2]

                extracted_text = ""

                if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                    ocr_results = reader.readtext(cropped_img)
                    for res in ocr_results:
                        extracted_text += res[1] + " "
                    print(f"Balon {j}'den çıkarılan metin: {extracted_text.strip()}")
                else:
                    print(f"Balon {j} için boş veya geçersiz kırpılmış resim atlanıyor")

                if extracted_text.strip():
                    try:
                        translated_text = llm.openai(extracted_text.strip())
                        #translated_text = translator.translate(extracted_text.strip())
                        print(f"Balon {j} için Google Translate çevirisi: {translated_text}")
                    except Exception as e:
                        print(f"Çeviri hatası: {e}")
                        translated_text = extracted_text.strip()
                else:
                    translated_text = ""
                    
                if not translated_text == "" and x1>10 and y1 > 10:
                    roi = img_with_text[y1:y2, x1:x2].copy()

                    roi_processed, largest_contour = process_bubble(roi)
        
                    if largest_contour is not None:
                        roi_with_text = add_text(roi_processed, translated_text, FONT_PATH, largest_contour)
                        img_with_text[y1:y2, x1:x2] = roi_with_text


            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_translated.jpg")
            cv2.imwrite(output_path, img_with_text)
            print(f"Sonuç kaydedildi: {output_path}")

print(f"\n{'='*50}")
print("Tüm görüntüler işlendi!")
print(f"Çıkış klasörü: {output_folder}")
print(f"{'='*50}")