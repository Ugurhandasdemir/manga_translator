from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from deep_translator import GoogleTranslator
import os
import glob
from PIL import Image, ImageDraw, ImageFont

# Google Translator'ı başlat
translator = GoogleTranslator(source='auto', target='tr')

# EasyOCR modelini başlat
reader = easyocr.Reader(['en', 'tr'])

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

if not image_files:
    print(f"Hata: {input_folder} klasöründe görüntü dosyası bulunamadı!")
    exit()

print(f"Toplam {len(image_files)} görüntü dosyası bulundu")

# Türkçe karakterleri destekleyen font
FONT_PATH = "/home/ugo/Documents/manga/ComicRelief.ttf"
FONT_SIZE = 40
TEXT_COLOR = (0, 0, 0) # Siyah

# PIL ile metin satırlarına bölme fonksiyonu
def wrap_text_pil(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]
        
        if width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines[:3] # En fazla 3 satır

# Her görüntü dosyasını işle
for img_index, img_path in enumerate(image_files):
    print(f"\n{'='*50}")
    print(f"İşleniyor: {os.path.basename(img_path)} ({img_index+1}/{len(image_files)})")
    print(f"{'='*50}")
    
    # Resmi yükle ve kontrol et
    img = cv2.imread(img_path)
    if img is None:
        print(f"Hata: Resim yüklenemedi - {img_path}")
        continue

    # YOLO modelini çalıştır
    results = model_yolo(img, save=False, conf=0.25, device="cpu")

    print(f"Toplam {len(results)} sonuç bulundu")

    for i, result in enumerate(results):
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"Sonuç {i}: {len(result.boxes)} kutu bulundu")

            img_with_text = img.copy()

            # OpenCV -> PIL dönüşümü
            img_pil = Image.fromarray(cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

            for j, box in enumerate(result.boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                cropped_img = img_with_text[y1:y2, x1:x2]

                extracted_text = ""

                if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                    # EasyOCR ile metin çıkarma
                    ocr_results = reader.readtext(cropped_img)
                    for res in ocr_results:
                        extracted_text += res[1] + " "
                    print(f"Balon {j}'den çıkarılan metin: {extracted_text.strip()}")
                else:
                    print(f"Balon {j} için boş veya geçersiz kırpılmış resim atlanıyor")

                                # Google Translate ile çeviri yap
                if extracted_text.strip():
                    try:
                        translated_text = translator.translate(extracted_text.strip())
                        print(f"Balon {j} için Google Translate çevirisi: {translated_text}")
                    except Exception as e:
                        print(f"Çeviri hatası: {e}")
                        translated_text = extracted_text.strip()
                else:
                    translated_text = ""
                    
                if not translated_text == "" and x1>10 and y1 > 10:
                    # Beyaz elips çiz (PIL ile)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                    
                    # PIL'de elips çizmek için bbox kullanıyoruz
                    bbox = [x1, y1, x2, y2]
                    draw.ellipse(bbox, fill='white')
                    
                if translated_text.strip():
                    # Baloncuğun yaklaşık genişliğini al
                    bubble_width = x2 - x1
                    text_max_width = int(bubble_width * 0.9)

                    # Metni satırlara böl
                    wrapped_lines = wrap_text_pil(translated_text.strip(), font, text_max_width)

                    # Satır yüksekliğini hesapla
                    test_bbox = font.getbbox("Test")
                    line_height = test_bbox[3] - test_bbox[1] + 5
                    total_text_height = len(wrapped_lines) * line_height

                    # Metni dikeyde ortala
                    start_y = center[1] - total_text_height // 2

                    for line_idx, line in enumerate(wrapped_lines):
                        line_bbox = font.getbbox(line)
                        line_width = line_bbox[2] - line_bbox[0]
                        line_x = center[0] - line_width // 2
                        line_y = start_y + (line_idx * line_height)

                        # Sınırları kontrol et
                        line_x = max(0, line_x)
                        line_y = max(0, line_y)

                        draw.text((line_x, line_y), line, font=font, fill=TEXT_COLOR)

            # PIL -> OpenCV dönüşümü
            img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Çıkış dosya adını oluştur
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_translated.jpg")
            cv2.imwrite(output_path, img_with_text)
            print(f"Sonuç kaydedildi: {output_path}")

print(f"\n{'='*50}")
print("Tüm görüntüler işlendi!")
print(f"Çıkış klasörü: {output_folder}")
print(f"{'='*50}")