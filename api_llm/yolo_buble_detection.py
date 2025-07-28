from ultralytics import YOLO
import cv2
import numpy as np
from doctr.models import ocr_predictor
from llm import LLM
from google import genai
import os
import glob
from PIL import Image, ImageDraw, ImageFont

# Google API anahtarını ayarla
os.environ["GOOGLE_API_KEY"] = "AIzaSyCmRWQCL2-yvRr4FkvEmSqxmZBPVeLWZXw"

llm = LLM()

# OCR modelini başlat
model_ocr = ocr_predictor(pretrained=True)

# Model ve resim klasörü yolu
model_yolo = YOLO("/home/ugo/Documents/manga/comic-speech-bubble-detector.pt")
img_folder = "/home/ugo/Downloads/Urek Mazino Chapter 7. Blackout _ Urek Mazino_ Tower of God Sidestory"

# Klasördeki tüm resim dosyalarını al
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_files = []

for extension in image_extensions:
    image_files.extend(glob.glob(os.path.join(img_folder, extension)))
    image_files.extend(glob.glob(os.path.join(img_folder, extension.upper())))

if not image_files:
    print(f"Hata: {img_folder} klasöründe resim dosyası bulunamadı!")
    print("Klasör yolunu kontrol edin!")
    exit()

print(f"Toplam {len(image_files)} resim dosyası bulundu")

# TTF Font yolu - Comic Sans MS
FONT_PATH = "/home/ugo/Downloads/comic.ttf"  # Comic Sans MS font dosyası
FONT_SIZE = 24
TEXT_COLOR = (0, 0, 0)  # Siyah renk (RGB)

# Font dosyasının varlığını kontrol et
if not os.path.exists(FONT_PATH):
    print(f"Uyarı: Font dosyası bulunamadı: {FONT_PATH}")
    print("Varsayılan font kullanılacak")
    FONT_PATH = None

# Metni belirli bir genişliğe göre satırlara bölen yardımcı fonksiyon (PIL için)
def wrap_text_pil(text, max_width, font):
    words = text.split(' ')
    lines = []
    current_line = []

    for word in words:
        # Mevcut satıra kelime eklenmiş halinin genişliğini test et
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current_line.append(word)
        else:
            # Eğer kelime mevcut satırı taşırıyorsa, yeni satıra geç
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            
            # Tek kelime bile max_width'i aşıyorsa zorla ekle
            bbox = font.getbbox(word)
            word_width = bbox[2] - bbox[0]
            if word_width > max_width and not lines:
                lines.append(word)
                current_line = []
            elif word_width > max_width and lines and not current_line:
                lines.append(word)
                current_line = []

    if current_line:
        lines.append(' '.join(current_line))

    # Satır sayısını 3 ile sınırla
    return lines[:3]

# OpenCV görüntüsünü PIL'e çeviren fonksiyon
def cv2_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# PIL görüntüsünü OpenCV'ye çeviren fonksiyon
def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Her resim dosyasını işle
for img_idx, img_path in enumerate(image_files):
    print(f"\n=== İşleniyor: {os.path.basename(img_path)} ({img_idx+1}/{len(image_files)}) ===")
    
    # Resmi yükle ve kontrol et
    img = cv2.imread(img_path)
    if img is None:
        print(f"Hata: Resim yüklenemedi - {img_path}")
        print("Dosya atlanıyor...")
        continue

    # YOLO modelini çalıştır
    results = model_yolo(img, save=False, conf=0.25, device="cpu")

    print(f"Toplam {len(results)} sonuç bulundu")

    for i, result in enumerate(results):
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"Sonuç {i}: {len(result.boxes)} kutu bulundu")

            img_with_text = img.copy() 
            
            # OpenCV görüntüsünü PIL'e çevir
            pil_img = cv2_to_pil(img_with_text)
            draw = ImageDraw.Draw(pil_img)
            
            # Font yükle
            try:
                if FONT_PATH and os.path.exists(FONT_PATH):
                    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
                else:
                    font = ImageFont.load_default()
                    print("Varsayılan font kullanılıyor")
            except Exception as e:
                print(f"Font yükleme hatası: {e}")
                font = ImageFont.load_default()

            for j, box in enumerate(result.boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                cropped_img = img_with_text[y1:y2, x1:x2]

                extracted_text = ""

                if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                    result_ocr = model_ocr([cropped_img])

                    for page in result_ocr.pages:
                        for block in page.blocks:
                            for line in block.lines:
                                extracted_text += " ".join([word.value for word in line.words]) + " "
                    print(f"Balon {j}'den çıkarılan metin: {extracted_text.strip()}")
                else:
                    print(f"Balon {j} için boş veya geçersiz kırpılmış resim atlanıyor")

                # Baloncuğu beyaz elips ile doldur (PIL kullanarak)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                draw.ellipse([x1, y1, x2, y2], fill='white', outline='white')
                
                # LLM çevirisi
                if extracted_text.strip():
                    try:
                        extracted_text = llm.translet(extracted_text)
                        print(f"Balon {j} için LLM cevabı: {extracted_text.strip()}")
                    except Exception as e:
                        print(f"LLM çeviri hatası: {e}")
                        print("Orijinal metin korunuyor")

                if extracted_text.strip(): # Sadece metin çıkarıldıysa ekle
                    # Baloncuğun yaklaşık genişliğini al (metin için kullanılabilecek alan)
                    bubble_width = x2 - x1
                    # Biraz kenar boşluğu bırakmak için genişliği azalt
                    text_max_width = int(bubble_width * 0.8)

                    # Metni satırlara böl
                    wrapped_lines = wrap_text_pil(extracted_text.strip(), text_max_width, font)

                    # Satır yüksekliğini hesapla
                    bbox = font.getbbox("Test")
                    line_height = bbox[3] - bbox[1] + 8  # Satır arası boşluk için +8
                    total_text_height = len(wrapped_lines) * line_height

                    # Metni dikeyde ortala
                    start_y = center[1] - total_text_height // 2

                    for line_idx, line in enumerate(wrapped_lines):
                        # Satır genişliğini hesapla
                        bbox = font.getbbox(line)
                        line_width = bbox[2] - bbox[0]
                        
                        # Satırı yatayda ortala
                        line_x = center[0] - line_width // 2
                        line_y = start_y + (line_idx * line_height)

                        # Metnin resim sınırları içinde kalmasını sağla
                        line_x = max(5, line_x)
                        line_y = max(line_height, line_y)

                        # Metni çiz
                        draw.text((line_x, line_y), line, font=font, fill=TEXT_COLOR)

            # PIL görüntüsünü tekrar OpenCV'ye çevir
            img_with_text = pil_to_cv2(pil_img)

            # Çıktı dosya adını oluştur
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = f"/home/ugo/Documents/manga/{base_name}_metinli_balon_{i}.jpg"
            cv2.imwrite(output_path, img_with_text)
            print(f"Sonuç kaydedildi: {output_path}")

print("\nTüm resimler işlendi!")