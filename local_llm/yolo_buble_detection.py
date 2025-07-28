from ultralytics import YOLO
import cv2
import numpy as np
from doctr.models import ocr_predictor
from llm import OllamaChatLLM

OllamaChatLLM = OllamaChatLLM(model="gemma3:12b")

# OCR modelini başlat
model_ocr = ocr_predictor(pretrained=True)

# Model ve resim yolu
model_yolo = YOLO("/home/ugo/Documents/manga/comic-speech-bubble-detector.pt")
img_path = "/home/ugo/Documents/manga/0.jpg"

# Resmi yükle ve kontrol et
img = cv2.imread(img_path)
if img is None:
    print(f"Hata: Resim yüklenemedi - {img_path}")
    print("Dosya yolunu kontrol edin!")
    exit()

# YOLO modelini çalıştır
results = model_yolo(img, save=True, conf=0.25, device="cpu")

print(f"Toplam {len(results)} sonuç bulundu")

# Metin çizim parametrelerini tanımla
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 0) # Metin için siyah renk

# Metni belirli bir genişliğe göre satırlara bölen yardımcı fonksiyon
def wrap_text(text, max_width, font, font_scale, font_thickness):
    words = text.split(' ')
    lines = []
    current_line = []

    for word in words:
        # Mevcut satıra kelime eklenmiş halinin genişliğini test et
        test_line = ' '.join(current_line + [word])
        (width, height), baseline = cv2.getTextSize(test_line, font, font_scale, font_thickness)

        if width <= max_width:
            current_line.append(word)
        else:
            # Eğer kelime mevcut satırı taşırıyorsa, yeni satıra geç
            if current_line: # Önceki satırda bir şeyler varsa ekle
                lines.append(' '.join(current_line))
            current_line = [word] # Yeni satırı mevcut kelimeyle başlat
            # Eğer tek bir kelime bile max_width'i aşıyorsa, onu zorla ekle (çok uzun kelimeler için)
            (width, height), baseline = cv2.getTextSize(word, font, font_scale, font_thickness)
            if width > max_width and not lines: # İlk kelime çok uzunsa ve henüz satır yoksa
                 lines.append(word)
                 current_line = [] # Mevcut satırı sıfırla
            elif width > max_width and lines and not current_line: # Sonraki kelime çok uzunsa
                 lines.append(word)
                 current_line = []


    if current_line:
        lines.append(' '.join(current_line))

    # Satır sayısını 3 ile sınırla
    return lines[:3]


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
                result_ocr = model_ocr([cropped_img])

                for page in result_ocr.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            extracted_text += " ".join([word.value for word in line.words]) + " "
                print(f"Balon {j}'den çıkarılan metin: {extracted_text.strip()}")
            else:
                print(f"Balon {j} için boş veya geçersiz kırpılmış resim atlanıyor")


            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(img_with_text, center, axes, 0, 0, 360, (255, 255, 255), -1)
            
            
            extracted_text=OllamaChatLLM.answer(extracted_text)
            print(f"Balon {j} için LLM cevabı: {extracted_text.strip()}")

            if extracted_text.strip(): # Sadece metin çıkarıldıysa ekle
                # Baloncuğun yaklaşık genişliğini al (metin için kullanılabilecek alan)
                bubble_width = x2 - x1
                # Biraz kenar boşluğu bırakmak için genişliği azalt
                text_max_width = int(bubble_width * 0.9)

                # Metni satırlara böl
                wrapped_lines = wrap_text(extracted_text.strip(), text_max_width, FONT, FONT_SCALE, FONT_THICKNESS)

                # Her satırı baloncuk içine çiz
                line_height = cv2.getTextSize("Test", FONT, FONT_SCALE, FONT_THICKNESS)[0][1] + 5 # Satır arası boşluk için +5
                total_text_height = len(wrapped_lines) * line_height

                # Metni dikeyde ortala
                start_y = center[1] - total_text_height // 2 + line_height // 2

                for line_idx, line in enumerate(wrapped_lines):
                    (line_width, line_height_actual), baseline = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)
                    line_x = center[0] - line_width // 2 # Satırı yatayda ortala
                    line_y = start_y + (line_idx * line_height)

                    # Metnin resim sınırları içinde kalmasını sağla
                    line_x = max(0, line_x)
                    line_y = max(line_height_actual, line_y) # Y'nin negatif olmamasını sağla

                    cv2.putText(img_with_text, line, (line_x, line_y),
                                FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        cv2.imwrite(f"/home/ugo/Documents/manga/0_metinli_balon_{i}.jpg", img_with_text)