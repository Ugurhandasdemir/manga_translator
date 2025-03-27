import os
from PIL import Image, ImageFont, ImageDraw
import pytesseract
import cv2
import pandas as pd
from googletrans import Translator
import numpy as np

translator = Translator()

# Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img_folder = r"C:\Users\ugurh\Documents\Python\Manga_web\a25705d4-4dab-47ef-b363-635ca0081ea6"
i = 0

# Döngüde dizindeki dosyaları kullanmak için
for file_name in os.listdir(img_folder):
    img_path = os.path.join(img_folder, file_name)
    i += 1

    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image {img_path} not found or could not be opened.")
        continue  # Dosya okunamadığında çıkmak yerine sonraki dosyaya geçiyoruz

    # Get detailed data using pytesseract
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    # Helper function: wrap text within a given width
    def wrap_text(text, font, font_scale, font_thickness, max_width):
        words = text.split()
        if not words:
            return []
        lines = []
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + " " + word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines

    # Process data from pytesseract
    words = data["text"]
    x_coords = data["left"]
    y_coords = data["top"]
    widths = data["width"]
    heights = data["height"]
    line_numbers = data["line_num"]

    # Threshold for detecting spaces between words
    space_threshold = 20  

    # (Optional) Print coordinates for word pairs with larger spaces
    for j in range(len(words) - 1):
        if words[j].strip() == "" or words[j + 1].strip() == "":
            continue
        if line_numbers[j] == line_numbers[j + 1]:
            space_between = x_coords[j + 1] - (x_coords[j] + widths[j])
            if space_between > space_threshold:
                print(f"Coordinates: ({x_coords[j]}, {y_coords[j]}) -> ({x_coords[j+1]}, {y_coords[j+1]})\n")

    # Group words into blocks (lines)
    blocks = []
    current_block = []
    current_line = None
    for j in range(len(words)):
        if words[j].strip() == "":
            continue
        # Group words that are on the same line (with 100px tolerance)
        if current_line is None or abs(y_coords[j] - current_line) <= 100:
            current_block.append((words[j], x_coords[j], y_coords[j], widths[j], heights[j]))
            current_line = y_coords[j]
        else:
            if current_block:
                blocks.append(current_block)
            current_block = [(words[j], x_coords[j], y_coords[j], widths[j], heights[j])]
            current_line = y_coords[j]
    if current_block:
        blocks.append(current_block)

    # Determine bounding coordinates for each block
    block_coordinates = []
    for block in blocks:
        if not block:
            continue
        min_x = min(word[1] for word in block)
        min_y = min(word[2] for word in block)
        max_x = max(word[1] + word[3] for word in block)
        max_y = max(word[2] + word[4] for word in block)
        text_block = " ".join(word[0] for word in block)
        block_coordinates.append((text_block, (min_x, min_y, max_x, max_y)))

    # Process each block: translate the text and write it inside a white rectangle
    for text, (x1, y1, x2, y2) in block_coordinates:
        print(f"Text Block: {text}")
        # Draw a white rectangle on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        # Translate the text with error handling
        try:
            translation = translator.translate(text, src='en', dest='tr')
            result = translation.text
        except Exception as e:
            print(f"Translation error for block '{text}': {str(e)}")
            result = text  # Fallback: orijinal metni kullan
        print(result)
        
        # Settings for drawing text inside the rectangle (PIL kullanılarak çizilecek)
        # Calculate available width and height within the rectangle (5px margin)
        available_width = x2 - x1
        available_height = y2 - y1
        max_text_width = available_width - 10
        
        # Wrap the translated text as needed (OpenCV metrikleri kullanılarak sarmalama)
        font_cv = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.75
        font_thickness = 1
        all_lines = []
        for line in result.splitlines():
            wrapped = wrap_text(line, font_cv, font_scale, font_thickness, max_text_width)
            if not wrapped:
                all_lines.append("")
            else:
                all_lines.extend(wrapped)
        
        # PIL ile metin çizdirme işlemi için OpenCV görüntüsünü PIL'e çeviriyoruz
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # Türkçe karakter desteği veren TrueType fontu kullanın (örn. Arial)
        font_path = r"C:\Windows\Fonts\arial.ttf"
        font_size = 35
        font_pil = ImageFont.truetype(font_path, font_size)
        
        # Yardımcı fonksiyon: metin boyutunu getbbox ile hesaplama
        def get_text_size(text, font):
            bbox = font.getbbox(text)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width, height

        # Calculate total height of text block using get_text_size
        total_text_height = 0
        for line in all_lines:
            (text_width, text_height) = get_text_size(line, font_pil)
            total_text_height += text_height
        total_text_height += (len(all_lines) - 1) * 5  # 5px gap between lines
        
        # Calculate vertical starting point to vertically center the text in the rectangle
        y_text = y1 + (available_height - total_text_height) // 2
        
        # Write each line, centered horizontally using PIL and get_text_size
        for line in all_lines:
            (text_width, text_height) = get_text_size(line, font_pil)
            x_line = x1 + (available_width - text_width) // 2
            draw.text((x_line, y_text), line, font=font_pil, fill=(0, 0, 0))
            y_text += text_height + 5  # 5px vertical gap
        
        # Convert PIL image back to OpenCV format
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Dosya yolunu oluştururken, i değerini kullanıyoruz
    output_path = os.path.join(r"C:\Users\ugurh\Desktop\test_sonuc2", f"{i}.jpg")
    cv2.imwrite(output_path, img)
    print(f"Image saved at '{output_path}'.")