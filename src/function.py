import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Balon içi kenar boşluğu (piksel)
PADDING = 8
MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 32


def process_bubble(image):
    """Konuşma balonunun içini beyaza boyar ve kontur döndürür."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, None

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
    image[mask == 255] = (255, 255, 255)
    return image, largest_contour


def _fit_text(draw, text, font_path, max_width, max_height):
    """Font boyutunu ve satır sarmalamayı balonun içine sığacak şekilde hesaplar.
    
    Binary search ile en büyük sığan font boyutunu bulur.
    """
    best_font = None
    best_lines = []
    best_line_height = 0

    low, high = MIN_FONT_SIZE, MAX_FONT_SIZE

    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, size=mid)
        ascent, descent = font.getmetrics()
        line_height = ascent + descent

        # Bir karakter genişliğinden sarma genişliği hesapla
        avg_char_w = font.getlength("A")
        if avg_char_w <= 0:
            avg_char_w = mid * 0.6
        wrap_width = max(1, int(max_width / avg_char_w))

        wrapped = textwrap.fill(text, width=wrap_width, break_long_words=True)
        lines = wrapped.split("\n")

        total_h = len(lines) * line_height
        # En uzun satır genişliğini kontrol et
        max_line_w = max(draw.textlength(line, font=font) for line in lines)

        if total_h <= max_height and max_line_w <= max_width:
            best_font = font
            best_lines = lines
            best_line_height = line_height
            low = mid + 1  # Daha büyük dene
        else:
            high = mid - 1  # Daha küçük dene

    return best_font, best_lines, best_line_height


def add_text(image, text, font_path, bubble_contour):
    """Çevrilmiş metni balonun tam ortasına yerleştirir."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = cv2.boundingRect(bubble_contour)

    # Padding uygula — metin balonun kenarlarına yapışmasın
    inner_x = x + PADDING
    inner_y = y + PADDING
    inner_w = max(1, w - 2 * PADDING)
    inner_h = max(1, h - 2 * PADDING)

    font, lines, line_height = _fit_text(draw, text, font_path, inner_w, inner_h)

    if font is None or not lines:
        # Hiçbir boyut sığmadıysa fallback
        font = ImageFont.truetype(font_path, size=MIN_FONT_SIZE)
        lines = [text]
        _, descent = font.getmetrics()
        line_height = MIN_FONT_SIZE + descent

    total_text_height = len(lines) * line_height

    # Dikey ortalama
    text_y = inner_y + (inner_h - total_text_height) // 2

    for line in lines:
        line_w = draw.textlength(line, font=font)
        # Yatay ortalama
        text_x = inner_x + (inner_w - line_w) // 2
        draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))
        text_y += line_height

    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image
