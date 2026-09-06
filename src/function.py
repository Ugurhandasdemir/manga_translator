import textwrap
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Balon içi kenar boşluğu (piksel)
PADDING = 8
MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 44
# Balon içi sayılabilmesi için beyaz bölgenin ROI'ye minimum oranı
MIN_INTERIOR_RATIO = 0.10


def process_bubble(image):
    """Konuşma balonunun içini beyaza boyar; (görüntü, kontur, iç maske) döndürür."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, None, None

    h, w = gray.shape
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # ROI merkezini kapsayan en büyük beyaz bölge balon içidir;
    # hiçbiri kapsamıyorsa en büyüğüne düş (merkez metnin üstüne denk gelebilir)
    chosen = None
    for c in contours[:5]:
        if cv2.pointPolygonTest(c, (w / 2.0, h / 2.0), False) >= 0:
            chosen = c
            break
    if chosen is None:
        chosen = contours[0]

    if cv2.contourArea(chosen) < MIN_INTERIOR_RATIO * h * w:
        return image, None, None

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [chosen], -1, 255, cv2.FILLED)
    image[mask == 255] = (255, 255, 255)
    return image, chosen, mask


def _row_spans(mask) -> Dict[int, Tuple[int, int]]:
    """Maskede her satır için kullanılabilir (sol, sağ) x aralığı."""
    spans: Dict[int, Tuple[int, int]] = {}
    for y in np.flatnonzero(mask.any(axis=1)):
        xs = np.flatnonzero(mask[y])
        spans[int(y)] = (int(xs[0]), int(xs[-1]))
    return spans


def _split_word(draw, word, font, max_width):
    """Satıra sığmayan kelimeyi 'önek-' + kalan olarak böler."""
    for cut in range(len(word) - 1, 1, -1):
        head = word[:cut] + "-"
        if draw.textlength(head, font=font) <= max_width:
            return head, word[cut:]
    return None, None


def _greedy_wrap(draw, words, font, widths) -> Optional[List[str]]:
    """Kelimeleri verilen satır genişliklerine sırayla doldurur; sığmazsa None."""
    words = list(words)
    lines: List[str] = []
    i = 0
    for max_w in widths:
        if i >= len(words):
            break
        cur = ""
        while i < len(words):
            cand = words[i] if not cur else cur + " " + words[i]
            if draw.textlength(cand, font=font) <= max_w:
                cur = cand
                i += 1
            else:
                if not cur:
                    head, tail = _split_word(draw, words[i], font, max_w)
                    if head is None:
                        return None
                    cur = head
                    words[i] = tail
                break
        lines.append(cur)
    if i < len(words):
        return None
    return lines


def _layout_text(draw, text, font, line_height, spans, inner_pad=2):
    """Metni balonun gerçek şekline yerleştirmeyi dener.

    Satır sayısını 1'den başlayarak artırır; her satırın genişliği o satırın
    denk geldiği maske satırlarından alınır (elips balonda üst/alt dar olur).
    Dönen değer: [(y, sol, sağ, satır_metni), ...] ya da None.
    """
    if not spans:
        return None
    ys = sorted(spans)
    y_top, y_bot = ys[0], ys[-1]
    max_lines = (y_bot - y_top + 1) // line_height
    if max_lines < 1:
        return None

    base_words = text.split()
    cy = (y_top + y_bot) / 2.0

    for k in range(1, max_lines + 1):
        block_h = k * line_height
        y0 = int(round(cy - block_h / 2.0))
        y0 = max(y_top, min(y0, y_bot - block_h + 1))

        boxes = []  # (y, sol, sağ)
        rows_ok = True
        for li in range(k):
            ya = y0 + li * line_height
            seg = [spans.get(y) for y in range(ya, ya + line_height)]
            if any(s is None for s in seg):
                rows_ok = False
                break
            left = max(s[0] for s in seg) + inner_pad
            right = min(s[1] for s in seg) - inner_pad
            if right - left < line_height:
                rows_ok = False
                break
            boxes.append((ya, left, right))
        if not rows_ok:
            continue

        lines = _greedy_wrap(draw, base_words, font, [r - l for (_, l, r) in boxes])
        if lines is None:
            continue
        return [(boxes[i][0], boxes[i][1], boxes[i][2], lines[i]) for i in range(len(lines))]
    return None


def add_text(image, text, font_path, bubble_contour, mask=None):
    """Çevrilmiş metni balonun şekline göre ortalayarak yerleştirir."""
    text = " ".join(str(text).split())
    if not text:
        return image

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    if mask is None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [bubble_contour], -1, 255, cv2.FILLED)

    # Balon kenarından PADDING kadar içeri çekil
    k = 2 * PADDING + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask, kernel)
    if not eroded.any():
        eroded = mask
    spans = _row_spans(eroded)

    # Sığan en büyük puntoyu bul (punto büyüdükçe sığma tekdüze bozulur)
    best = None
    low, high = MIN_FONT_SIZE, MAX_FONT_SIZE
    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, size=mid)
        ascent, descent = font.getmetrics()
        layout = _layout_text(draw, text, font, ascent + descent, spans)
        if layout is not None:
            best = (font, layout)
            low = mid + 1
        else:
            high = mid - 1

    if best is not None:
        font, layout = best
        for ly, left, right, line in layout:
            if not line:
                continue
            lw = draw.textlength(line, font=font)
            lx = left + (right - left - lw) // 2
            draw.text((lx, ly), line, font=font, fill=(0, 0, 0))
    else:
        # Hiçbir punto şekle sığmadı: bounding rect içine en küçük puntoyla bas
        font = ImageFont.truetype(font_path, size=MIN_FONT_SIZE)
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        x, y, w, h = cv2.boundingRect(bubble_contour)
        inner_w = max(1, w - 2 * PADDING)
        avg = max(1.0, font.getlength("A"))
        lines = textwrap.fill(text, width=max(1, int(inner_w / avg)),
                              break_long_words=True).split("\n")
        ty = y + PADDING + max(0, (h - 2 * PADDING - len(lines) * line_height) // 2)
        for line in lines:
            lw = draw.textlength(line, font=font)
            tx = x + PADDING + max(0, (inner_w - lw) // 2)
            draw.text((tx, ty), line, font=font, fill=(0, 0, 0))
            ty += line_height

    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image
