import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap

def process_bubble(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
        image[mask == 255] = (255, 255, 255)
        return image, largest_contour
    else:
        return image, None


def add_text(image, text, font_path, bubble_contour):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = cv2.boundingRect(bubble_contour)

    line_height = 32
    font_size = 30
    wrapping_ratio = 0.065

    wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio), 
                                 break_long_words=True)
    
    font = ImageFont.truetype(font_path, size=font_size)

    lines = wrapped_text.split('\n')
    total_text_height = (len(lines)) * line_height

    while total_text_height > h:
        line_height -= 2
        font_size -= 2
        wrapping_ratio += 0.025

        wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio), 
                                 break_long_words=True)
                                 
        font = ImageFont.truetype(font_path, size=font_size)

        lines = wrapped_text.split('\n')
        total_text_height = (len(lines)) * line_height                         

    text_y = y + (h - total_text_height) // 2

    for line in lines:
        text_length = draw.textlength(line, font=font)

        text_x = x + (w - text_length) // 2

        draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))

        text_y += line_height

    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image
