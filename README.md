# Manga Translator

Bu proje, manga panelindeki konuşma balonlarından OCR (Optical Character Recognition) kullanarak metin çıkarır, LLM veya API tabanlı çeviri ile metni İngilizceden Türkçeye çevirir ve çeviri metnini görüntü üzerinde yeniden yerleştirir.

## Özellikler

- **Balon Tespiti:** [YOLO](https://github.com/ultralytics/ultralytics) ile konuşma balonlarını otomatik olarak algılar.
- **OCR İşlemi:** [easyocr](https://github.com/JaidedAI/EasyOCR) ile balon içindeki metni çıkarır.
- **Metin Çevirisi:** [Ollama](https://ollama.com/) ile local LLM, [Google Gemini](https://ai.google.dev/) API veya OpenAI GPT ile İngilizceden Türkçeye çeviri.
- **Görüntü İşleme:** [OpenCV](https://opencv.org/) ile balon içine metni yeniden yerleştirir. Özel fontlar için [Pillow (PIL)](https://pillow.readthedocs.io/) kullanılabilir.

## Gereksinimler

- Python 3.x
- ultralytics (YOLO)
- easyocr
- deep-translator
- opencv-python
- pillow
- numpy
- langchain, langchain_ollama
- ollama (local LLM için)
- google-generativeai (Google Gemini için)
- openai (OpenAI GPT için)
- python-dotenv (API anahtarları için)

## Kurulum

1. Gerekli Python paketlerini yükleyin:
    ```sh
    pip install ultralytics easyocr deep-translator opencv-python pillow numpy langchain langchain_ollama google-generativeai openai python-dotenv
    ```
2. Local LLM için [Ollama](https://ollama.com/download) kurun ve bir model indirin:
    ```sh
    ollama pull llama3.1:8b
    ```
3. Google Gemini ve OpenAI API anahtarlarınızı `.env` dosyasına ekleyin:
    ```
    gemini_api_key=YOUR_GEMINI_API_KEY
    openai_api_key=YOUR_OPENAI_API_KEY
    ```
4. Türkçe karakter destekli bir font dosyasını (örn. ComicRelief.ttf) `FONT_PATH` olarak belirtin.

## Kullanım

`src/main.py` dosyasını çalıştırarak:

- YOLO ile konuşma balonları tespit edilir,
- Her balon için OCR ile metin çıkarılır,
- Çıkarılan metin LLM veya API ile Türkçeye çevrilir,
- Çeviri sonucu balonun içine yeniden yazılır ve yeni görsel kaydedilir.



## Örnek Sonuç

![Örnek Manga Balonu](https://github.com/user-attachments/assets/b88a0280-3e18-4de7-a7ad-aec5b63ec52d)

## Yapılacaklar (TODO)

- [ ] Balon dışı metinler için de OCR ve çeviri desteği ekle.
- [ ] Daha iyi satır kaydırma ve font boyutu otomasyonu.
- [ ] GUI arayüzü ekle.
- [ ] Qwen api ekle.
