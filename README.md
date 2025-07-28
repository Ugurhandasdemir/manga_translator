# Manga Translator

Bu proje, manga panelindeki konuşma balonlarından OCR (Optical Character Recognition) kullanarak metin çıkarır, LLM veya API tabanlı çeviri ile metni İngilizceden Türkçeye çevirir ve çeviri metnini görüntü üzerinde yeniden yerleştirir.

## Özellikler

- **Balon Tespiti:** [YOLO](https://github.com/ultralytics/ultralytics) ile konuşma balonlarını otomatik olarak algılar.
- **OCR İşlemi:** [doctr](https://github.com/mindee/doctr) ile balon içindeki metni çıkarır.
- **Metin Çevirisi:** [Ollama](https://ollama.com/) ile local LLM veya [Google Gemini](https://ai.google.dev/) API ile İngilizceden Türkçeye çeviri.
- **Görüntü İşleme:** [OpenCV](https://opencv.org/) ile balon içine metni yeniden yerleştirir. Özel fontlar için [Pillow (PIL)](https://pillow.readthedocs.io/) kullanılabilir.

## Gereksinimler

- Python 3.x
- ultralytics (YOLO)
- doctr
- OpenCV
- Pillow
- numpy
- Ollama (local LLM için)
- Google Gemini API (isteğe bağlı)
- (İsteğe bağlı) Tesseract OCR ve googletrans (alternatif eski yöntem için)

## Kurulum

1. Gerekli Python paketlerini yükleyin:
    ```sh
    pip install ultralytics doctr opencv-python pillow numpy
    ```
2. Local LLM için [Ollama](https://ollama.com/download) kurun ve bir model indirin:
    ```sh
    ollama pull gemma:12b
    ```
3. Google Gemini API kullanacaksanız, [API anahtarınızı](https://ai.google.dev/) alın ve kodda ayarlayın.

## Kullanım

`yolo_buble_detection.py` dosyasını çalıştırarak:

- YOLO ile konuşma balonları tespit edilir,
- Her balon için OCR ile metin çıkarılır,
- Çıkarılan metin LLM veya API ile Türkçeye çevrilir,
- Çeviri sonucu balonun içine yeniden yazılır ve yeni görsel kaydedilir.

### Örnek Çalışma Akışı

1. Görüntü dosyasını `img_path` ile belirtin.
2. Scripti çalıştırın:
    ```sh
    python local_llm/yolo_buble_detection.py
    ```
3. Sonuçlar `0_metinli_balon_X.jpg` olarak kaydedilir.

## Örnek Sonuç

![Örnek Manga Balonu](https://github.com/user-attachments/assets/b88a0280-3e18-4de7-a7ad-aec5b63ec52d)

## Notlar

- OCR ve çeviri kalitesi kullanılan modele göre değişebilir.
- Özel font ile Türkçe karakter desteği için Pillow kullanabilirsiniz.
- Eski yöntem olarak Tesseract ve googletrans da kullanılabilir.
