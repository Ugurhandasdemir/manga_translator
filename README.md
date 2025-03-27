# Manga Translator

Bu proje, manga panelindeki konuşma balonlarından OCR (Optical Character Recognition) kullanarak metin çıkarır, Google Translate kullanarak metni İngilizceden Türkçeye çevirir ve çeviri metnini görüntü üzerinde yeniden yerleştirir.

## Özellikler

- **OCR İşlemi:** [Tesseract](https://github.com/tesseract-ocr/tesseract) kullanılarak görüntüden metin çıkarılır.
- **Metin Çevirisi:** [Googletrans API](https://py-googletrans.readthedocs.io/) aracılığıyla çıkarılan metin Türkçeye çevrilir.
- **Görüntü İşleme:** [OpenCV](https://opencv.org/) ve [PIL](https://pillow.readthedocs.io/) kullanılarak metin, resim üzerinde beyaz kutular içerisine yeniden çizilir.

## Gereksinimler

- Python 3.x
- Tesseract OCR (Kurulum ve `tesseract.exe` yolu doğru olarak ayarlanmalıdır)
- OpenCV
- Pillow
- pytesseract
- pandas
- googletrans
- numpy

## Kurulum

1. Gerekli Python paketlerini yükleyin:
    ```sh
    pip install opencv-python pillow pytesseract pandas googletrans numpy
    ```
2. Tesseract OCR'ı [indirip](https://github.com/tesseract-ocr/tesseract) kurduğunuzdan emin olun ve `main.py` içerisindeki `tesseract_cmd` yolunu kendi sisteminize göre ayarlayın.

## Kullanım

`main.py` içerisindeki kod, belirtilen klasördeki (ör. `a25705d4-4dab-47ef-b363-635ca0081ea6`) tüm resimleri işler. Her resim için:

- OCR ile metin çıkarılır,
- Çıkarılan metin Türkçeye çevrilir,
- Çeviri sonucu orijinal metin kutusunun üzerine yeniden basılır ve
- Sonuçlar `test_sonuc2` klasörüne kaydedilir.

Projeyi çalıştırmak için:
```sh
python [main.py](http://_vscodecontentref_/0)
