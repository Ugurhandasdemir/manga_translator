import subprocess


def recognize_text(image_path: str) -> str:
    """Run GLM-OCR via the Ollama CLI and return the recognized text."""
    result = subprocess.run(
        ["ollama", "run", "glm-ocr", "Text Recognition:", image_path],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GLM-OCR on a local image and print the extracted text."
    )
    parser.add_argument("image", help="Path to the image file to process.")
    args = parser.parse_args()

    extracted = recognize_text(args.image)
    print(extracted)
