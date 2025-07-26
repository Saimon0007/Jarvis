"""
OCR (Image-to-Text) skill for Jarvis: Extracts text from images using pytesseract.
Requires the pytesseract and Pillow libraries, and Tesseract installed on the system.
"""

def ocr_skill(user_input, conversation_history=None, search_skill=None):
    """
    Extracts text from an image file.
    Usage: ocr <image_path>
    """
    import os
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return "OCR requires Pillow and pytesseract. Please install them with 'pip install pillow pytesseract'."
    parts = user_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        return "Please provide the path to an image file."
    image_path = parts[1].strip()
    if not os.path.isfile(image_path):
        return f"File not found: {image_path}"
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip() if text.strip() else "No text found in image."
    except Exception as e:
        return f"OCR failed: {e}"

def register(jarvis):
    jarvis.register_skill("ocr", ocr_skill)
