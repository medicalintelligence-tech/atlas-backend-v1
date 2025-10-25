import re
import codecs
import unicodedata
import ftfy


def clean_ocr_text(text: str) -> str:
    """
    Clean OCR text to fix encoding issues and problematic characters.

    This function handles common OCR artifacts:
    - JSON-escaped characters
    - Unicode escape sequences
    - Encoding issues (using ftfy)
    - Control characters
    - Excessive newlines
    - Extra whitespace
    - Repeated special characters

    Args:
        text: Raw OCR text to clean

    Returns:
        Cleaned text with normalized encoding and whitespace
    """
    # Handle JSON-escaped characters
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    text = text.replace("\\r", "\r")

    # Handle Unicode escape sequences
    try:
        if "\\u" in text:
            text = codecs.decode(text, "unicode_escape")
    except Exception:
        pass

    # Fix encoding issues
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters
    text = re.sub(r"[\x00\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # AGGRESSIVE: Replace ALL newlines with spaces
    text = text.replace("\n", " ")

    # Clean up extra spaces
    text = re.sub(r" +", " ", text)

    # Clean up repeated special characters
    text = re.sub(r"([^\w\s])\1{5,}", r"\1", text)

    return text.strip()
