from .base import OCRService


class MockOCRService(OCRService):
    """
    Mock OCR service for testing.
    Returns pre-configured text instead of actually performing OCR.
    """

    def __init__(self, mock_text: str):
        """
        Initialize the mock OCR service.

        Args:
            mock_text: The text to return when run_ocr is called
        """
        self.mock_text = mock_text

    async def run_ocr(self, pdf_bytes: bytes) -> str:
        """
        Returns the pre-configured mock text, ignoring the input bytes.

        Args:
            pdf_bytes: The PDF document as bytes (ignored)

        Returns:
            The mock text string
        """
        return self.mock_text
