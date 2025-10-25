from abc import ABC, abstractmethod


class OCRService(ABC):
    """
    Abstract base class for OCR services.
    All OCR implementations must inherit from this class and implement run_ocr.
    """

    @abstractmethod
    def run_ocr(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes using OCR.

        Args:
            pdf_bytes: The PDF document as bytes

        Returns:
            The extracted text as a string
        """
        pass
