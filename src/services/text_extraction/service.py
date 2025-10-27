import re
import time
from typing import Optional

from bs4 import BeautifulSoup

from src.schemas.text_extraction import TextExtractionResult
from src.utils.tokens import count_tokens
from .ocr.base import OCRService


class UnsupportedContentTypeError(Exception):
    """Raised when an unsupported content type is provided."""

    pass


class OCRServiceRequiredError(Exception):
    """Raised when OCR service is required but not provided."""

    pass


class TextExtractionService:
    """
    Service for extracting text from various document types.
    Orchestrates extraction and calculates metadata (character count, token count, duration).
    """

    def __init__(self, ocr_service: Optional[OCRService] = None):
        """
        Initialize the text extraction service.

        Args:
            ocr_service: Optional OCR service for extracting text from PDFs
        """
        self.ocr_service = ocr_service

    async def extract_text(
        self, document_bytes: bytes, content_type: str
    ) -> TextExtractionResult:
        """
        Extract text from a document and return result with metadata.

        Args:
            document_bytes: The document as bytes
            content_type: The MIME type of the document

        Returns:
            TextExtractionResult containing text and metadata

        Raises:
            UnsupportedContentTypeError: If the content type is not supported
        """
        start_time = time.time()

        # Route to appropriate extraction method
        text = await self._route_extraction(document_bytes, content_type)

        # Calculate metadata
        duration = time.time() - start_time  # Duration in seconds
        char_length = len(text)
        token_length = count_tokens(text)

        return TextExtractionResult(
            text=text,
            character_length=char_length,
            token_length=token_length,
            duration=duration,
        )

    async def _route_extraction(self, document_bytes: bytes, content_type: str) -> str:
        """
        Route to the appropriate extraction method based on content type.

        Args:
            document_bytes: The document as bytes
            content_type: The MIME type of the document

        Returns:
            The extracted text

        Raises:
            UnsupportedContentTypeError: If the content type is not supported
        """
        if content_type == "application/pdf":
            return await self._extract_from_pdf(document_bytes)
        elif content_type == "text/plain":
            return self._extract_from_txt(document_bytes)
        elif content_type == "text/html":
            return self._extract_from_html(document_bytes)
        else:
            raise UnsupportedContentTypeError(
                f"Content type '{content_type}' is not supported"
            )

    async def _extract_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract text from a PDF document using OCR.

        Args:
            pdf_bytes: The PDF document as bytes

        Returns:
            The extracted text

        Raises:
            OCRServiceRequiredError: If no OCR service was provided
        """
        if self.ocr_service is None:
            raise OCRServiceRequiredError(
                "OCR service is required to extract text from PDFs"
            )

        return await self.ocr_service.run_ocr(pdf_bytes)

    def _extract_from_txt(self, txt_bytes: bytes) -> str:
        """
        Extract text from a plain text document.

        Args:
            txt_bytes: The text document as bytes

        Returns:
            The decoded text
        """
        return txt_bytes.decode("utf-8")

    def _extract_from_html(self, html_bytes: bytes) -> str:
        """
        Extract text from an HTML document using BeautifulSoup.

        Args:
            html_bytes: The HTML document as bytes

        Returns:
            The extracted and cleaned text
        """
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_bytes, "html.parser")

        # Extract text, removing extra whitespace
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()

        return text
