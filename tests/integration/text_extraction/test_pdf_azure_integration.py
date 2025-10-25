import pytest
from src.schemas.text_extraction import TextExtractionResult
from src.services.text_extraction.service import TextExtractionService
from tests.fixtures.text_extraction import (
    AZURE_TEST_PDF_BYTES,
    AZURE_TEST_TEXT,
    CONTENT_TYPE,
)
from src.config.settings import settings
from src.services.text_extraction.ocr.azure import AzureOCRService


@pytest.mark.integration
async def test_extract_text_from_pdf_with_azure_ocr():
    """
    Integration test for extracting text from a PDF using Azure Document Intelligence OCR.

    This validates:
    - The Azure OCR service can extract text from real PDF bytes
    - The extracted text matches the expected content (25 words)
    - Character length matches the expected text length
    - Token length is properly calculated
    - Duration is measured and positive
    """
    # Arrange: Initialize Azure OCR service
    azure_ocr = AzureOCRService()

    # Create text extraction service with Azure OCR
    text_service = TextExtractionService(ocr_service=azure_ocr)

    # Act: Extract text from PDF bytes using real OCR
    result = await text_service.extract_text(AZURE_TEST_PDF_BYTES, CONTENT_TYPE)

    # Assert: Verify result structure and content
    assert isinstance(result, TextExtractionResult)

    # The text should match expected text after cleaning
    assert result.text is not None

    # Validate text content matches exactly
    assert result.text == AZURE_TEST_TEXT

    # Validate character length matches
    assert result.character_length == len(AZURE_TEST_TEXT)
    assert result.character_length == 192  # Exact character count

    # Validate token length is calculated (should be ~26 tokens for 25 words)
    assert result.token_length > 0
    assert result.token_length >= 25  # At least 25 tokens for 25 words

    # Validate duration is measured
    assert result.duration > 0
