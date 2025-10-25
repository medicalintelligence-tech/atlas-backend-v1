from src.schemas.text_extraction import TextExtractionResult
from src.services.text_extraction.service import TextExtractionService
from src.services.text_extraction.ocr.mock import MockOCRService
from tests.fixtures.text_extraction import DUMMY_PDF_BYTES, EXPECTED_TEXT, CONTENT_TYPE


async def test_extract_text_from_pdf():
    """
    Test that the TextExtractionService can extract text from a PDF using MockOCRService.

    This validates:
    - The service correctly routes PDF extraction to the OCR service
    - The result contains the expected text
    - Metadata (character_length, token_length, duration) is properly calculated
    """
    # Arrange: Create mock OCR service with expected text
    mock_ocr = MockOCRService(mock_text=EXPECTED_TEXT)
    text_service = TextExtractionService(ocr_service=mock_ocr)

    # Act: Extract text from PDF bytes
    result = await text_service.extract_text(DUMMY_PDF_BYTES, CONTENT_TYPE)

    # Assert: Verify result structure and content
    assert isinstance(result, TextExtractionResult)
    assert result.text == EXPECTED_TEXT
    assert result.character_length == len(EXPECTED_TEXT)
    assert result.character_length == 149
    assert result.token_length == 26  # Tiktoken counts 26 tokens (25 words + period)
    assert result.duration >= 0  # Duration should be measured
