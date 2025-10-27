from src.schemas.text_extraction import TextExtractionResult
from src.services.text_extraction.service import TextExtractionService
from tests.fixtures.text_extraction import (
    DUMMY_PLAIN_TEXT_BYTES,
    PLAIN_TEXT_EXPECTED_TEXT,
    PLAIN_TEXT_CONTENT_TYPE,
)


async def test_extract_text_from_plain_text():
    """
    Test that the TextExtractionService can extract text from plain text documents.

    This validates:
    - The service correctly handles plain text (text/plain) content type
    - The text is properly decoded from UTF-8 bytes
    - The result contains the exact expected text
    - Metadata (character_length, token_length, duration) is properly calculated
    """
    # Arrange: Create text extraction service (no OCR needed for plain text)
    text_service = TextExtractionService()

    # Act: Extract text from plain text bytes
    result = await text_service.extract_text(
        DUMMY_PLAIN_TEXT_BYTES, PLAIN_TEXT_CONTENT_TYPE
    )

    # Assert: Verify result structure and content
    assert isinstance(result, TextExtractionResult)
    assert result.text == PLAIN_TEXT_EXPECTED_TEXT
    assert result.character_length == len(PLAIN_TEXT_EXPECTED_TEXT)
    assert result.character_length == 156
    assert result.token_length > 0  # Should have tokens
    assert result.duration >= 0  # Duration should be measured
