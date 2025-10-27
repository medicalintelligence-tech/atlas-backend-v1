from src.schemas.text_extraction import TextExtractionResult
from src.services.text_extraction.service import TextExtractionService
from tests.fixtures.text_extraction import (
    DUMMY_HTML_BYTES,
    HTML_EXPECTED_TEXT,
    HTML_CONTENT_TYPE,
)


async def test_extract_text_from_html():
    """
    Test that the TextExtractionService can extract text from HTML using BeautifulSoup.

    This validates:
    - The service correctly extracts text from HTML documents
    - Script and style tags are properly excluded from the text
    - The result contains the expected text with whitespace normalized
    - Metadata (character_length, token_length, duration) is properly calculated
    """
    # Arrange: Create text extraction service (no OCR needed for HTML)
    text_service = TextExtractionService()

    # Act: Extract text from HTML bytes
    result = await text_service.extract_text(DUMMY_HTML_BYTES, HTML_CONTENT_TYPE)

    # Assert: Verify result structure and content
    assert isinstance(result, TextExtractionResult)
    assert result.text == HTML_EXPECTED_TEXT
    assert result.character_length == len(HTML_EXPECTED_TEXT)
    assert result.character_length == 186
    assert result.token_length > 0  # Should have tokens
    assert result.duration >= 0  # Duration should be measured
    assert "console.log" not in result.text  # Script content should be excluded
