import pytest
from src.schemas.text_extraction import TextExtractionResult
from src.services.text_extraction.service import TextExtractionService
from tests.fixtures.text_extraction import DUMMY_PDF_BYTES, EXPECTED_TEXT, CONTENT_TYPE

# from src.services.text_extraction.ocr.azure import AzureOCRService  # TODO: Uncomment when implemented


@pytest.mark.integration
@pytest.mark.skip(reason="Azure OCR service not yet implemented")
async def test_extract_text_from_pdf_with_azure_ocr():
    """
    Integration test for extracting text from a PDF using Azure Document Intelligence OCR.

    This validates:
    - The Azure OCR service can extract text from real PDF bytes
    - The extracted text matches the expected content
    - Metadata is properly calculated from the actual extraction

    TODO: Before running this test, you need to:
    1. Implement AzureOCRService in src/services/text_extraction/ocr/azure.py
    2. Set up environment variables or config for Azure credentials
    3. Remove the @pytest.mark.skip decorator
    4. Uncomment the import at the top
    """
    # TODO: Get Azure credentials from environment variables
    # azure_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    # azure_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    # TODO: Initialize Azure OCR service
    # azure_ocr = AzureOCRService(
    #     endpoint=azure_endpoint,
    #     api_key=azure_key
    # )

    # TODO: Create text extraction service with Azure OCR
    # text_service = TextExtractionService(ocr_service=azure_ocr)

    # Act: Extract text from PDF bytes using real OCR
    # result = await text_service.extract_text(DUMMY_PDF_BYTES, CONTENT_TYPE)

    # Assert: Verify the extracted text
    # NOTE: Real OCR might have slight variations (spacing, casing, etc.)
    # Consider using fuzzy matching or normalizing both strings
    # assert result.text.strip().lower() == EXPECTED_TEXT.strip().lower()
    # assert result.character_length > 0
    # assert result.token_length > 0
    # assert result.duration > 0
    # assert isinstance(result, TextExtractionResult)

    pass


# TODO: Add more integration tests for edge cases:
# - Large PDF documents
# - PDFs with images and complex layouts
# - Scanned documents (actual OCR needed)
# - Multi-page PDFs
# - PDFs with tables
# - Error handling (invalid PDFs, API failures, etc.)
