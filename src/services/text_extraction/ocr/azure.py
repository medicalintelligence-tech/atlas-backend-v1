import logging
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
)
from azure.core.credentials import AzureKeyCredential

from src.services.text_extraction.ocr.base import OCRService
from src.config.settings import settings
from src.utils.text_cleaning import clean_ocr_text

logger = logging.getLogger(__name__)


class AzureOCRService(OCRService):
    """
    Azure Document Intelligence OCR implementation.

    Uses the prebuilt-read model to extract text from documents.
    Returns text in Markdown format for better structure preservation.
    """

    def __init__(
        self,
        endpoint: str = settings.azure_document_intelligence_endpoint,
        api_key: str = settings.azure_document_intelligence_api_key,
    ):
        """
        Initialize Azure OCR service.

        Args:
            endpoint: Azure Document Intelligence endpoint
            api_key: Azure Document Intelligence API key
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.credential = AzureKeyCredential(self.api_key)
        logger.info(f"Initialized AzureOCRService with endpoint: {self.endpoint}")

    async def run_ocr(self, document_bytes: bytes) -> str:
        """
        Extract text from document bytes using Azure Document Intelligence.

        Args:
            document_bytes: Raw bytes of the document to process

        Returns:
            Extracted and cleaned text as a string
        """
        logger.debug(f"Starting OCR for document ({len(document_bytes)} bytes)")

        async with DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=self.credential,
        ) as client:
            # Start analysis with bytes
            poller = await client.begin_analyze_document(
                "prebuilt-read",
                AnalyzeDocumentRequest(bytes_source=document_bytes),
                output_content_format=DocumentContentFormat.MARKDOWN,
            )

            logger.debug("Waiting for OCR to complete...")
            result = await poller.result()

        # Extract text from result
        extracted_text = result.content if result.content else ""
        logger.info(f"OCR completed, extracted {len(extracted_text)} characters")

        # Clean the text
        cleaned_text = clean_ocr_text(extracted_text)
        logger.debug(f"Text cleaned, final length: {len(cleaned_text)} characters")

        return cleaned_text
