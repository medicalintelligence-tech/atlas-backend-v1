# Text Extraction Service Foundation

**Date**: 2025-10-25

## Starting State

- Fresh backend project with basic structure (`main.py`, `pyproject.toml`)
- Single test file (`tests/text_extraction/test_pdf.py`) with:
  - A Pydantic model (`TextExtractionResult`)
  - Dummy PDF bytes for testing
  - Lots of comments/notes about how to approach the problem
- Goal: Build a text extraction service that could handle PDFs using OCR, with the ability to swap OCR providers

## Discussion & Evolution

### Initial Architecture Question
**Question**: How to design an OCR service that's mockable for testing but swappable for real implementations?

**Options Considered**:
1. Simple duck typing (no formal interface)
2. Protocol-based (typing.Protocol) for structural subtyping
3. Abstract Base Class (ABC) with explicit inheritance

**Decision**: ABC approach - balances simplicity with explicitness, provides type safety and IDE support, aligns with "Zen of Python" while being practical.

### Layered Architecture Realization
Through discussion, realized we needed TWO layers:
- **Layer 1 (Orchestrator)**: `TextExtractionService` - routes by content type, calculates metadata
- **Layer 2 (Strategy)**: `OCRService` - just extracts text from bytes

**Key insight**: OCR service should return `str`, not `TextExtractionResult`. The orchestrator handles common metadata (duration, char count, token count) for ALL document types, avoiding duplication.

### Project Structure Evolution

**Initial thought**: Keep everything flat or use generic names like "atlas"

**Iterations**:
- Considered `atlas/` but felt redundant since already in `atlas_v1/backend`
- Briefly considered `app/` but that felt like FastAPI-specific
- **Landed on**: `src/` - clean, conventional, works well with tests as sibling

**Final structure**:
```
src/
  schemas/         # Pydantic models (DB models will go in models/)
  services/        # Business logic
  utils/           # Reusable functions

tests/
  fixtures/        # Shared test data
  unit/           # Fast, mocked tests  
  integration/    # Real API tests (run separately)
```

### Testing Organization
**Question**: How to organize unit vs integration tests?

**Decision**: Separate folders (`tests/unit/`, `tests/integration/`) rather than naming conventions
- More explicit and scalable
- Easy to run selectively: `pytest tests/unit`
- Common pattern in well-organized projects

**Fixture sharing**: Created `tests/fixtures/` to avoid duplicating test data (dummy PDF bytes) across test files.

### Async from the Start
**Decision**: Make everything async immediately, even though mock doesn't need it
- Future-proof for real API calls
- Easier to start async than retrofit later
- Configured `pytest-asyncio` with `asyncio_mode = "auto"` so tests don't need decorators

### Metadata Calculations
**Evolution**:
- Initially thought OCR service would return full `TextExtractionResult`
- Realized token counting and duration measurement are the SAME for all document types
- **Moved metadata logic to orchestrator** - single responsibility, DRY principle
- Created `count_tokens()` utility using tiktoken - reusable anywhere

### Details That Mattered
- **Seconds vs milliseconds**: Switched to seconds for duration (simpler, more standard)
- **Test data as f-string**: Put `EXPECTED_TEXT` at top, inject into PDF bytes - single source of truth
- **Token count reality**: Expected 25 words but tiktoken returned 26 tokens (period is separate)

## Final Solution

### Architecture
```
TextExtractionService (orchestrator)
  ├─ extract_text() - public API, handles metadata
  ├─ _route_extraction() - dispatches by content type
  ├─ _extract_from_pdf() - uses OCR service
  └─ _extract_from_txt() - simple decode

OCRService (ABC)
  ├─ MockOCRService - returns pre-configured text
  └─ AzureOCRService - skeleton with comprehensive TODOs
```

### Key Files Created
```
src/
  schemas/text_extraction.py
  services/text_extraction/
    service.py
    ocr/
      base.py
      mock.py
      azure.py (skeleton)
  utils/tokens.py

tests/
  fixtures/text_extraction.py
  unit/text_extraction/test_pdf.py
  integration/text_extraction/test_pdf_azure_integration.py
```

### Design Patterns
- **Strategy Pattern**: OCR services are interchangeable strategies
- **Dependency Injection**: OCR service passed to constructor
- **Single Responsibility**: Each component has one job
- **DRY**: Shared test fixtures, common metadata calculation

## Characteristics

### Why This Works
- **Clean separation**: Orchestration vs extraction strategies
- **Easy to test**: Mock returns string, no API needed
- **Easy to extend**: Add new document types or OCR providers independently
- **Type safe**: ABC enforces contract at instantiation
- **Async-ready**: All services use async/await for future API calls
- **Conventional**: Standard Python patterns, clear structure

### Project Values Applied
- **Zen of Python**: Simple, explicit, not clever
- **Start organized**: Structure scales as features are added
- **Document decisions**: TODOs in Azure skeleton explain what to do
- **Test-first mindset**: Unit tests with mocks, integration tests for real usage

## Commands

```bash
# Run fast unit tests
pytest tests/unit -v

# Run integration tests (when Azure implemented)
pytest tests/integration -v

# Run all tests
pytest -v

# Skip integration tests
pytest -m "not integration"
```

## Future Considerations

### When to Add Azure OCR
The skeleton is ready in `src/services/text_extraction/ocr/azure.py` with comprehensive TODOs:
1. Add Azure SDK dependency
2. Implement client initialization
3. Implement async `run_ocr()` method
4. Handle errors and retries
5. Remove `@pytest.mark.skip` from integration test

### Potential Enhancements
- Add more document types (DOCX, images, etc.) - just add `_extract_from_X()` methods
- Add more OCR providers (AWS, Google) - create new OCR service classes
- Consider caching for repeated OCR calls
- Add structured data extraction (tables, forms) beyond just text
- Batch processing support

### Known Patterns Established
- Services go in `src/services/`
- Abstract strategies in `base.py`, implementations in separate files
- Utils are standalone functions, not classes
- Test fixtures in `tests/fixtures/`
- Integration tests are marked and skipped until implemented

## Reflection

This session established the foundation for document processing in Atlas. The key win was taking time to think through the architecture before coding - the back-and-forth about layers, responsibilities, and structure led to a much cleaner design than jumping straight to implementation.

The decision to start async, organize tests properly, and create comprehensive skeletons/TODOs sets up good habits for the rest of the project. Everything is testable, swappable, and ready to scale.

