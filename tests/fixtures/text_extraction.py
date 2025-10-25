"""
Shared test fixtures for text extraction tests.
Contains sample documents and expected outputs used across unit and integration tests.
"""

# Expected text from the dummy PDF (exactly 25 words)
EXPECTED_TEXT = "This is a test PDF document with exactly twenty five words to use for testing text extraction and validation functionality right here today now done."

# Dummy PDF document with the expected text embedded
DUMMY_PDF_BYTES = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 195 >>
stream
BT
/F1 12 Tf
50 700 Td
({EXPECTED_TEXT}) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000317 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
511
%%EOF
""".encode(
    "utf-8"
)

CONTENT_TYPE = "application/pdf"
