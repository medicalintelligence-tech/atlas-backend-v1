"""
Shared test fixtures for text extraction tests.
Contains sample documents and expected outputs used across unit and integration tests.
"""

# Expected text from the dummy PDF (exactly 25 words)
EXPECTED_TEXT = "This is a test PDF document with exactly twenty five words to use for testing text extraction and validation functionality right here today now done."

# Dummy PDF document with the expected text embedded (hardcoded to ensure correct Length)
DUMMY_PDF_BYTES = """%PDF-1.4
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
<< /Length 181 >>
stream
BT
/F1 12 Tf
50 700 Td
(This is a test PDF document with exactly twenty five words to use for testing text extraction and validation functionality right here today now done.) Tj
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

# Simple test PDF with just "Hello World"
SIMPLE_TEXT = "Hello World"

SIMPLE_PDF_BYTES = """%PDF-1.4
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
<< /Length 44 >>
stream
BT
/F1 12 Tf
50 700 Td
(Hello World) Tj
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
469
%%EOF
""".encode(
    "utf-8"
)

# Azure integration test PDF with exactly 25 words
AZURE_TEST_TEXT = "Medical records show patient has chronic hypertension and diabetes. Treatment plan includes medication monitoring and lifestyle modifications for optimal health management and disease control."

AZURE_TEST_PDF_BYTES = """%PDF-1.4
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
<< /Length 233 >>
stream
BT
/F1 12 Tf
50 700 Td
(Medical records show patient has chronic hypertension and diabetes.) Tj
0 -15 Td
(Treatment plan includes medication monitoring and lifestyle) Tj
0 -15 Td
(modifications for optimal health management and disease control.) Tj
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
563
%%EOF
""".encode(
    "utf-8"
)

# HTML test fixtures
HTML_EXPECTED_TEXT = "Medical Report Patient: John Doe Diagnosis: The patient presents with chronic hypertension and type 2 diabetes. Treatment plan includes lifestyle modifications and medication management."

HTML_CONTENT_TYPE = "text/html"

DUMMY_HTML_BYTES = """<!DOCTYPE html>
<html>
<body>
    <div class="header">
        <h1>Medical Report</h1>
    </div>
    <div class="content">
        <h2>Patient: John Doe</h2>
        <p><strong>Diagnosis:</strong></p>
        <p>The patient presents with chronic hypertension and type 2 diabetes.</p>
        <p>Treatment plan includes lifestyle modifications and medication management.</p>
    </div>
    <script>
        // This script content should be ignored
        console.log("This should not appear in extracted text");
    </script>
</body>
</html>
""".encode(
    "utf-8"
)

# Plain text test fixtures
PLAIN_TEXT_EXPECTED_TEXT = "Patient notes: Initial consultation completed. Blood pressure readings elevated. Recommend follow-up in two weeks for reassessment and treatment adjustment."

PLAIN_TEXT_CONTENT_TYPE = "text/plain"

DUMMY_PLAIN_TEXT_BYTES = PLAIN_TEXT_EXPECTED_TEXT.encode("utf-8")
