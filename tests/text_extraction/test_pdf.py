from pydantic import BaseModel, Field


class TextExtractionResult(BaseModel):
    text: str
    character_length: int
    token_length: int
    duration: float = Field(description="duration in milliseconds")


# Dummy PDF document with exactly 25 words
DUMMY_PDF_BYTES = b"""%PDF-1.4
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
"""

CONTENT_TYPE = "application/pdf"

# TextExtractionResult model
# - text -> string
# - char lenght -> int
# - token length -> int
# - duration -> float

# OCR Service (mock only for right now - so need to somehow specify the text to return ? not sure how that works but think through it)
# - needs an abstract method that all OCR services have which has
# - input
# -- text -> string
# - output
# -- TextExtractionResult

# Assertions
# - check that text isn't empty and has a string with length of over x amount
# - check that char length is over certain amount
# - check that token length is over certain amount

# I know this worked if I provide bytes and get back text with the char / token length / duration properly populated

# NOTE
# - not sure how to handle this, but when i do the mock thing I guess the mock service can have something that specifies what i expect to return cause I'm mocking this - so maybe it just returns whatever i tell it to and then the live one would jsut return the actual text from the bytes

# - when i do the live one i can validate that the output is very similar to what I know it already is
# -- I guess i could do that here as well which probably makes sense

# - i guess the question is what am I trying to validate
# -- when I pass bytes for a pdf to this service I get back the text from those bytes with the associated metadata
# so this is kinda an integration test, where if i was unit testing i'd unit test the OCR service itself, which I'll do when I get to the live OCR service
