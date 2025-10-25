from pydantic import BaseModel, Field


class TextExtractionResult(BaseModel):
    text: str
    character_length: int
    token_length: int
    duration: float = Field(description="duration in milliseconds")


# initial state
# TODO - need bytes for a dummy pdf document - should be pretty small, let's call it 25 words
# TODO - content type is pdf

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
