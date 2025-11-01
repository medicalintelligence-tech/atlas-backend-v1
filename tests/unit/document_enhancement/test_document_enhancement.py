# create a document enhancement service
# pass in the llm service that will be used to do the document enhancment
# - which itself has an llm client to use (i think)

# the llm service has an abstract method called enhance document or something that runs the document enhancement
# returns the enhancement data (need schema for this) - in this case mocked
# returns the llm usage (need schema for this) - in this case mocked


# validate that the enhancement data is correct
# validate that the llm usage data is correct

# after you do this try a live one just to make sure it works, but don't worry about the enhancmenet itself until you get to the llm eval part which will be sepearate

# then do the actual llm eval stuff (or at least set it up i think, but not sure about this one - you might try doing the rest of it and then come back to the eval stuff since it's slightly different) - but idk, you'll have to do it at some point either way
