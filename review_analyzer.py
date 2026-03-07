from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
import re


class ReviewAnalysisResponse(BaseModel):
    summary: str = Field(
        description="A brief summary of the customer review with maximum 3 lines"
    )
    positives: List[str] = Field(
        description="A list showing the positives mentioned by the customer in the review if any - max 3 points"
    )
    negatives: List[str] = Field(
        description="A list showing the negatives mentioned by the customer in the review if any - max 3 points"
    )
    sentiment: str = Field(
        description="One word showing the sentiment of the review - positive, negative or neutral"
    )


parser = PydanticOutputParser(pydantic_object=ReviewAnalysisResponse)
llm = ChatOllama(model="phi3:mini")

prompt = PromptTemplate(
    template="""
You are an AI assistant that analyzes customer reviews.

Return ONLY valid JSON following this schema.

{format_instructions}

Customer Review:
{review}
""",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def clean_json(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


print("Customer Review Analyzer")
print("Type 'exit' to quit")

while True:

    review = input("\nEnter customer review: ")

    if review.lower() == "exit":
        print("Exiting!.....👋")
        break

    try:

        formatted_prompt = prompt.format(review=review)

        response = llm.invoke(formatted_prompt)

        cleaned = clean_json(response.content)

        structured_output = parser.parse(cleaned)

        print("\nStructured Analysis:\n")
        print(structured_output.model_dump_json(indent=2))

    except Exception as e:

        print("\nError occurred while processing the review.")
        print("Details:", e)
