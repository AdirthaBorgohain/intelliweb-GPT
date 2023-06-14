from typing import Tuple
from pydantic import BaseModel, Field
from intelliweb_GPT.prompts import SOURCE_SELECTION_PROMPT
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate


class SearchHelper(BaseModel):
    source: str = Field(
        description="source to use to best answer user query"
    )
    search_query: str = Field(
        description="optimal search query to use to get best results for user query in case of web search, else return "
                    "NA"
    )


class SourceSelector:
    def __init__(self):
        self._chat_model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
        self._parser = PydanticOutputParser(pydantic_object=SearchHelper)
        self._prompt = self._initialize_prompt()

    def _initialize_prompt(self):
        return ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(SOURCE_SELECTION_PROMPT)
            ],
            input_variables=["query"],
            partial_variables={"format_instructions": self._parser.get_format_instructions()}
        )

    def select_optimal_source(self, query: str) -> Tuple[str, str]:
        _input = self._prompt.format_prompt(query=query)
        output = self._chat_model(_input.to_messages())
        try:
            parsed_output = self._parser.parse(output.content)
            print(f"Using source: {repr(parsed_output.source)} with search query: "
                  f"{repr(parsed_output.search_query)}")
            return parsed_output.source, parsed_output.search_query
        except:
            print("Failed to pick any source for answering. Defaulting to 'Google Web Search' with "
                  f"search query: {repr(query)}")
            print(output.content)
            return "Google Web Search", query


__all__ = ['SourceSelector']
