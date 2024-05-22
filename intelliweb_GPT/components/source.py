from typing import Tuple, Literal
from pydantic import BaseModel, Field

from intelliweb_GPT.llms import load_llm
from intelliweb_GPT.prompts import SOURCE_SELECTION

from llama_index.core.llms import LLM
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser


class SearchHelper(BaseModel):
    """
    Pydantic Class that aids in search processes. It consists of two fields - 'source' and 'search_query'.
    The 'source' signifies the source to be used in generating the answer for the user query whereas 'search_query'
    represents the most suitable search query for the chosen 'source'.
    """
    source: Literal['Google Web Search', 'Google News Search', 'LLM'] = Field(
        description="Denotes the source employed to provide the most suitable answer for the user query",
        example="Google Web Search"
    )
    search_query: str = Field(
        description="Represents the most suitable search query for obtaining the optimum results for the user's web "
                    "search. In case of no web search, 'NA' is returned. It must not contain any conditional operators "
                    "such as 'AND', 'OR', etc. and filters like 'site', etc."
    )


class SourceSelector:
    """
    SourceSelector class to select the optimal source for answering user query.
    """

    def select_optimal_source(self, query: str, model: None | LLM = None) -> Tuple[str, str]:
        """
        Select the optimal source to answer the user query.

        Args:
            query (str): The user query.
            model (None | LLM): Model to use for selecting the source. If none, loads a gpt-4 model

        Returns:
            Tuple[str, str]: The selected source and the optimal search query.
        """
        llm = model or load_llm(model='gpt-4o')

        program = LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(SearchHelper),
            prompt_template_str=SOURCE_SELECTION,
            llm=llm,
            verbose=True,
        )
        try:
            output = program(query=query)
            print(
                f"Using source: {repr(output.source)} with search query: {repr(output.search_query)}")
            return output.source, output.search_query
        except:
            print("Failed to pick any source for answering. Defaulting to 'Google Web Search' with "
                  f"search query: {repr(query)}")
        return "Google Web Search", query


__all__ = ['SourceSelector']
