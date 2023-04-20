from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate


class SearchHelper(BaseModel):
    source: str = Field(description="source to use to best answer user query")
    search_query: str = Field(
        description="optimal search query to use to get best results for user query in case of web search, else return NA")


tool_picker_template = (
    "Based on the user query, decide on what source to use. Your possible sources are given below:\n"
    "1. LLM Model: Useful when query is complex, and requires analytical or logical reasoning and does not need recent data. Also useful for general conversational queries.\n"
    "If query is related to time-sensitive information, recent developments, or needs current data, choose one of the web search sources:\n"
    "2. Google Web Search\n"
    "3. Google News Search\n\n"
    "{format_instructions}\n{query}\n"
)
parser = PydanticOutputParser(pydantic_object=SearchHelper)
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(tool_picker_template)
    ],
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chat_model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)


def get_best_tool(query):
    _input = prompt.format_prompt(query=query)
    output = chat_model(_input.to_messages())
    try:
        parsed_output = parser.parse(output.content)
        return parsed_output.source, parsed_output.search_query
    except:
        return 'llm', 'NA'
