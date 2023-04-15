from datetime import date
from GoogleNews import GoogleNews
from googlesearch import search
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from llama_index import Document, GPTListIndex, LLMPredictor, ServiceContext, QuestionAnswerPrompt, RefinePrompt
from pydantic import BaseModel, Field
from tqdm import tqdm

from intelliweb_GPT.text_extractors import extract_text_from_web_page

__all__ = ['generate_answer']


class SearchHelper(BaseModel):
    action: str = Field(description="source to use to best answer user query")
    search_query: str = Field(
        description="optimal search query to use to get best results for user query in case of web search, else return NA")


def generate_answer(query: str):
    _input = prompt.format_prompt(query=query)
    output = chat_model(_input.to_messages())
    try:
        parsed_output = parser.parse(output.content)
        tool_to_use = parsed_output.action
    except:
        print('failed...')
        parsed_output = None
        tool_to_use = 'llm'

    if tool_to_use in ['Google News Search', 'Google Web Search']:
        search_query = parsed_output.search_query
        print(f"Using {tool_to_use} and searching for '{search_query}'...!")
        if tool_to_use == "Google News Search":
            googlenews = GoogleNews()
            googlenews.search(search_query)
            urls = [data['link'] for data in googlenews.results(sort=True)[:5]]
            googlenews.clear()
        else:
            res = search(search_query, num_results=5)
            urls = [r for r in res]

        text_contents = [extract_text_from_web_page(url) for url in tqdm(urls)]
        text_contents = [text_content for text_content in text_contents if str(text_content) != "nan"]
        index = GPTListIndex.from_documents([Document(text_content) for text_content in text_contents],
                                            service_context=service_context)
        response = index.query(query, response_mode="tree_summarize", use_async=True,
                               text_qa_template=SUMMARY_PROMPT, refine_template=REFINE_SUMMARY_PROMPT_CHAT)
        return {"answer": response.response.strip(), "references": urls}
    else:
        print("Need to use LLM to get answer...")
        raise NotImplementedError("Not implemented answer generation with LLM model directly yet...")


parser = PydanticOutputParser(pydantic_object=SearchHelper)

init_template = (
    "Based on the user query, decide on what source to use. Your possible sources are given below:\n"
    "1. LLM Model: Useful when query is complex, and requires analytical/logical reasoning and does not need recent data. Also useful for general conversational queries.\n"
    "If query is related to time-sensitive information, recent developments, or needs current data, choose one of the web search sources:\n"
    "2. Google Web Search\n"
    "3. Google News Search\n\n"
    "{format_instructions}\n{query}\n"
)

chat_model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(init_template)
    ],
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

SUMMARY_PROMPT_TMPL = (
    "Based on the provided web search results below. \n"
    "------------\n"
    "{context_str}\n"
    "\n------------\n"
    "Generate a comprehensive, very informative and detailed response (but not more than 100 words) "
    "to answer the question below. Your response must solely based on the provided web search results above. \n"
    "Combine search results together into a coherent answer. Do not repeat text.\n"
    f"Today's date is: {str(date.today())} and use this as your reference point.\n"
    "{query_str}\n"
)
SUMMARY_PROMPT = QuestionAnswerPrompt(SUMMARY_PROMPT_TMPL)

CHAT_REFINE_SUMMARY_PROMPT_TMPL_MSGS = [
    HumanMessagePromptTemplate.from_template("{query_str}"),
    AIMessagePromptTemplate.from_template("{existing_answer}"),
    HumanMessagePromptTemplate.from_template(
        "We have the opportunity to refine the above answer "
        "(only if needed) with some more context below extracted from web search results.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question (but not more than 100 words). Make sure everything you say is supported by the web search results."
        "Answer in a comprehensive, very informative and detailed manner. Do not repeat text. "
        "If the context isn't useful, output the original answer again.",
    ),
]
CHAT_REFINE_SUMMARY_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_SUMMARY_PROMPT_TMPL_MSGS)
REFINE_SUMMARY_PROMPT_CHAT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_SUMMARY_PROMPT_LC)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.4, model_name='gpt-3.5-turbo'))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
