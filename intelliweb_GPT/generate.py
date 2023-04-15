from datetime import date
from GoogleNews import GoogleNews
from googlesearch import search
from intelliweb_GPT.tool_picker import get_best_tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, \
    SystemMessagePromptTemplate
from llama_index import Document, GPTListIndex, LLMPredictor, ServiceContext, QuestionAnswerPrompt, RefinePrompt
from tqdm import tqdm

from intelliweb_GPT.text_extractors import extract_text_from_web_page

__all__ = ['generate_answer']


def generate_answer(query: str):
    tool_to_use, search_query = get_best_tool(query)
    if tool_to_use in ['Google News Search', 'Google Web Search']:
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
        index = GPTListIndex.from_documents([Document(text_content) for text_content in text_contents[:5]],
                                            service_context=service_context)
        response = index.query(query, response_mode="tree_summarize", use_async=True,
                               text_qa_template=SUMMARY_PROMPT, refine_template=REFINE_SUMMARY_PROMPT_CHAT)
        return {"answer": response.response.strip(), "references": urls}
    else:
        print(f"Using {tool_to_use} to generate your response...!")
        chat_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(SYSTEM_CHAT_TMPL),
                                                        HumanMessagePromptTemplate.from_template(query)])
        res = chat_model(chat_prompt.format_prompt(query=query).to_messages())
        return {"answer": res.content.strip()}


SYSTEM_CHAT_TMPL = (
    "You are a helpful answering assistant that can answer user queries on any topic. Respond in a very comprehensive, "
    "very informative and detailed manner"
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

chat_model = ChatOpenAI(temperature=0.4, model_name='gpt-3.5-turbo')
llm_predictor = LLMPredictor(llm=chat_model)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
