from datetime import date
from langchain.prompts.chat import AIMessagePromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

QA_PROMPT_NEWS_TMPL = (
    "Based on the provided web search results below. \n"
    "------------\n"
    "{context_str}\n"
    "\n------------\n"
    "Generate a comprehensive, very informative and detailed response (but not more than 150 words) "
    "to answer the question below. Your response must solely based on the provided web search results above. \n"
    "Combine search results together into a coherent answer. Do not repeat text.\n"
    f"For your reference, today's date is: {str(date.today())}.\n"
    "{query_str}\n"
)

QA_PROMPT_WEB_TMPL = (
    "You are asked to provide answer to the question below. \n"
    "{query_str}\n"
    "Generate a very comprehensive, informative and detailed response (but not more than 150 words)"
    "based on your extensive training knowledge. Do not repeat text.\n"
    "If needed, you can use the additional context below to better your answer.\n"
    "------------\n"
    "{context_str}\n"
    "\n------------\n"
    f"For your reference, today's date is: {str(date.today())} and context provided is up-to-date.\n"
)

# Refine QA Prompt for gpt-3.5-turbo/gpt-4 model
CHAT_REFINE_QA_PROMPT_NEWS_TMPL_MSGS = [
    HumanMessagePromptTemplate.from_template("{query_str}"),
    AIMessagePromptTemplate.from_template("{existing_answer}"),
    HumanMessagePromptTemplate.from_template(
        "You have the opportunity to refine your above answer "
        "(only if needed) with some more context below extracted from web search results.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question (but not more than 150 words). Make sure everything you say is supported by the web "
        "search results. Answer in a comprehensive, very informative and detailed manner. Do not repeat text. "
        "If the context isn't useful, output the original answer again.",
    ),
]

SYSTEM_CHAT_TMPL = (
    "You are a helpful answering assistant that can answer user queries on any topic. Respond in a very "
    "comprehensive, informative and detailed manner"
)

CHAT_REFINE_QA_PROMPT_WEB_TMPL_MSGS = [
    SystemMessagePromptTemplate.from_template(
        SYSTEM_CHAT_TMPL
    ),
    HumanMessagePromptTemplate.from_template("{query_str}"),
    AIMessagePromptTemplate.from_template("{existing_answer}"),
    HumanMessagePromptTemplate.from_template(
        "You have the opportunity to refine your above answer "
        "(only if needed) with some more context below extracted from web search results.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context and your prior knowledge, you can refine the original answer if "
        "anything new and relevant to the answer can be added. Make sure your answer is not more than 150 words. "
        "Answer in a comprehensive, very informative and detailed manner. Do not repeat text. Do not mention the usage "
        "of this additional context anywhere in your answer. If the context isn't useful, output the original answer "
        "again.",
    ),
]

SOURCE_SELECTION_PROMPT = (
    "Based on the user query, decide on what source to use. Your possible sources are given below:\n"
    "1. LLM Model: Useful for answering conversational queries and for queries related to capabilities of the "
    "model.\n"
    "If query is related to time-sensitive information, recent developments, or needs current data, only then "
    "choose one of the web search sources:\n"
    "2. Google Web Search: Useful when query asks about a specific topic or for events more than 3 weeks old\n"
    "3. Google News Search: Useful when query asks about very recent events or news\n\n"
    "{format_instructions}\n{query}\n"
)
