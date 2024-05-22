from datetime import date
from llama_index.core.types import ChatMessage, MessageRole


def create_chat_messages(system_msg, user_msg):
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=system_msg),
        ChatMessage(role=MessageRole.USER, content=user_msg)
    ]


QA_NEWS = (
    "You are asked to provide answer to the question below.\n"
    "{query_str}\n\n"
    "You should use the provided context below which you extracted from the internet to generate a very "
    "comprehensive, informative and detailed response. Think step by step before answering.\n"
    "------------\n"
    "{context_str}\n"
    "\n------------\n"
    "Your response must solely based on the provided context.\n"
    "Combine search results together into a coherent answer. Do not repeat text.\n"
    f"For your reference, today's date is: {str(date.today())}.\n"
    "Rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the given context, ...' or 'The context information ...' or anything along "
    "those lines."
)

QA_WEB = (
    "You are asked to provide answer to the question below.\n"
    "{query_str}\n\n"
    "Generate a very comprehensive, informative and detailed response based on your extensive training knowledge.\n"
    "You can take help of the additional context below, which you searched for and extracted from the internet to "
    "frame your answer. Think step by step before answering.\n"
    "------------\n"
    "{context_str}\n"
    "\n------------\n"
    f"Make sure you always answer the question, even if the context isn't helpful. Do not repeat text. For your "
    f"reference, today's date is: {str(date.today())} and context provided is up-to-date.\n"
    "Rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the given context, ...' or 'The context information ...' or anything along "
    "those lines."
)

# Refine QA Prompt for gpt-3.5-turbo/gpt-4 model
REFINE_QA_NEWS = (
    "You have the opportunity to refine the original answer (only if needed) with some more context below extracted "
    "from the web which you did on your own.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better answer the question: {query_str}.\n"
    "Make sure everything you say is supported by the web search results. Answer in a comprehensive, very informative "
    "and detailed manner. Do not repeat text and do not refer to the original answer as 'Original Answer' in your "
    "response. Think step by step before answering.\n "
    f"For your reference, today's date is: {str(date.today())}.\n"
    "Rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the given context, ...' or 'The context information ...' or anything along "
    "those lines. If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)

SYSTEM_MESSAGE = (
    "You are a helpful answering assistant called intelliweb-GPT that can answer user queries on any topic. "
    "Respond in a very comprehensive, informative and detailed manner"
)

REFINE_QA_WEB = (
    "You have the opportunity to refine the original answer (only if needed) with some more context below extracted "
    "from the web which you did on your own.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context and your prior knowledge, you can refine the original answer if anything new, and relevant "
    "to the question can be added. The question is: {query_str}. Make sure everything you say is supported by the web "
    "search results. Answer in a comprehensive, very informative and detailed manner. Do not repeat text and do not "
    "refer to the original answer as 'Original Answer' in your response. Think step by step before answering.\n"
    f"For your reference, today's date is: {str(date.today())}.\n"
    "Rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the given context, ...' or 'The context information ...' or anything along "
    "those lines. If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)

SOURCE_SELECTION = (
    "Based on the user query, decide on what source to use to best answer it. Your possible sources are given below:\n"
    "1. LLM: Useful for answering conversational queries and for queries related to capabilities of the model.\n"
    "If query is related to time-sensitive information, recent developments, or needs current data, then "
    "choose one of the sources below to get up-to-date info:\n"
    "2. Google Web Search: Useful when query asks about a specific topic or for events more than 3 weeks old. (Prefer "
    "this source for most cases)\n"
    "3. Google News Search: Useful when query asks about very recent events or news\n\n"
    "{query}\n"
)

QUERY_REFRAMING = (
    "You are given a conversation between a human and an assistant and a user query below. If the user query is "
    "related to the topic of messages in the conversation, reframe the query to include context from previous "
    "messages, specifically mentioning all relevant terms like diseases, genes, drugs, conditions, companies or "
    "organizations etc., in case it is mentioned in the conversation, so that the reframed query becomes standalone.\n"
    "If the user query is in no way related to the topic of the previous messages, return the query unmodified.\n\n"
    "<Conversation>\n{chat_history}\n"
    "<User Query>\n{user_query}\n"
    "<Reframed Query>\n"
)

FOLLOW_UP_QUERY_CREATION = (
    "You are part of the backend of an AI biomedical chatbot product. You, given the previous line of questioning of "
    "the chatbot user and the chatbot's response, infer the next questions that the user might want to ask.\n"
    "Predict the next questions the user may want to ask, giving three brief examples of plausible questions in the "
    "user’s voice. The chatbot user can then select one of them as their next input.\n\n"
    "Conversation:\n"
    "{serialized_conversation}\n"
)
