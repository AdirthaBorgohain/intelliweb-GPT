from intelliweb_GPT.prompts import *
from intelliweb_GPT.components import QueryAnswerer, SourceSelector, WebRetriever, DocumentGetter

source_selector = SourceSelector()
query_answerer = QueryAnswerer()
web_retriever = WebRetriever()
document_getter = DocumentGetter()


def generate_answer(query: str, use_serper_api: bool = False):
    source_to_use, search_query = source_selector.select_optimal_source(query)
    if source_to_use == "LLM Model":
        formatted_chat_history = [{'role': 'system', 'content': SYSTEM_CHAT_TMPL}]
        return query_answerer.answer_from_knowledge(query, chat_history=formatted_chat_history)
    elif source_to_use == "Google News Search":
        source = 'news'
        QA_PROMPT, CHAT_REFINE_QA_PROMPT_LC = QA_PROMPT_NEWS_TMPL, CHAT_REFINE_QA_PROMPT_NEWS_TMPL_MSGS
    else:
        source = 'search'
        QA_PROMPT, CHAT_REFINE_QA_PROMPT_LC = QA_PROMPT_WEB_TMPL, CHAT_REFINE_QA_PROMPT_WEB_TMPL_MSGS

    retrieved_urls = web_retriever.retrieve_relevant_urls(search_query, source, use_serper_api)
    documents, references = document_getter.get_documents_from_urls(retrieved_urls)
    return {"answer": query_answerer.answer_from_documents(query, documents,
                                                           qa_prompt=QA_PROMPT,
                                                           refine_prompt=CHAT_REFINE_QA_PROMPT_LC),
            "references": references
            }
