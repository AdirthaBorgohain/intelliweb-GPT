import nest_asyncio

nest_asyncio.apply()

from typing import Dict
from intelliweb_GPT.prompts import *
from intelliweb_GPT.components import QueryAnswerer, SourceSelector, WebRetriever, DocumentGetter

source_selector = SourceSelector()
query_answerer = QueryAnswerer()
web_retriever = WebRetriever()
document_getter = DocumentGetter()


async def generate_answer(query: str, use_serper_api: bool = False, stream: bool = False) -> Dict | str:
    """
    Generates answer for a given user query
    Args:
        query: User query
        use_serper_api: Whether to use serper_api or directly scrape from the search results. Defaults to False.
        stream: Whether to stream the answer or not. Defaults to False

    Returns:
        Returns a dictionary with the answer to the query and URL references from the web used to generate the answer
        (if any)
    """
    source_to_use, search_query = source_selector.select_optimal_source(query)
    if source_to_use == "LLM":
        formatted_chat_history = [{'role': 'system', 'content': SYSTEM_MESSAGE}]
        response = await query_answerer.answer_from_knowledge(query, chat_history=formatted_chat_history, stream=stream)
        if stream:
            return response
        else:
            async for r in response:
                return {"answer": r}
    elif source_to_use == "Google News Search":
        source = 'news'
        QA_PROMPT, CHAT_REFINE_QA_PROMPT_LC = QA_NEWS, REFINE_QA_NEWS
    else:
        source = 'search'
        QA_PROMPT, CHAT_REFINE_QA_PROMPT_LC = QA_WEB, REFINE_QA_WEB

    retrieved_urls = web_retriever.retrieve_relevant_urls(search_query, source, use_serper_api)
    documents, references = document_getter.get_documents_from_urls(retrieved_urls)
    response = await query_answerer.answer_from_documents(
        query, documents, stream=stream,
        qa_prompt=create_chat_messages(SYSTEM_MESSAGE, QA_PROMPT),
        refine_prompt=create_chat_messages(SYSTEM_MESSAGE, CHAT_REFINE_QA_PROMPT_LC)
    )
    if stream:
        return {
            "answer_generator": response,
            "references": references
        }
    else:
        async for r in response:
            return {
                "answer": r,
                "references": references
            }
