from typing import List, Dict, AsyncGenerator
from intelliweb_GPT.llms import load_llm

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.schema import Document
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


class QueryAnswerer:

    @staticmethod
    def _format_msgs(messages: List) -> List:
        formatted_msgs = []
        for msg in messages:
            match msg['role']:
                case 'system':
                    role = MessageRole.SYSTEM
                case 'user':
                    role = MessageRole.USER
                case 'assistant':
                    role = MessageRole.ASSISTANT
                case _:
                    raise Exception("Invalid message role found. Must be one of 'system', 'user', 'assistant'")
            formatted_msgs.append(ChatMessage(role=role, content=msg['content']))
        return formatted_msgs

    async def answer_from_documents(self, query: str, documents: List[Document], qa_prompt: List[ChatMessage] = None,
                                    refine_prompt: List[ChatMessage] = None, stream: bool = False,
                                    **kwargs) -> AsyncGenerator | Dict:
        """
        Given a query and list of documents, it generates answer to the query using the texts from the documents as
        contest.
        """

        async def _response_stream():
            query_results = await query_engine.aquery(query)
            for token in query_results.response_gen:
                yield token

        async def _response():
            query_results = await query_engine.aquery(query)
            yield query_results.response.strip()

        print("Generating answer from documents..")
        model_params = {
            'model': 'gpt-4o',
            'temperature': 0.8,
        }
        print(f"Chat model params for answer generation: {model_params}")
        llm = load_llm(**model_params)

        index = VectorStoreIndex.from_documents(
            documents,
            use_async=True,
            show_progress=True
        )
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=kwargs.get('similarity_top_k', 5),
        )
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode=kwargs.get('response_mode', 'compact'),
            streaming=stream,
            text_qa_template=ChatPromptTemplate(qa_prompt or []),
            refine_template=ChatPromptTemplate(refine_prompt or []),
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        return _response_stream() if stream else _response()

    async def answer_from_knowledge(self, query: str, chat_history: List[Dict] = None,
                                    stream: bool = False) -> AsyncGenerator | Dict:
        """
        Given a query, it generates answer to the query directly from the LLM model
        """

        async def _response_stream():
            query_results = await llm.astream_chat(formatted_messages)
            async for token in query_results:  # Iterate through the query results
                yield token

        async def _response():
            query_results = await llm.achat(formatted_messages)
            yield query_results.message.content.strip()

        print(f"Generating answer from training knowledge for query: {repr(query)}..")
        model_params = {
            'model': 'gpt-4o',
            'temperature': 0.8,
        }
        print(f"Chat model params for answer generation: {model_params}")
        llm = load_llm(**model_params)
        formatted_messages = self._format_msgs(messages=chat_history) + [
            ChatMessage(role=MessageRole.USER, content=query)]

        return _response_stream() if stream else _response()


__all__ = ['QueryAnswerer']
