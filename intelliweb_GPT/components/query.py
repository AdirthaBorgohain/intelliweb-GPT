import asyncio
from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage

from llama_index import Document, QuestionAnswerPrompt, RefinePrompt, \
    LangchainEmbedding, LLMPredictor, ServiceContext, GPTVectorStoreIndex


class QueryAnswerer:
    def __init__(self):
        self._embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1"))

    def _initialize_model(self, model: str, temperature: float):
        """
        Initializes the necessary model and service contexts for generating answers
        """
        max_tokens = 1024 if model == 'gpt-3.5-turbo' else 2048
        chat_model = ChatOpenAI(model_name=model, temperature=temperature, max_tokens=max_tokens, request_timeout=120)
        return chat_model, ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=chat_model),
                                                        embed_model=self._embed_model, num_output=max_tokens)

    @staticmethod
    def _format_msgs(messages: List) -> List:
        formatted_msgs = []
        for msg in messages:
            match msg['role']:
                case 'system':
                    formatted_msgs.append(SystemMessage(content=msg['content']))
                case 'user':
                    formatted_msgs.append(HumanMessage(content=msg['content']))
                case 'assistant':
                    formatted_msgs.append(AIMessage(content=msg['content']))
        return formatted_msgs

    def answer_from_documents(self, query: str, documents: List[Document],
                              qa_prompt: str, refine_prompt: List[BaseMessage], **kwargs) -> str:
        """
        Given a query and list of documents, it generates answer to the query using the texts from the documents as
        contest.
        """
        print("Generating answer from documents..")
        model_params = {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.4,
        }
        print(f"Chat model params for answer generation: {model_params}")
        _, service_context = self._initialize_model(**model_params)
        index = GPTVectorStoreIndex.from_documents(documents, use_async=True, service_context=service_context)
        print(f"Documents successfully indexed! Generating answer for query: {repr(query)}...")
        query_engine = index.as_query_engine(
            response_mode="tree_summarize",
            service_context=service_context,
            similarity_top_k=kwargs.get('similarity_top_k', 5),
            text_qa_template=QuestionAnswerPrompt(qa_prompt),
            refine_template=RefinePrompt.from_langchain_prompt(
                ChatPromptTemplate.from_messages(refine_prompt)),
        )
        query_results = asyncio.run(query_engine.aquery(query))
        return query_results.response.strip()

    def answer_from_knowledge(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        Given a query, it generates answer to the query directly from the LLM model
        """
        print(f"Generating answer from training knowledge for query: {repr(query)}..")
        model_params = {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.4,
        }
        print(f"Chat model params for answer generation: {model_params}")
        chat_model, _ = self._initialize_model(**model_params)
        formatted_messages = self._format_msgs(messages=chat_history) + [HumanMessage(content=query)]
        query_results = chat_model(formatted_messages)
        return query_results.content.strip()


__all__ = ['QueryAnswerer']
