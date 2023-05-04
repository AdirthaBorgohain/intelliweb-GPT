import os
import json
import asyncio
import requests
import trafilatura
from trafilatura.settings import use_config
from GoogleNews import GoogleNews
from googlesearch import search
from intelliweb_GPT.prompts import *
from intelliweb_GPT.tool_picker import get_best_tool
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, \
    SystemMessagePromptTemplate

from llama_index import Document
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index import GPTVectorStoreIndex, LangchainEmbedding, LLMPredictor, ServiceContext, \
    QuestionAnswerPrompt, RefinePrompt

__all__ = ['generate_answer']
serper_url = "https://google.serper.dev"
serper_api_key = os.getenv("SERPER_API_KEY")

config = use_config()
config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")


async def extract_text(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        response = trafilatura.extract(downloaded, include_comments=False, include_images=False, config=config)
        if response:
            return response
    return None


async def extract_text_from_url(urls):
    tasks = [extract_text(url) for url in urls]
    extracted_texts = await asyncio.gather(*tasks)
    return extracted_texts


def get_relevant_urls(query: str, source: str):
    url = f"{serper_url}/{source}"

    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload).json()
    if source == "search":
        urls = []
        if response.get('answerBox') and response['answerBox'].get('link'):
            urls.append(response['answerBox']['link'])
        for r in response['organic'][:7]:
            urls.append(r['link'])
        return urls
    elif source == "news":
        return [r['link'] for r in response['news'][:7]]


def generate_answer(query: str, use_serper_api: bool = False):
    tool_to_use, search_query = get_best_tool(query)
    if tool_to_use in ['Google News Search', 'Google Web Search']:
        print(f"Using {tool_to_use} and searching for '{search_query}'...!")
        if tool_to_use == "Google News Search":
            if use_serper_api:
                urls = get_relevant_urls(search_query, "news")
            else:
                googlenews = GoogleNews()
                googlenews.search(search_query)
                urls = [data['link'] for data in googlenews.results(sort=True)[:7]]
                googlenews.clear()
            QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_NEWS_TMPL)
            CHAT_REFINE_QA_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_QA_PROMPT_NEWS_TMPL_MSGS)
            REFINE_QA_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_QA_PROMPT_LC)
        else:
            if use_serper_api:
                urls = get_relevant_urls(search_query, "search")
            else:
                res = search(search_query, num_results=7)
                urls = [r for r in res]
            QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_WEB_TMPL)
            CHAT_REFINE_QA_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_QA_PROMPT_WEB_TMPL_MSGS)
            REFINE_QA_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_QA_PROMPT_LC)

        extracted_texts = asyncio.run(extract_text_from_url(urls=urls))
        documents = []
        for text in extracted_texts:
            if text:
                texts = text_splitter.split_text(text)
                for t in texts:
                    documents.append(Document(t))

        index = GPTVectorStoreIndex.from_documents(documents, use_async=True, service_context=service_context)
        query_engine = index.as_query_engine(
            response_mode="tree_summarize",
            service_context=service_context,
            similarity_top_k=5,
            text_qa_template=QA_PROMPT,
            refine_template=REFINE_QA_PROMPT
        )
        response = asyncio.run(query_engine.aquery(query))
        return {"answer": response.response.strip(), "references": urls}
    else:
        print(f"Using {tool_to_use} to generate your response...!")
        chat_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(SYSTEM_CHAT_TMPL),
                                                        HumanMessagePromptTemplate.from_template(query)])
        res = chat_model(chat_prompt.format_prompt(query=query).to_messages())
        return {"answer": res.content.strip()}


text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=10)
chat_model = ChatOpenAI(temperature=0.4, model_name='gpt-4', max_tokens=512)
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='multi-qa-MiniLM-L6-cos-v1'))
llm_predictor = LLMPredictor(llm=chat_model)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
