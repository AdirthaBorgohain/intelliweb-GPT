import asyncio
from datetime import date
from GoogleNews import GoogleNews
from googlesearch import search
from intelliweb_GPT.tool_picker import get_best_tool
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, \
    SystemMessagePromptTemplate
from llama_index.readers import TrafilaturaWebReader
from llama_index import GPTSimpleVectorIndex, LangchainEmbedding, LLMPredictor, ServiceContext, \
    QuestionAnswerPrompt, RefinePrompt

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
            QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_NEWS_TMPL)
            CHAT_REFINE_QA_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_QA_PROMPT_NEWS_TMPL_MSGS)
            REFINE_QA_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_QA_PROMPT_LC)
        else:
            res = search(search_query, num_results=5)
            urls = [r for r in res]
            QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_WEB_TMPL)
            CHAT_REFINE_QA_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_QA_PROMPT_WEB_TMPL_MSGS)
            REFINE_QA_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_QA_PROMPT_LC)

        documents = reader.load_data(urls=urls)
        index = GPTSimpleVectorIndex.from_documents(documents, use_async=True, service_context=service_context)
        response = asyncio.run(
            index.aquery(query, response_mode='tree_summarize', similarity_top_k=5,
                         text_qa_template=QA_PROMPT, refine_template=REFINE_QA_PROMPT)
        )
        return {"answer": response.response.strip(), "references": urls}
    else:
        print(f"Using {tool_to_use} to generate your response...!")
        chat_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(SYSTEM_CHAT_TMPL),
                                                        HumanMessagePromptTemplate.from_template(query)])
        res = chat_model(chat_prompt.format_prompt(query=query).to_messages())
        return {"answer": res.content.strip()}


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

chat_model = ChatOpenAI(temperature=0.4, model_name='gpt-3.5-turbo', max_tokens=512)
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='multi-qa-MiniLM-L6-cos-v1'))
llm_predictor = LLMPredictor(llm=chat_model)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
reader = TrafilaturaWebReader()
