import asyncio
from typing import List, Tuple

import trafilatura
from trafilatura.settings import use_config

from llama_index import Document
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter


class DocumentGetter:
    def __init__(self):
        self._trafilatura_config = use_config()
        self._trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
        self._text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=10)

    async def _extract_text(self, url: str) -> str | None:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            response = trafilatura.extract(downloaded, include_comments=False, include_images=False,
                                           config=self._trafilatura_config)
            if response:
                return response
        return None

    async def _extract_text_from_url(self, urls: List[str]):
        tasks = [self._extract_text(url) for url in urls]
        extracted_texts = await asyncio.gather(*tasks)
        return extracted_texts, [url for text, url in zip(extracted_texts, urls) if text]

    def get_documents_from_urls(self, urls: List[str]) -> Tuple[List, List]:
        extracted_texts, scraped_urls = asyncio.run(self._extract_text_from_url(urls=urls))
        documents = self.get_documents_from_texts(extracted_texts)
        return documents, scraped_urls

    def get_documents_from_texts(self, texts: List[str]) -> List:
        documents = []
        for text in texts:
            if text:
                for t in self._text_splitter.split_text(text):
                    documents.append(Document(t))
        return documents
