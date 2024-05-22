import os
from time import time
from typing import List
import concurrent.futures
from trafilatura.settings import use_config

from llama_index.core.schema import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.readers.web import TrafilaturaWebReader, SpiderWebReader


class DocumentGetter:
    """
    Fetches and extracts documents from the provided URLs.

    Attributes:
        _spider_reader: Configured document reader for spider
        _trafilatura_reader: Configured document reader for trafilatura
    """

    def __init__(self):
        self._text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=10)
        self._spider_reader = SpiderWebReader(
            api_key=os.getenv("SPIDER_API_KEY", "dummy-key"),
            mode="scrape",
            params={
                "proxy": True,
                "headless": True,
                "metadata": False,
                "return_format": "text"
            }
        )
        self._trafilatura_config = use_config()
        self._trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
        self._trafilatura_reader = TrafilaturaWebReader()

    def _scrape_with_trafilatura(self, url: str):
        return self._trafilatura_reader.load_data(urls=[url], include_comments=False, include_tables=False,
                                                  config=self._trafilatura_config)

    def _scrape_with_spider(self, url: str):
        return self._spider_reader.load_data(url=url)

    def get_documents_from_urls(self, urls: List[str], scraper: str = None) -> (List, List):
        scraper = scraper or os.getenv("SCRAPER")
        match scraper:
            case "spider":
                print("Scraping with spider...")
                scrape_url_func = self._scrape_with_spider
            case _:
                print("Scraping with trafilatura...")
                scrape_url_func = self._scrape_with_trafilatura

        start_time = time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_to_url = {executor.submit(scrape_url_func, url): url for url in urls}
            documents, scraped_urls = [], []

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    document = future.result()
                    if document:
                        documents.extend(document)
                        scraped_urls.append(url)
                except Exception as exc:
                    pass
                    # print(f'{url} generated an exception: {exc}')
        end_time = time()
        print(f"Scraping {len(urls)} URLs took {end_time - start_time} seconds.")
        return documents, scraped_urls

    def get_documents_from_texts(self, texts: List[str]) -> List[Document]:
        """
        Converts texts into documents.

        Args:
            texts (List[str]): List of texts to create documents from

        Returns:
            List[Document]: List of created documents
        """
        documents = []
        for text in texts:
            if text:
                for t in self._text_splitter.split_text(text):
                    documents.append(Document(text=t))
        return documents


if __name__ == "__main__":
    document_getter = DocumentGetter()
    documents, scraped_urls = document_getter.get_documents_from_urls(urls=[
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2517082/",
        "https://tau.amegroups.org/article/view/11491/html",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8022167/",
        "https://www.betterhealth.vic.gov.au/health/conditionsandtreatments/prostate-gland-and-urinary-problems",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4150581/",
        "https://www.goodrx.com/drugs/side-effects/medications-that-cause-gynecomastia",
        "https://www.cancerresearchuk.org/about-cancer/treatment/hormone-therapy/side-effects-men",
        "https://emedicine.medscape.com/article/231574-treatment",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4884108/",
    ], scraper="default")
    print(documents)
    print(f"Successfully scraped {len(scraped_urls)} URLs: {scraped_urls}")
