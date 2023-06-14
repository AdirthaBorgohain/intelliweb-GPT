import os
import json
import requests
from GoogleNews import GoogleNews
from googlesearch import search


class WebRetriever:
    def __init__(self):
        self._headers = self.generate_headers()
        self._googlenews = GoogleNews()

    @staticmethod
    def generate_headers():
        return {
            'X-API-KEY': os.getenv('SERPER_API_KEY'),
            'Content-Type': 'application/json'
        }

    def _retrieve_from_serper_api(self, query: str, source: str):
        url = f"https://google.serper.dev/{source}"

        payload = json.dumps({
            "q": query
        })

        response = requests.request("POST", url, headers=self._headers, data=payload).json()
        if source == "news":
            urls = [r['link'] for r in response['news'][:7]]
        else:
            urls = []
            if response.get('answerBox') and response['answerBox'].get('link'):
                urls.append(response['answerBox']['link'])
            for r in response['organic'][:7]:
                urls.append(r['link'])

        print(f"Relevant urls fetched: {urls}")
        return urls

    def _retrieve_from_scraping(self, query: str, source: str):
        if source == "news":
            self._googlenews.search(query)
            urls = [data['link'] for data in self._googlenews.results(sort=True)[:7]]
            self._googlenews.clear()
        else:
            res = search(query, num_results=7)
            urls = [r for r in res]

        print(f"Relevant urls fetched: {urls}")
        return urls

    def retrieve_relevant_urls(self, query: str, source: str, use_serper_api: bool):
        if use_serper_api:
            return self._retrieve_from_serper_api(query, source)
        else:
            return self._retrieve_from_scraping(query, source)
