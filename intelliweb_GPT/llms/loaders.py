import os
import re

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic

ollama_api_url = os.getenv("OLLAMA_API_URL")

_llm_token_limits = {
    'gpt-4o': 4096,
    'gpt-4-turbo': 4096,
    'gpt-4': 4096 - 512,
    'gpt-3.5-turbo': 2048,
    'claude-3-opus-20240229': 4096,
    'claude-3-sonnet-20240229': 4096,
    'claude-3-haiku-20240307': 4096,
    'zephyr-7b': 4096 - 512,
    'mixtral-8x7b': 4096 - 512,
}


def load_llm(model: str = 'gpt-4o', temperature: float = 0.4, **kwargs) -> LLM:
    if bool(re.match(r'^gpt', model)):
        llm = OpenAI(
            model=model,
            temperature=temperature,
            max_tokens=_llm_token_limits[model],
            request_timeout=120,
            seed=42
        )
    elif bool(re.match(r'^claude', model)):
        llm = Anthropic(
            model=model,
            temperature=temperature,
            max_tokens=_llm_token_limits[model],
        )
    elif ollama_api_url:
        llm = Ollama(
            model=model,
            base_url=ollama_api_url,
            temperature=temperature,
            additional_kwargs={
                'num_predict': _llm_token_limits[model]
                # 'template': "{{ .Prompt }}",  # Not sure if this is necessary or not
            },
        )
    else:
        raise Exception(f"Configuration for {model} model not found. Please re-verify your model name."
                        f"Valid model names are: {', '.join([f'{repr(m)}' for m in _llm_token_limits.keys()])}.")

    print(f"Loaded LLM: {model}")
    return llm
