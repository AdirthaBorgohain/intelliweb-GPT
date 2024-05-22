from transformers import AutoTokenizer
from typing import Optional, Sequence, List
from llama_index.core.llms import ChatMessage

_tokenizers = {
    'zephyr-7b': AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta'),
    'mixtral-8x7b': AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1'),
}


def messages_to_prompt(
        messages: Sequence[ChatMessage], model: str, system_prompt: Optional[str] = None
) -> str:
    tokenizer = _tokenizers[model]
    if messages[0].role != 'system' and isinstance(messages, List):
        messages.insert(0, system_prompt)
    match model:
        case 'mixtral-8x7b':
            messages[0].content = f"{messages.pop(0).content}\n\n{messages[0].content}"
        case _:
            pass
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
