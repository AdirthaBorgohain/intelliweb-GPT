import asyncio
from pprint import pprint
from intelliweb_GPT import generate_answer

query = "How did the Super Mario Bros. movie fare at the box office?"
answer_dict = asyncio.run(generate_answer(query, use_serper_api=True))

pprint(answer_dict)
