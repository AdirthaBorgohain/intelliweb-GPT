import os
from pprint import pprint
from intelliweb_GPT import generate_answer

os.environ['OPENAI_API_KEY'] = "sk-XXXXXXXXXXXXXX"

query = "How did the Super Mario Bros. movie fare at the box office?"
answer_dict = generate_answer(query)

pprint(answer_dict)
