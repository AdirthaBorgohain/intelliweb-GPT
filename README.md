# Intelliweb-GPT

Intelliweb-GPT is an intelligent search engine / Question Answering module that uses GPT technology to provide accurate
and relevant answers to user queries. The search engine can search multiple sources, including Google News and Google
Web, to generate answers, and can also directly answer general user queries using GPT's existing knowledge. With GPT
connected to the internet, it's basically GPT on steroids! This is how Bing makes it's searches at the moment when you
use Bing Chat.

## Features 🚀

Intelliweb-GPT provides the following features:

- Multi-source search: The module can connect to the internet and can search Google News and Google Web to generate
  answers for user queries.
- Direct answer generation: Intelliweb-GPT can directly answer general user queries using GPT's existing knowledge.
- Intelligent decision-making: The module breaks down the user query to intelligently decide whether to use GPT's
  existing knowledge or external sources to generate answers.
- Easy-to-use API: Intelliweb-GPT provides a simple API that developers can use to integrate the search engine into
  their own applications.

[//]: # (## Installation 🔭)

[//]: # ()

[//]: # (You can install Intelliweb-GPT via pip:)

[//]: # ()

[//]: # (```shell)

[//]: # (pip install intelliweb_GPT)

[//]: # ()

[//]: # (```)

## Example Usage 💻

First install all libraries and modules mentioned in requirements.txt:

```shell
pip install -r requirements.txt
```

In the `.env` file, replace the `OPENAI_API_KEY` ENV variable's value with your own OpenAI API key.

Also, if you want to use serper API to fetch the relevant articles from the web, get your own API key
from [here](https://serper.dev/)
and add your API key to the `SERPER_API_KEY` env variable. It gives you 1000 free searches to start with and is safer
and more reliable than using the other methods. While calling the `generate_answer` method, set the `use_serper_api`
argument to `True`.

Next, here's a simple example of how you can use Intelliweb-GPT:

```python
from pprint import pprint
from intelliweb_GPT import generate_answer

query = "How did the Super Mario Bros. movie fare at the box office?"
answer_dict = generate_answer(query, use_serper_api=False)

pprint(answer_dict)
```

Output:

```bash
{'answer': 'The Super Mario Bros. movie has been a massive success at the box '
           'office, breaking multiple records and becoming the '
           'highest-grossing video game adaptation of all time, surpassing '
           'Detective Pikachu. As of 2023-04-15, the film grossed $204.6 '
           'million domestically and $377 million globally in its opening '
           'weekend, and has since surpassed $500 million worldwide. It has '
           'become the biggest opening of 2023, the highest-grossing debut  '
           'for Illumination, and the second-biggest debut ever for an '
           'animated movie. The movie has already made more than 2.5 times its '
           'budget, which is generally what a movie needs to break even. '
           "There's a strong chance that the video game movie could join the "
           'billion-dollar club despite the bad reviews. However, it remains '
           'to be seen how well it will do in the following weeks.',
           
 'sources': ['https://deadline.com/2023/04/super-mario-bros-movie-crosses-500-million-worldwide-box-office-1235325476/',
             'https://collider.com/super-mario-bros-movie-global-box-office-434-million/',
             'https://variety.com/2023/film/news/super-mario-bros-movie-box-office-records-opening-weekend-1235577764/',
             'https://www.shacknews.com/article/135069/super-mario-bros-movie-highest-grossing-video-game-film',
             'https://screenrant.com/super-mario-bros-movie-every-box-office-record/']}
```

### GUI

There is also a GUI available to interact with intelliweb-GPT. UI is created using chainlit and this can be run by
using the command below:

```shell
PYTHONPATH=$PWD chainlit run chainlit_app/app.py
```

Demo of how the Chainlit UI works with intelliweb-GPT:
![](assets/chainlit_demo.gif)

## Contributing 💡

Contributions to Intelliweb-GPT are welcome! To contribute, please follow these steps:

- Fork the Intelliweb-GPT repository.
- Create a new branch for your changes.
- Make your changes and commit them to your branch.
- Submit a pull request.
- Please make sure to follow the contribution guidelines when submitting your pull request.

## License 📖

Intelliweb-GPT is licensed under the MIT License. See LICENSE for more information.

### This project is built using:

<a href="https://github.com/run-llama/llama_index" target="_blank">
    <img alt="Llama Index Logo" src="https://www.llamaindex.ai/llamaindex.svg" width="100"/>
</a>
&nbsp&nbsp&nbsp&nbsp
<a href="https://github.com/Chainlit/chainlit" target="_blank">
    <img alt="Chainlit Logo" src="https://help.chainlit.io/logo?theme=dark" width="100"/>
</a>