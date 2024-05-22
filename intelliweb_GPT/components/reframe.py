import os
from typing import List, Dict
from pydantic import BaseModel, Field

from intelliweb_GPT.llms import load_llm
from intelliweb_GPT.prompts import QUERY_REFRAMING

from llama_index.core.llms import LLM
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import LLMTextCompletionProgram


class ReframedQuery(BaseModel):
    """
    This Pydantic class aids in reformulating user queries based on the context of the conversation, ensuring that
    the reframed query can function independently.
    """
    reframed_query: str = Field(
        description="A user's query that has been modified to be self-sufficient and does not rely on prior "
                    "conversation context"
    )


class QueryReframer:
    """
    A class for reframing queries.
    """

    @staticmethod
    def _format_to_chat_buffer(messages: List):
        buffer = ""
        for chat in messages:
            role = f"{chat['role'].title() if chat['role'] != 'user' else 'Human'}: " \
                   f"{chat['content'].split('<hr><h3>References:</h3>')[0]}"
            buffer += "\n" + role + "\n"
        return buffer

    def reframe_query(self, messages: List[Dict], model: None | LLM = None):
        """
        Reframes user query from chat history.

        Args:
           messages (List[Dict]): Chat history
           model (None | LLM): Model to use for selecting the source. If none, loads a gpt-4 model

        Returns:
           reframed_query (Dict): Reframed query.
        """
        if len(messages) == 1:
            return messages[0]['content']

        llm = model or load_llm(model=os.getenv('GPT_MODEL_LITE', 'gpt-4o'))

        user_query, chat_history = messages[-1], messages[:-1]
        print(f"Reframing user query {repr(user_query['content'])} from chat history: {chat_history}")
        chat_buffer = self._format_to_chat_buffer(chat_history)

        program = LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(ReframedQuery),
            prompt_template_str=QUERY_REFRAMING,
            llm=llm,
            verbose=True,
        )
        reframed_query = user_query['content']  # default value
        for _ in range(4):
            try:
                output = program(chat_history=chat_buffer, user_query=user_query['content'])
                reframed_query = output.reframed_query
                break
            except:
                pass
        reframed_query = reframed_query or user_query['content']
        print(f"Reframed query from chat history: {repr(reframed_query)}")
        return reframed_query


__all__ = ['QueryReframer']
