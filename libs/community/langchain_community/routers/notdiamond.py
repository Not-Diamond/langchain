import os
from typing import List, Any, Dict, Sequence

from langchain_community.adapters.openai import convert_message_to_dict
from langchain_core.prompt_values import PromptValue, StringPromptValue, ChatPromptValue
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages.utils import convert_to_messages
from langchain_core.runnables.base import Runnable, Output

from notdiamond import NotDiamond, LLMConfig

class NotDiamondRunnable(Runnable[LanguageModelInput, str]):
    """
    See Runnable docs for details
    https://python.langchain.com/v0.1/docs/expression_language/interface/
    """
    llm_configs: List[LLMConfig | str]
    api_key: str

    def __init__(self, llm_configs: List[LLMConfig | str], api_key: str=os.getenv("NOTDIAMOND_API_KEY")):
        self.client = NotDiamond(llm_configs=llm_configs, api_key=api_key)

    def _model_select(self, input: List[LanguageModelInput]) -> str:
        messages = _convert_input_to_message_dicts(input)
        _, provider = self.client.chat.completions.model_select(messages=messages)
        return provider

    def invoke(self, input: List[LanguageModelInput]) -> str:
        return self._model_select(input)

    def chain(self, input: List[LanguageModelInput]) -> str:
        return self._model_select(input)

    def batch(self, input: List[LanguageModelInput]) -> str:
        """ [a9] do i need to change this to list[list[language_model_input]]? """
        return self._model_select(input)

def _convert_input_to_message_dicts(input: LanguageModelInput) -> List[Dict[str, str]]:
    if isinstance(input, PromptValue):
        output = input
    elif isinstance(input, str):
        output = StringPromptValue(text=input)
    elif isinstance(input, Sequence):
        output = ChatPromptValue(messages=convert_to_messages(input))
    else:
        raise ValueError(
            f"Invalid input type {type(input)}. "
            "Must be a PromptValue, str, or list of BaseMessages."
        )
    return [
        convert_message_to_dict(message)
        for message in output.to_messages()
    ]


