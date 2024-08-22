import os
from typing import List, Dict, Sequence, AsyncIterator

from langchain.chat_models.base import init_chat_model
from langchain_community.adapters.openai import convert_message_to_dict
from langchain_core.prompt_values import PromptValue, StringPromptValue, ChatPromptValue
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages.utils import convert_to_messages
from langchain_core.runnables.base import Runnable

import notdiamond as nd


class NotDiamondRunnable(Runnable[LanguageModelInput, str]):
    """
    See Runnable docs for details
    https://python.langchain.com/v0.1/docs/expression_language/interface/
    """

    llm_configs: List[nd.LLMConfig | str]
    api_key: str
    client: nd.NotDiamond

    def __init__(
        self,
        llm_configs: List[nd.LLMConfig | str],
        api_key: str = os.getenv("NOTDIAMOND_API_KEY"),
        default_model: str = None,
    ):
        self.client = nd.NotDiamond(
            llm_configs=llm_configs, api_key=api_key, default=default_model
        )

    def _model_select(self, input: List[LanguageModelInput]) -> str:
        messages = _convert_input_to_message_dicts(input)
        _, provider = self.client.chat.completions.model_select(messages=messages)
        provider_str = _nd_provider_to_langchain_provider(str(provider))
        return provider_str

    async def _amodel_select(self, input: List[LanguageModelInput]) -> str:
        messages = _convert_input_to_message_dicts(input)
        _, provider = await self.client.chat.completions.amodel_select(
            messages=messages
        )
        provider_str = _nd_provider_to_langchain_provider(str(provider))
        return provider_str

    def stream(self, input: LanguageModelInput) -> str:
        return self._model_select(input)

    def invoke(self, input: LanguageModelInput) -> str:
        return self._model_select(input)

    def batch(self, inputs: List[LanguageModelInput]) -> List[str]:
        return [self._model_select(input) for input in inputs]

    async def astream(self, input: LanguageModelInput) -> AsyncIterator[str]:
        return await self._amodel_select(input)

    async def ainvoke(self, input: LanguageModelInput) -> str:
        return await self._amodel_select(input)

    async def abatch(self, inputs: List[LanguageModelInput]) -> List[str]:
        return [await self._amodel_select(input) for input in inputs]


class NotDiamondRoutedRunnable(Runnable[LanguageModelInput, str]):
    def __init__(
        self,
        llm_configs: List[nd.LLMConfig | str],
        api_key: str = os.getenv("NOTDIAMOND_API_KEY"),
        default_model: str = None,
        configurable_fields: List[str] = None,
        *args,
        **kwargs,
    ):
        self._ndrunnable = NotDiamondRunnable(llm_configs, api_key, default_model)

        _routed_fields = ["model", "model_provider"]
        if configurable_fields is None:
            configurable_fields = []
        self._configurable_fields = _routed_fields + configurable_fields
        self._configurable_model = init_chat_model(
            configurable_fields=self._configurable_fields,
            config_prefix="nd",
            *args,
            **kwargs,
        )

    def stream(self, input: LanguageModelInput, config: dict = None) -> str:
        provider_str = self._ndrunnable._model_select(input)
        _config = self._build_model_config(provider_str, config)
        return self._configurable_model.stream(input, config=_config)

    def invoke(self, input: LanguageModelInput, config: dict = None) -> str:
        provider_str = self._ndrunnable._model_select(input)
        _config = self._build_model_config(provider_str, config)
        return self._configurable_model.invoke(input, config=_config)

    def batch(
        self, inputs: List[LanguageModelInput], config: dict | List[dict] = None
    ) -> List[str]:
        config = config or {}

        provider_strs = [self._ndrunnable._model_select(input) for input in inputs]
        if isinstance(config, dict):
            _configs = [self._build_model_config(ps, config) for ps in provider_strs]
        else:
            _configs = [
                self._build_model_config(ps, config[i])
                for i, ps in enumerate(provider_strs)
            ]

        return self._configurable_model.batch(inputs, config=_configs)

    async def astream(self, input: LanguageModelInput, config: dict = None) -> AsyncIterator[str]:
        provider_str = await self._ndrunnable._amodel_select(input)
        _config = self._build_model_config(provider_str, config)
        async for chunk in self._configurable_model.astream(input, config=_config):
            yield chunk

    async def ainvoke(self, input: LanguageModelInput, config: dict = None) -> str:
        provider_str = await self._ndrunnable._amodel_select(input)
        _config = self._build_model_config(provider_str, config)
        return await self._configurable_model.ainvoke(input, config=_config)

    async def abatch(
        self, inputs: List[LanguageModelInput], config: dict | List[dict] = None
    ) -> List[str]:
        config = config or {}

        provider_strs = [await self._ndrunnable._amodel_select(input) for input in inputs]
        if isinstance(config, dict):
            _configs = [self._build_model_config(ps, config) for ps in provider_strs]
        else:
            _configs = [
                self._build_model_config(ps, config[i])
                for i, ps in enumerate(provider_strs)
            ]

        return await self._configurable_model.abatch(inputs, config=_configs)

    def _build_model_config(self, provider_str: str, config: dict = None) -> dict:
        config = config or {}

        _nd_llm_config = nd.LLMConfig.from_string(
            _nd_provider_to_langchain_provider(provider_str)
        )
        _config = {
            "configurable": {
                "nd_model": _nd_llm_config.model,
                "nd_model_provider": _nd_llm_config.model_provider,
            }
        }

        for k, v in config.items():
            _config["configurable"][f"nd_{k}"] = v
        return _config


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
    return [convert_message_to_dict(message) for message in output.to_messages()]


def _nd_provider_to_langchain_provider(llm_config_str: str) -> str:
    return llm_config_str.replace("google", "google_genai")
