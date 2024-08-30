from importlib import metadata
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Sequence

from langchain.chat_models.base import init_chat_model
from langchain_community.adapters.openai import convert_message_to_dict
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages.utils import convert_to_messages
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.runnables import Runnable, RunnableConfig

import notdiamond as nd

_USER_AGENT = f"langchain-community/{metadata.version('notdiamond')}"
_LANGCHAIN_PROVIDERS = {
    "openai",
    "anthropic",
    "google",
    "mistral",
    "togetherai",
    "cohere",
}


class NotDiamondRunnable(Runnable[LanguageModelInput, str]):
    """
    See Runnable docs for details
    https://python.langchain.com/v0.1/docs/expression_language/interface/
    """

    llm_configs: List[nd.LLMConfig | str]
    api_key: str | None = os.getenv("NOTDIAMOND_API_KEY")
    client: nd.NotDiamond

    def __init__(
        self,
        nd_llm_configs: List[nd.LLMConfig | str] | None = None,
        nd_api_key: str | None = None,
        nd_client: nd.NotDiamond | None = None,
    ):
        if not nd_client:
            if not nd_api_key or not nd_llm_configs:
                raise ValueError(
                    "Must provide either client or api_key and llm_configs to instantiate NotDiamondRunnable."
                )
            nd_client = nd.NotDiamond(
                llm_configs=nd_llm_configs,
                api_key=nd_api_key,
            )

        for llm_config in nd_client.llm_configs:
            if isinstance(llm_config, str):
                llm_config = nd.LLMConfig.from_string(llm_config)
            if llm_config.provider not in _LANGCHAIN_PROVIDERS:
                raise ValueError(
                    f"Requested provider in {llm_config} supported by Not Diamond but not "
                    "langchain.chat_models.base.init_chat_model. Please remove it from your llm_configs."
                )

        try:
            nd_client.user_agent = _USER_AGENT
        except AttributeError:
            pass

        self.client = nd_client
        self.api_key = nd_client.api_key
        self.llm_configs = nd_llm_configs

    def _model_select(self, input: LanguageModelInput) -> str:
        messages = _convert_input_to_message_dicts(input)
        _, provider = self.client.chat.completions.model_select(messages=messages)
        provider_str = _nd_provider_to_langchain_provider(str(provider))
        print(provider_str)
        return provider_str

    async def _amodel_select(self, input: LanguageModelInput) -> str:
        messages = _convert_input_to_message_dicts(input)
        _, provider = await self.client.chat.completions.amodel_select(
            messages=messages
        )
        provider_str = _nd_provider_to_langchain_provider(str(provider))
        return provider_str

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[str]:
        yield self._model_select(input)

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None
    ) -> str:
        return self._model_select(input)

    def batch(
        self,
        inputs: List[LanguageModelInput],
        config: RunnableConfig | List[RunnableConfig] | None = None,
        **kwargs: Any | None,
    ) -> List[str]:
        return [self._model_select(input) for input in inputs]

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[str]:
        yield await self._amodel_select(input)

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> str:
        return await self._amodel_select(input)

    async def abatch(
        self,
        inputs: List[LanguageModelInput],
        config: RunnableConfig | List[RunnableConfig] | None = None,
        **kwargs: Any | None,
    ) -> List[str]:
        return [await self._amodel_select(input) for input in inputs]


class NotDiamondRoutedRunnable(Runnable[LanguageModelInput, Any]):
    def __init__(
        self,
        *args,
        configurable_fields: List[str] | None = None,
        nd_llm_configs: List[nd.LLMConfig | str] | None = None,
        nd_api_key: str | None = None,
        nd_client: nd.NotDiamond | None = None,
        **kwargs: Any | None,
    ):
        self._ndrunnable = NotDiamondRunnable(
            *args,
            **kwargs,
            nd_api_key=nd_api_key,
            nd_llm_configs=nd_llm_configs,
            nd_client=nd_client,
        )
        _nd_kwargs = {kw for kw in kwargs.keys() if kw.startswith("nd_")}

        _routed_fields = ["model", "model_provider"]
        if configurable_fields is None:
            configurable_fields = []
        self._configurable_fields = _routed_fields + configurable_fields
        self._configurable_model = init_chat_model(
            configurable_fields=self._configurable_fields,
            config_prefix="nd",
            *args,
            **{kw: kwv for kw, kwv in kwargs.items() if kw not in _nd_kwargs},
        )

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        provider_str = self._ndrunnable._model_select(input)
        _config = self._build_model_config(provider_str, config)
        yield from self._configurable_model.stream(input, config=_config)

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Any:
        provider_str = self._ndrunnable._model_select(input)
        _config = self._build_model_config(provider_str, config)
        return self._configurable_model.invoke(input, config=_config)

    def batch(
        self,
        inputs: List[LanguageModelInput],
        config: RunnableConfig | List[RunnableConfig] | None = None,
        **kwargs: Any | None,
    ) -> List[Any]:
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

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        provider_str = await self._ndrunnable._amodel_select(input)
        _config = self._build_model_config(provider_str, config)
        async for chunk in self._configurable_model.astream(input, config=_config):
            yield chunk

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Any:
        provider_str = await self._ndrunnable._amodel_select(input)
        _config = self._build_model_config(provider_str, config)
        return await self._configurable_model.ainvoke(input, config=_config)

    async def abatch(
        self,
        inputs: List[LanguageModelInput],
        config: RunnableConfig | List[RunnableConfig] | None = None,
        **kwargs: Any | None,
    ) -> List[Any]:
        config = config or {}

        provider_strs = [
            await self._ndrunnable._amodel_select(input) for input in inputs
        ]
        if isinstance(config, dict):
            _configs = [self._build_model_config(ps, config) for ps in provider_strs]
        else:
            _configs = [
                self._build_model_config(ps, config[i])
                for i, ps in enumerate(provider_strs)
            ]

        return await self._configurable_model.abatch(inputs, config=_configs)

    def _build_model_config(
        self, provider_str: str, config: RunnableConfig | None = None
    ) -> RunnableConfig:
        """
        Provider string should take the form '{model}/{model_provider}'
        """
        config = config or RunnableConfig()

        model_provider, model = provider_str.split("/")
        _config = RunnableConfig(
            configurable={
                "nd_model": model,
                "nd_model_provider": model_provider,
            },
        )

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
    provider, model = llm_config_str.split("/")
    provider = (
        provider.replace("google", "google_genai")
        .replace("mistral", "mistralai")
        .replace("togetherai", "together")
    )
    return f"{provider}/{model}"
