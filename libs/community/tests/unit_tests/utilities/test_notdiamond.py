from importlib import import_module
import pytest
import random
from typing import Dict, List
import uuid
from unittest.mock import MagicMock, AsyncMock, patch

from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models import BaseChatModel
from langchain_community.utilities.notdiamond import NotDiamondRunnable, NotDiamondRoutedRunnable, _nd_provider_to_langchain_provider

from notdiamond import LLMConfig, NotDiamond

@pytest.fixture
def llm_configs() -> List[LLMConfig]:
    return [
        LLMConfig(provider="openai", model="gpt-4o"),
        LLMConfig(provider="anthropic", model="claude-3-opus-20240229"),
        LLMConfig(provider='google', model='gemini-1.5-pro-latest')
    ]

@pytest.fixture
def llm_config_to_chat_model() -> Dict[str, BaseChatModel]:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    return {
        'openai/gpt-4o': MagicMock(spec=ChatOpenAI, model='gpt-4o'),
        'anthropic/claude-3-opus-20240229': MagicMock(spec=ChatAnthropic, model='claude-3-opus-20240229'),
        'google/gemini-1.5-pro-latest': MagicMock(spec=ChatGoogleGenerativeAI, model='gemini-1.5-pro-latest')
    }

@pytest.fixture
def nd_client(llm_configs):
    client = MagicMock(spec=NotDiamond, llm_configs=llm_configs, api_key='', default='openai/gpt-4o')
    selected_model = random.choice(llm_configs)
    client.chat.completions.model_select = MagicMock(return_value=(uuid.uuid4(), selected_model))
    client.chat.completions.amodel_select = AsyncMock(return_value=(uuid.uuid4(), selected_model))
    return client

@pytest.fixture
def not_diamond_runnable(nd_client):
    return NotDiamondRunnable(nd_client=nd_client)

@pytest.fixture
def not_diamond_routed_runnable(nd_client):
    routed_runnable = NotDiamondRoutedRunnable(nd_client=nd_client)
    routed_runnable._configurable_model = MagicMock(spec=_ConfigurableModel)
    return routed_runnable

@pytest.mark.requires("notdiamond")
class TestNotDiamondRunnable:

    def test_model_select(self, not_diamond_runnable, llm_configs):
        actual_select = not_diamond_runnable._model_select('Hello, world!')
        assert str(actual_select) in [_nd_provider_to_langchain_provider(str(config)) for config in llm_configs]

    @pytest.mark.asyncio
    async def test_amodel_select(self, not_diamond_runnable, llm_configs):
        actual_select = await not_diamond_runnable._amodel_select('Hello, world!')
        assert str(actual_select) in [_nd_provider_to_langchain_provider(str(config)) for config in llm_configs]

class TestNotDiamondRoutedRunnable:

    def test_invoke(self, not_diamond_routed_runnable):
        not_diamond_routed_runnable.invoke('Hello, world!')
        assert not_diamond_routed_runnable._configurable_model.invoke.called, f"{not_diamond_routed_runnable._configurable_model}"

        # Check the call list
        call_list = not_diamond_routed_runnable._configurable_model.invoke.call_args_list
        assert len(call_list) == 1
        args, kwargs = call_list[0]
        assert args[0] == 'Hello, world!'


    def test_stream(self, not_diamond_routed_runnable):
        for result in not_diamond_routed_runnable.stream('Hello, world!'):
            assert result is not None
        assert not_diamond_routed_runnable._configurable_model.stream.called, f"{not_diamond_routed_runnable._configurable_model}"

    def test_batch(self, not_diamond_routed_runnable):
        not_diamond_routed_runnable.batch(['Hello, world!', 'How are you today?'])
        assert not_diamond_routed_runnable._configurable_model.batch.called, f"{not_diamond_routed_runnable._configurable_model}"

        # Check the call list
        call_list = not_diamond_routed_runnable._configurable_model.batch.call_args_list
        assert len(call_list) == 1
        args, kwargs = call_list[0]
        assert args[0] == ['Hello, world!', 'How are you today?']

    @pytest.mark.asyncio
    async def test_ainvoke(self, not_diamond_routed_runnable):
        await not_diamond_routed_runnable.ainvoke('Hello, world!')
        assert not_diamond_routed_runnable._configurable_model.ainvoke.called, f"{not_diamond_routed_runnable._configurable_model}"

        # Check the call list
        call_list = not_diamond_routed_runnable._configurable_model.ainvoke.call_args_list
        assert len(call_list) == 1
        args, kwargs = call_list[0]
        assert args[0] == 'Hello, world!'

    @pytest.mark.asyncio
    async def test_astream(self, not_diamond_routed_runnable):
        async for result in not_diamond_routed_runnable.astream('Hello, world!'):
            assert result is not None
        assert not_diamond_routed_runnable._configurable_model.astream.called, f"{not_diamond_routed_runnable._configurable_model}"

    @pytest.mark.asyncio
    async def test_abatch(self, not_diamond_routed_runnable):
        await not_diamond_routed_runnable.abatch(['Hello, world!', 'How are you today?'])
        assert not_diamond_routed_runnable._configurable_model.abatch.called

        # Check the call list
        call_list = not_diamond_routed_runnable._configurable_model.abatch.call_args_list
        assert len(call_list) == 1
        args, kwargs = call_list[0]
        assert args[0] == ['Hello, world!', 'How are you today?']

    @pytest.mark.parametrize("target_model,patch_class", [
        ("openai/gpt-4o", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4o-2024-08-06", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4o-2024-05-13", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4-turbo-2024-04-09", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4-0125-preview", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4-1106-preview", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4-0613", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-3.5-turbo-0125", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4o-mini-2024-07-18", "langchain_openai.ChatOpenAI"),
        ("anthropic/claude-3-5-sonnet-20240620", "langchain_anthropic.ChatAnthropic"),
        ("anthropic/claude-3-opus-20240229", "langchain_anthropic.ChatAnthropic"),
        ("anthropic/claude-3-sonnet-20240229", "langchain_anthropic.ChatAnthropic"),
        ("anthropic/claude-3-haiku-20240307", "langchain_anthropic.ChatAnthropic"),
        ("google/gemini-1.5-pro-latest", "langchain_google_genai.ChatGoogleGenerativeAI"),
        ("google/gemini-1.5-flash-latest", "langchain_google_genai.ChatGoogleGenerativeAI"),
        ("mistral/open-mixtral-8x22b", "langchain_mistralai.ChatMistralAI"),
        ("mistral/codestral-latest", "langchain_mistralai.ChatMistralAI"),
        ("mistral/open-mixtral-8x7b", "langchain_mistralai.ChatMistralAI"),
        ("mistral/mistral-large-2407", "langchain_mistralai.ChatMistralAI"),
        ("mistral/mistral-large-2402", "langchain_mistralai.ChatMistralAI"),
        ("mistral/mistral-medium-latest", "langchain_mistralai.ChatMistralAI"),
        ("mistral/mistral-small-latest", "langchain_mistralai.ChatMistralAI"),
        ("mistral/open-mistral-7b", "langchain_mistralai.ChatMistralAI"),
        ("togetherai/Llama-3-70b-chat-hf", "langchain_together.ChatTogether"),
        ("togetherai/Llama-3-8b-chat-hf", "langchain_together.ChatTogether"),
        ("togetherai/Meta-Llama-3.1-8B-Instruct-Turbo", "langchain_together.ChatTogether"),
        ("togetherai/Meta-Llama-3.1-70B-Instruct-Turbo", "langchain_together.ChatTogether"),
        ("togetherai/Meta-Llama-3.1-405B-Instruct-Turbo", "langchain_together.ChatTogether"),
        ("togetherai/Qwen2-72B-Instruct", "langchain_together.ChatTogether"),
        ("togetherai/Mixtral-8x22B-Instruct-v0.1", "langchain_together.ChatTogether"),
        ("togetherai/Mixtral-8x7B-Instruct-v0.1", "langchain_together.ChatTogether"),
        ("togetherai/Mistral-7B-Instruct-v0.2", "langchain_together.ChatTogether"),
        ("cohere/command-r-plus", "langchain_cohere.ChatCohere"),
        ("cohere/command-r", "langchain_cohere.ChatCohere"),
    ])
    def test_invokable(self, target_model, patch_class):
        nd_client = MagicMock(spec=NotDiamond, llm_configs=[target_model], api_key='', default=target_model)
        nd_client.chat.completions.model_select = MagicMock(return_value=(uuid.uuid4(), target_model))

        module_name, cls_name = patch_class.split('.')
        cls = getattr(import_module(module_name), cls_name)
        mock_client = MagicMock(spec=cls)

        with patch(patch_class, autospec=True) as mock_class:
            mock_class.return_value = mock_client
            runnable = NotDiamondRoutedRunnable(nd_client=nd_client)
            runnable.invoke("Test prompt")
            assert mock_client.invoke.called

        mock_client.reset_mock()

        with patch(patch_class, autospec=True) as mock_class:
            mock_class.return_value = mock_client
            runnable = NotDiamondRoutedRunnable(nd_api_key='sk-...', nd_llm_configs=[target_model])
            runnable.invoke("Test prompt")
            assert mock_client.invoke.called

    def test_init_perplexity(self):
        target_model = "perplexity/llama-3.1-sonar-large-128k-online"
        nd_client = MagicMock(spec=NotDiamond, llm_configs=[target_model], api_key='', default=target_model)
        nd_client.chat.completions.model_select = MagicMock(return_value=(uuid.uuid4(), target_model))

        with pytest.raises(ValueError):
            NotDiamondRoutedRunnable(nd_client=nd_client)