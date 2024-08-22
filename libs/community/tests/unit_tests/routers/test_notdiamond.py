import pytest
import random
from typing import Dict, List
import uuid
from unittest.mock import MagicMock, AsyncMock

from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models import BaseChatModel
from langchain_community.routers.notdiamond import NotDiamondRunnable, NotDiamondRoutedRunnable, _nd_provider_to_langchain_provider

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
    client = MagicMock(spec=NotDiamond, llm_configs=llm_configs)
    selected_model = random.choice(llm_configs)
    client.chat.completions.model_select = MagicMock(return_value=(uuid.uuid4(), selected_model))
    client.chat.completions.amodel_select = AsyncMock(return_value=(uuid.uuid4(), selected_model))
    return client

@pytest.fixture
def not_diamond_runnable(llm_configs, nd_client):
    return NotDiamondRunnable(llm_configs=llm_configs, api_key='', default_model='openai/gpt-4o', client=nd_client)

@pytest.fixture
def not_diamond_routed_runnable(llm_configs, nd_client):
    routed_runnable = NotDiamondRoutedRunnable(llm_configs=llm_configs, api_key='', default_model='openai/gpt-4o', client=nd_client)
    routed_runnable._configurable_model = MagicMock(spec=_ConfigurableModel)
    return routed_runnable

@pytest.mark.requires("notdiamond")
class TestNotDiamondRunnables:

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