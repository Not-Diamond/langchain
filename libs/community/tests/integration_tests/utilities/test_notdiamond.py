import os
import pytest

from langchain_community.utilities.notdiamond import NotDiamondRoutedRunnable
from langchain_core.runnables import RunnableLambda

from notdiamond import LLMConfig, NotDiamond
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def nd_routed_runnable():
    api_key = os.getenv('NOTDIAMOND_API_KEY')
    llm_configs = [
        LLMConfig(provider="openai", model="gpt-4o-2024-08-06"),
        LLMConfig(provider="openai", model="gpt-4o-mini-2024-07-18")
    ]
    nd_client = NotDiamond(api_key=api_key, llm_configs=llm_configs, default='openai/gpt-4o-mini-2024-07-18')
    return NotDiamondRoutedRunnable(nd_client=nd_client)

def test_notdiamond_routed_runnable(nd_routed_runnable):
    result = nd_routed_runnable.invoke('Hello, world!')
    assert result.response_metadata is not None
    assert 'gpt' in result.response_metadata['model_name']

def test_notdiamond_routed_runnable_chain(nd_routed_runnable):
    fn = lambda x: x + '!'
    chain = RunnableLambda(fn) | nd_routed_runnable
    result = chain.invoke("Hello there! Not Diamond sent me to you. Which OpenAI model are you?")
    assert result.response_metadata is not None
    assert 'gpt' in result.response_metadata['model_name']
    print(result)
    assert False
