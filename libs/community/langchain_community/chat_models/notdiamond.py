from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env, pre_init

logger = logging.getLogger(__name__)


class ChatNotDiamondException(Exception):
    """Error with the `Not Diamond` library"""


def _create_retry_decorator(
    llm: ChatNotDiamond,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle PaLM exceptions"""
    from notdiamond import exceptions

    errors = [
        exceptions.UnsupportedLLMProvider,
        exceptions.InvalidApiKey,
        exceptions.MissingApiKey,
        exceptions.MissingLLMConfigs,
        exceptions.ApiError,
        exceptions.CreateUnavailableError
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )

def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""

        additional_kwargs = {}
        if _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(_dict["function_call"])

        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]

        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)

def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict

def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    if _dict.get("function_call"):
        additional_kwargs = {"function_call": dict(_dict["function_call"])}
    else:
        additional_kwargs = {}

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]
    
def completion_with_retry(
    llm: ChatNotDiamond,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        return llm.client.chat.completions.create(**kwargs)

    return _completion_with_retry(**kwargs)

async def acompletion_with_retry(
    llm: ChatNotDiamond,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await llm.client.chat.completions.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)

class ChatNotDiamond(BaseChatModel):
    """Chat model that uses the Not Diamond SDK."""

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    # Not Diamond parameters
    llm_configs: List[str] = Field(min_items=1)
    default: Union[int, str] = None
    max_model_depth: Optional[int] = 1
    latency_tracking: bool = True
    hash_content: bool = False
    tradeoff: Optional[str] = None
    preference_id: Optional[str] = None
    tools: Optional[Sequence[Union[Dict[str, Any], Callable]]] = None
    
    # API keys
    notdiamond_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    azure_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    replicate_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # other parameters like temperature, top_p, top_k, n go in model_kwargs
    streaming: bool = False
    max_retries: int = 6
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key and python package exists."""
        try:
            from notdiamond import NotDiamond
        except ImportError:
            raise ChatNotDiamondException(
                "Could not import notdiamond python package. "
                "Please install it with `pip install notdiamond[create]`"
            )
        
        values["notdiamond_api_key"] = get_from_dict_or_env(
            values, "notdiamond_api_key", "NOTDIAMOND_API_KEY", default=""
        )
        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY", default=""
        )
        values["anthropic_api_key"] = get_from_dict_or_env(
            values, "anthropic_api_key", "ANTHROPIC_API_KEY", default=""
        )
        values["replicate_api_key"] = get_from_dict_or_env(
            values, "replicate_api_key", "REPLICATE_API_KEY", default=""
        )
        values["cohere_api_key"] = get_from_dict_or_env(
            values, "cohere_api_key", "COHERE_API_KEY", default=""
        )
        values["perplexity_api_key"] = get_from_dict_or_env(
            values, "perplexity_api_key", "PERPLEXITYAI_API_KEY", default=""
        )
        values["together_ai_api_key"] = get_from_dict_or_env(
            values, "together_ai_api_key", "TOGETHERAI_API_KEY", default=""
        )
        values["mistral_api_key"] = get_from_dict_or_env(
            values, "mistral_api_key", "MISTRAL_API_KEY", default=""
        )
        notdiamond_client = NotDiamond(llm_configs=values["llm_configs"], api_key=values["notdiamond_api_key"])
        values["client"] = notdiamond_client
        return values
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Not Diamond's API."""
        return {
            "model": self.llm_configs,
            "default": self.default,
            "max_model_depth": self.max_model_depth,
            "latency_tracking": self.latency_tracking,
            "hash_content": self.hash_content,
            "tradeoff": self.tradeoff,
            "preference_id": self.preference_id,
            # "tools": self.tools,
            **self.model_kwargs,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "llm_configs": self.llm_configs,
            "tradeoff": self.tradeoff,
            "tools": self.tools,
        }

    @property
    def _llm_type(self) -> str:
        return "notdiamond-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        print("message_dicts", message_dicts)
        print("params", params)
        print("kwargs", kwargs)
        should_stream = stream if stream is not None else self.streaming

        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        
        params = {**params, **kwargs}
        response, session_id, provider = completion_with_retry(
            self,
            messages=message_dicts,
            run_manager=run_manager, 
            **params
        )
        result = {
            'response': response,
            'session_id': session_id,
            'provider': provider
        }
        print("result", result)
        # print("type of response:", type(response))
        # return ChatResult(generations=['response'], llm_output={'session_id': session_id, 'provider': provider.model})
        return self._create_chat_result(result)
        # return response

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        should_stream = stream if stream is not None else self.streaming

        if should_stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        params = {**params, **kwargs}
        
        response, session_id, provider = await acompletion_with_retry(
            self,
            messages=message_dicts,
            run_manager=run_manager, 
            **params
        )
        # result = {
        #     'response': response,
        #     'session_id': session_id,
        #     'provider': provider
        # }
        return self._create_chat_result(result)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        generator_response, session_id, provider = completion_with_retry(
            self,
            messages=message_dicts,
            run_manager=run_manager,
            **params,
        )
        for chunk in generator_response:
            if len(chunk["choices"]) == 0:
                continue
            delta = chunk["choices"][0]["delta"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        generator_response, session_id, provider = completion_with_retry(
            self,
            messages=message_dicts,
            models_priority_list=self.models_priority_list,
            run_manager=run_manager,
            **params,
        )
        async for chunk in await generator_response:
            if len(chunk["choices"]) == 0:
                continue
            delta = chunk["choices"][0]["delta"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
            yield cg_chunk

    # def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
    #     generations = []
    #     for res in response['response']:
    #         message = _convert_dict_to_message(res)
    #         gen = ChatGeneration(
    #             message=message,
    #             generation_info=dict(finish_reason=res.response_metadata.get("finish_reason")),
    #         )
    #         generations.append(gen)
    #     input_tokens = response.usage_metadata.get("input_tokens", {})
    #     output_tokens = response.usage_metadata.get("output_tokens", {})
    #     total_tokens = response.usage_metadata.get("total_tokens", {})
    #     session_id = response['session_id']
    #     provider = response['provider']
    #     llm_output = {
    #         "session_id": session_id,
    #         "recommended_model": provider.model,
    #         "input_tokens": input_tokens,
    #         "output_tokens": output_tokens,
    #         "total_tokens": total_tokens,
    #     }
    #     return ChatResult(generations=generations, llm_output=llm_output)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        res = response['response']
        gen = ChatGeneration(
            message=AIMessage(content=res.content),
            generation_info=dict(finish_reason=res.response_metadata.get("finish_reason")),
        )
        input_tokens = res.usage_metadata.get("input_tokens", {})
        output_tokens = res.usage_metadata.get("output_tokens", {})
        total_tokens = res.usage_metadata.get("total_tokens", {})
        session_id = response['session_id']
        provider = response['provider']
        llm_output = {
            "session_id": session_id,
            "model_name": f"{provider.provider}/{provider.model}",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        return ChatResult(generations=[gen], llm_output=llm_output)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        return self.client.bind_tools(tools)