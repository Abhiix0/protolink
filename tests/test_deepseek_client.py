from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from protolink.llms.actions import FinalAction, ToolCallAction
from protolink.llms.api.deepseek_client import DeepSeekLLM
from protolink.llms.history import ConversationHistory
from protolink.llms.tool_calling import AGENT_INFER_TOOL_NAME, AGENT_TOOL_CALL_TOOL_NAME


class DummyTool:
    name = "lookup"
    description = "Look up a value."
    input_schema: ClassVar[dict] = {"key": {"type": "string", "required": True}}
    output_schema: ClassVar[dict] = {"type": "string"}
    tags: ClassVar[list] = []


def _make_deepseek_llm(*, supports_tool_calling: bool = True) -> DeepSeekLLM:
    llm = DeepSeekLLM.__new__(DeepSeekLLM)
    llm.model = "deepseek-chat"
    llm._model_params = {"temperature": 1.0, "top_p": 1.0}
    llm.history = ConversationHistory()
    llm.system_prompt = ""
    llm._supports_tool_calling = supports_tool_calling
    llm._client = MagicMock()
    return llm


def test_deepseek_call_returns_expected_text():
    llm = _make_deepseek_llm()

    mock_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hello from DeepSeek!"))])
    llm._client.chat.completions.create.return_value = mock_response

    history = ConversationHistory()
    history.add_system("You are a helpful assistant.")
    history.add_user("Say hi.")

    result = llm.call(history)

    assert result == "Hello from DeepSeek!"
    llm._client.chat.completions.create.assert_called_once_with(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hi."},
        ],
        stream=False,
        temperature=1.0,
        top_p=1.0,
    )


def test_deepseek_call_action_builds_tool_call_action():
    llm = _make_deepseek_llm()

    tool_call = SimpleNamespace(function=SimpleNamespace(name="lookup", arguments='{"key": "alpha"}'))
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))]
    )
    llm._client.chat.completions.create.return_value = mock_response

    history = ConversationHistory()
    history.add_user("Find alpha")
    result = llm.call_action(
        history,
        tools={"lookup": DummyTool()},
        agent_callback_available=False,
    )

    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["stream"] is False
    assert kwargs["tool_choice"] == "auto"

    assert isinstance(result.action, ToolCallAction)
    assert result.action.tool == "lookup"
    assert result.action.args == {"key": "alpha"}
    assert result.native is True
    assert result.metadata["provider"] == "deepseek"


def test_deepseek_call_action_builds_final_action():
    llm = _make_deepseek_llm()

    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Here is the final answer.", tool_calls=None))]
    )
    llm._client.chat.completions.create.return_value = mock_response

    history = ConversationHistory()
    history.add_user("Answer the question")
    result = llm.call_action(history, tools={"lookup": DummyTool()})

    assert isinstance(result.action, FinalAction)
    assert result.action.content == "Here is the final answer."
    assert result.native is True
    assert result.metadata["provider"] == "deepseek"


def test_deepseek_call_action_raises_on_empty_content_and_no_tool_calls():
    llm = _make_deepseek_llm()

    mock_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="   ", tool_calls=[]))])
    llm._client.chat.completions.create.return_value = mock_response

    history = ConversationHistory()
    history.add_user("Silent response")

    with pytest.raises(ValueError, match="DeepSeek response did not contain content or tool_calls"):
        llm.call_action(history, tools={"lookup": DummyTool()})


@pytest.mark.asyncio
async def test_deepseek_call_stream_yields_text_deltas():
    llm = _make_deepseek_llm()

    llm._client.chat.completions.create.return_value = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Deep"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Seek"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" stream"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
    ]

    history = ConversationHistory()
    history.add_user("Stream test")

    chunks = [chunk async for chunk in llm.call_stream(history)]

    assert chunks == ["Deep", "Seek", " stream"]
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["messages"] == [{"role": "user", "content": "Stream test"}]


def test_deepseek_call_action_respects_agent_tool_exclusion():
    llm = _make_deepseek_llm()

    mock_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Done", tool_calls=None))])
    llm._client.chat.completions.create.return_value = mock_response

    history = ConversationHistory()
    history.add_user("Check tool exclusion")

    # When agent_callback_available is False
    llm.call_action(
        history,
        tools={"lookup": DummyTool()},
        agent_callback_available=False,
    )
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    tool_names = {t["function"]["name"] for t in kwargs["tools"]}
    assert tool_names == {"lookup"}
    assert AGENT_INFER_TOOL_NAME not in tool_names
    assert AGENT_TOOL_CALL_TOOL_NAME not in tool_names

    # When agent_callback_available is True and agent_cards are provided
    llm.call_action(
        history,
        tools={"lookup": DummyTool()},
        agent_callback_available=True,
        agent_cards=[{"name": "helper_agent"}],
    )
    kwargs = llm._client.chat.completions.create.call_args.kwargs
    tool_names = {t["function"]["name"] for t in kwargs["tools"]}
    assert "lookup" in tool_names
    assert AGENT_INFER_TOOL_NAME in tool_names
    assert AGENT_TOOL_CALL_TOOL_NAME in tool_names


@pytest.mark.asyncio
async def test_deepseek_call_action_stream_collects_tool_call_deltas():
    llm = _make_deepseek_llm()

    llm._client.chat.completions.create.return_value = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_ds_1",
                                function=SimpleNamespace(name="lookup", arguments='{"key":'),
                            )
                        ],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                function=SimpleNamespace(name=None, arguments='"alpha"}'),
                            )
                        ],
                    )
                )
            ]
        ),
    ]

    history = ConversationHistory()
    history.add_user("Find alpha via stream")
    result = await llm.call_action_stream(
        history,
        tools={"lookup": DummyTool()},
        agent_callback_available=False,
    )

    kwargs = llm._client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert isinstance(result.action, ToolCallAction)
    assert result.action.tool == "lookup"
    assert result.action.args == {"key": "alpha"}
    assert result.native is True
    assert result.metadata["streaming"] is True
    assert result.metadata["provider"] == "deepseek"


@pytest.mark.asyncio
async def test_deepseek_call_action_stream_text_deltas_with_callback():
    llm = _make_deepseek_llm()

    llm._client.chat.completions.create.return_value = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello", tool_calls=None))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" world!", tool_calls=None))]),
    ]

    callback = AsyncMock()
    history = ConversationHistory()
    history.add_user("Stream text")

    result = await llm.call_action_stream(
        history,
        tools={"lookup": DummyTool()},
        chunk_callback=callback,
    )

    assert isinstance(result.action, FinalAction)
    assert result.action.content == "Hello world!"
    assert result.native is True
    assert callback.await_count == 2
    callback.assert_any_await("Hello")
    callback.assert_any_await(" world!")


def test_deepseek_tool_calling_properties():
    llm_with_tools = _make_deepseek_llm(supports_tool_calling=True)
    assert llm_with_tools.uses_native_action_prompt is True
    assert llm_with_tools.supports_native_action_stream is True

    llm_without_tools = _make_deepseek_llm(supports_tool_calling=False)
    assert llm_without_tools.uses_native_action_prompt is False
    assert llm_without_tools.supports_native_action_stream is False
