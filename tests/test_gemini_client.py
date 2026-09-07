from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from protolink.llms.actions import FinalAction, ToolCallAction
from protolink.llms.api.gemini_client import GeminiLLM
from protolink.llms.history import ConversationHistory
from protolink.llms.tool_calling import (
    AGENT_INFER_TOOL_NAME,
    AGENT_TOOL_CALL_TOOL_NAME,
    gemini_function_declarations,
)


class DummyTool:
    name = "lookup"
    description = "Look up a value."
    input_schema: ClassVar[dict] = {"key": {"type": "string", "required": True}}
    output_schema: ClassVar[dict] = {"type": "string"}
    tags: ClassVar[list] = []


def _make_gemini_llm() -> GeminiLLM:
    llm = GeminiLLM.__new__(GeminiLLM)
    llm.model = "gemini-test"
    llm._model_params = {"temperature": 1.0, "top_p": 1.0}
    llm.history = ConversationHistory()
    llm.system_prompt = ""
    llm._client = MagicMock()
    llm._GenerateContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)
    return llm


def test_gemini_call_returns_expected_text():
    llm = _make_gemini_llm()

    mock_response = SimpleNamespace(text="Hello from Gemini!")
    llm._client.models.generate_content.return_value = mock_response

    history = ConversationHistory()
    history.add_system("You are a helpful assistant.")
    history.add_user("Say hello.")

    result = llm.call(history)

    assert result == "Hello from Gemini!"
    llm._client.models.generate_content.assert_called_once_with(
        model="gemini-test",
        contents="You are a helpful assistant.\nSay hello.",
        config={"temperature": 1.0, "top_p": 1.0},
    )


def test_gemini_call_action_converts_function_call_to_tool_call_action():
    llm = _make_gemini_llm()

    function_call = SimpleNamespace(name="lookup", args={"key": "alpha"})
    part = SimpleNamespace(function_call=function_call)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    mock_response = SimpleNamespace(candidates=[candidate], text=None)
    llm._client.models.generate_content.return_value = mock_response

    history = ConversationHistory()
    history.add_user("Find alpha")
    result = llm.call_action(
        history,
        tools={"lookup": DummyTool()},
        agent_callback_available=False,
    )

    assert isinstance(result.action, ToolCallAction)
    assert result.action.tool == "lookup"
    assert result.action.args == {"key": "alpha"}
    assert result.native is True
    assert result.metadata["provider"] == "gemini"


def test_gemini_call_action_returns_final_action_when_no_function_call():
    llm = _make_gemini_llm()

    mock_response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(function_call=None)]))],
        text="Here is the final answer.",
    )
    llm._client.models.generate_content.return_value = mock_response

    history = ConversationHistory()
    history.add_user("Answer the question")
    result = llm.call_action(history, tools={"lookup": DummyTool()})

    assert isinstance(result.action, FinalAction)
    assert result.action.content == "Here is the final answer."
    assert result.native is True
    assert result.metadata["provider"] == "gemini"


def test_gemini_call_action_raises_on_empty_content_and_no_function_call():
    llm = _make_gemini_llm()

    mock_response = SimpleNamespace(candidates=[], text="   ")
    llm._client.models.generate_content.return_value = mock_response

    history = ConversationHistory()
    history.add_user("No answer")

    with pytest.raises(ValueError, match="Gemini response did not contain text or a function call"):
        llm.call_action(history, tools={"lookup": DummyTool()})


@pytest.mark.asyncio
async def test_gemini_call_stream_yields_expected_text_chunks():
    llm = _make_gemini_llm()

    llm._client.models.generate_content_stream.return_value = [
        SimpleNamespace(text="Hello"),
        SimpleNamespace(text=" "),
        SimpleNamespace(text="from"),
        SimpleNamespace(text=" Gemini"),
        SimpleNamespace(text="!"),
    ]

    history = ConversationHistory()
    history.add_user("Stream please")

    chunks = [chunk async for chunk in llm.call_stream(history)]

    assert chunks == ["Hello", " ", "from", " Gemini", "!"]
    llm._client.models.generate_content_stream.assert_called_once_with(
        model="gemini-test",
        contents="Stream please",
        config={"temperature": 1.0, "top_p": 1.0},
    )


def test_gemini_tool_declarations_match_gemini_function_declarations():
    llm = _make_gemini_llm()

    mock_response = SimpleNamespace(candidates=[], text="Done")
    llm._client.models.generate_content.return_value = mock_response

    tools = {"lookup": DummyTool()}
    expected_declarations = gemini_function_declarations(tools, include_agent_tools=False)

    history = ConversationHistory()
    history.add_user("Test tool declarations")

    # When agent_callback_available is False
    llm.call_action(history, tools=tools, agent_callback_available=False)

    call_args = llm._client.models.generate_content.call_args
    passed_config = call_args.kwargs["config"]
    assert "tools" in passed_config
    assert len(passed_config["tools"]) == 1
    assert "function_declarations" in passed_config["tools"][0]

    actual_declarations = passed_config["tools"][0]["function_declarations"]
    assert actual_declarations == expected_declarations
    assert len(actual_declarations) == 1
    assert actual_declarations[0] == {
        "name": "lookup",
        "description": "Look up a value.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    }

    # When agent_callback_available is True and agent_cards are provided
    llm.call_action(
        history,
        tools=tools,
        agent_callback_available=True,
        agent_cards=[{"name": "worker_agent"}],
    )
    passed_config_with_agents = llm._client.models.generate_content.call_args.kwargs["config"]
    tool_names = {decl["name"] for decl in passed_config_with_agents["tools"][0]["function_declarations"]}
    assert "lookup" in tool_names
    assert AGENT_INFER_TOOL_NAME in tool_names
    assert AGENT_TOOL_CALL_TOOL_NAME in tool_names


@pytest.mark.asyncio
async def test_gemini_call_action_stream_collects_function_call():
    llm = _make_gemini_llm()

    function_call = SimpleNamespace(name="lookup", args={"key": "alpha"})
    part = SimpleNamespace(function_call=function_call)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)

    llm._client.models.generate_content_stream.return_value = [
        SimpleNamespace(candidates=[candidate], text=""),
    ]

    history = ConversationHistory()
    history.add_user("Find alpha streamed")
    result = await llm.call_action_stream(
        history,
        tools={"lookup": DummyTool()},
        agent_callback_available=False,
    )

    assert isinstance(result.action, ToolCallAction)
    assert result.action.tool == "lookup"
    assert result.action.args == {"key": "alpha"}
    assert result.native is True
    assert result.metadata["streaming"] is True
    assert result.metadata["provider"] == "gemini"


@pytest.mark.asyncio
async def test_gemini_call_action_stream_text_deltas_with_callback():
    llm = _make_gemini_llm()

    llm._client.models.generate_content_stream.return_value = [
        SimpleNamespace(candidates=[], text="Streaming"),
        SimpleNamespace(candidates=[], text=" response"),
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
    assert result.action.content == "Streaming response"
    assert result.native is True
    assert callback.await_count == 2
    callback.assert_any_await("Streaming")
    callback.assert_any_await(" response")


def test_gemini_properties():
    llm = _make_gemini_llm()
    assert llm.uses_native_action_prompt is True
    assert llm.supports_native_action_stream is True
