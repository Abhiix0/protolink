from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from protolink.llms.api.hugging_face_client import HuggingFaceLLM
from protolink.llms.history import ConversationHistory


def _make_hugging_face_llm() -> HuggingFaceLLM:
    llm = HuggingFaceLLM.__new__(HuggingFaceLLM)
    llm.model = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm._model_params = {"temperature": 0.7}
    llm.history = ConversationHistory()
    llm.system_prompt = ""
    llm._client = MagicMock()
    return llm


def test_hugging_face_call_sends_messages_list_not_raw_string():
    llm = _make_hugging_face_llm()

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Response text")
            )
        ]
    )
    llm._client.chat_completion.return_value = mock_response

    history = ConversationHistory()
    history.add_system("You are a helpful assistant.")
    history.add_user("Hello, world!")

    llm.call(history)

    llm._client.chat_completion.assert_called_once()
    args, kwargs = llm._client.chat_completion.call_args

    messages = args[0]
    assert isinstance(messages, list)
    assert not isinstance(messages, str)
    assert messages == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"},
    ]
    assert kwargs["model"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert kwargs["temperature"] == 0.7


def test_hugging_face_call_extracts_choices_message_content():
    llm = _make_hugging_face_llm()

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Extracted content from choices")
            )
        ]
    )
    llm._client.chat_completion.return_value = mock_response

    history = ConversationHistory()
    history.add_user("Generate content")

    result = llm.call(history)
    assert result == "Extracted content from choices"


def test_hugging_face_call_defensive_fallback():
    llm = _make_hugging_face_llm()
    llm._client.chat_completion.return_value = "unexpected raw string"

    history = ConversationHistory()
    history.add_user("Test fallback")

    result = llm.call(history)
    assert result == "unexpected raw string"


def test_hugging_face_call_raises_value_error_on_stop_iteration():
    llm = _make_hugging_face_llm()
    llm._client.chat_completion.side_effect = StopIteration("mapping failure")

    history = ConversationHistory()
    history.add_user("Trigger StopIteration")

    with pytest.raises(
        ValueError,
        match="HuggingFace API provider mapping failed for model 'meta-llama/Meta-Llama-3-8B-Instruct'",
    ):
        llm.call(history)


@pytest.mark.asyncio
async def test_hugging_face_call_stream_yields_expected_chunks():
    llm = _make_hugging_face_llm()
    llm._client.chat_completion.return_value = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content="Hello"))
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content=" "))
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content="streaming"))
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content="!"))
            ]
        ),
    ]

    history = ConversationHistory()
    history.add_user("Stream please")

    chunks = [chunk async for chunk in llm.call_stream(history)]
    assert chunks == ["Hello", " ", "streaming", "!"]


@pytest.mark.asyncio
async def test_hugging_face_call_stream_passes_messages_and_stream_true():
    llm = _make_hugging_face_llm()
    llm._client.chat_completion.return_value = []

    history = ConversationHistory()
    history.add_system("System instructions")
    history.add_user("User message")

    _ = [chunk async for chunk in llm.call_stream(history)]

    llm._client.chat_completion.assert_called_once()
    args, kwargs = llm._client.chat_completion.call_args

    messages = args[0]
    assert isinstance(messages, list)
    assert not isinstance(messages, str)
    assert messages == [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "User message"},
    ]
    assert kwargs["model"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert kwargs["stream"] is True
    assert kwargs["temperature"] == 0.7
