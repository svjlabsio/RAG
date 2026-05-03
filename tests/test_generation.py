from rag.generation import build_prompt


def test_build_prompt_includes_question():
    chunks = [{"content": "RAG combines retrieval and generation.", "filename": "doc.txt", "chunk_index": 0}]
    prompt = build_prompt("What is RAG?", chunks)
    assert "What is RAG?" in prompt
    assert "RAG combines retrieval and generation" in prompt


def test_build_prompt_numbers_chunks():
    chunks = [
        {"content": "First chunk.", "filename": "a.txt", "chunk_index": 0},
        {"content": "Second chunk.", "filename": "b.txt", "chunk_index": 1},
    ]
    prompt = build_prompt("test", chunks)
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert prompt.index("[1]") < prompt.index("[2]")


def test_build_prompt_no_chunks():
    prompt = build_prompt("Any question?", [])
    assert "Any question?" in prompt
    assert "No context" in prompt


def test_generate_answer_calls_claude(mocker, reset_generation_client):
    import rag.generation as gen_module

    mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    mock_anthropic = mocker.patch("rag.generation.anthropic.Anthropic")
    mock_response = mocker.MagicMock()
    mock_block = mocker.MagicMock()
    mock_block.type = "text"
    mock_block.text = "The answer is 42."
    mock_response.content = [mock_block]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 10
    mock_response.model = gen_module.MODEL
    mock_anthropic.return_value.messages.create.return_value = mock_response

    chunks = [{"content": "The answer is 42.", "filename": "test.txt", "chunk_index": 0}]
    answer, meta = gen_module.generate_answer("What is the answer?", chunks)

    assert answer == "The answer is 42."
    assert meta["input_tokens"] == 100
    assert meta["output_tokens"] == 10
    assert meta["model"] == gen_module.MODEL
    assert "latency_ms" in meta
    assert "prompt" in meta

    mock_anthropic.return_value.messages.create.assert_called_once_with(
        model=gen_module.MODEL,
        max_tokens=1024,
        system=gen_module.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": mocker.ANY}],
    )


def test_generate_answer_raises_on_no_text_blocks(mocker, reset_generation_client):
    import rag.generation as gen_module
    import pytest

    mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    mock_anthropic = mocker.patch("rag.generation.anthropic.Anthropic")
    mock_response = mocker.MagicMock()
    tool_block = mocker.MagicMock()
    tool_block.type = "tool_use"
    mock_response.content = [tool_block]
    mock_anthropic.return_value.messages.create.return_value = mock_response

    with pytest.raises(ValueError, match="No text content"):
        gen_module.generate_answer("q", [{"content": "x", "filename": "f.txt", "chunk_index": 0}])
