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


def test_generate_answer_calls_claude(mocker):
    import rag.generation as gen_module
    from rag.generation import generate_answer, MODEL, SYSTEM_PROMPT

    # Reset the singleton so the mock is picked up
    gen_module._client = None

    mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    mock_anthropic = mocker.patch("rag.generation.anthropic.Anthropic")
    mock_response = mocker.MagicMock()
    mock_response.content = [mocker.MagicMock(text="The answer is 42.")]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 10
    mock_response.model = MODEL
    mock_anthropic.return_value.messages.create.return_value = mock_response

    chunks = [{"content": "The answer is 42.", "filename": "test.txt", "chunk_index": 0}]
    answer, meta = generate_answer("What is the answer?", chunks)

    assert answer == "The answer is 42."
    assert meta["input_tokens"] == 100
    assert meta["output_tokens"] == 10
    assert meta["model"] == MODEL
    assert "latency_ms" in meta
    assert "prompt" in meta

    mock_anthropic.return_value.messages.create.assert_called_once_with(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": mocker.ANY}],
    )

    # Clean up singleton so other tests are not affected
    gen_module._client = None
