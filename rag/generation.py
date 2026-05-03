import os
import time

import anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the user's question using ONLY the provided "
    "context chunks. If the answer is not in the context, say "
    "'I don't have enough context to answer that.' Be concise and technical."
)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def build_prompt(question: str, chunks: list[dict]) -> str:
    if not chunks:
        context = "No context available."
    else:
        context = "\n\n".join(
            f"[{i + 1}] (source: {c['filename']}, chunk {c['chunk_index']})\n{c['content']}"
            for i, c in enumerate(chunks)
        )
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, dict]:
    client = _get_client()
    prompt = build_prompt(question, chunks)

    start = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.time() - start) * 1000)

    text_blocks = [b for b in response.content if hasattr(b, "text")]
    if not text_blocks:
        raise ValueError(f"No text content in Claude response: {response.content}")
    return text_blocks[0].text, {
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_ms": latency_ms,
        "prompt": prompt,
    }
