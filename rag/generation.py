import os
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the user's question using ONLY the provided "
    "context chunks. If the answer is not in the context, say "
    "'I don't have enough context to answer that.' Be concise and technical."
)


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
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = build_prompt(question, chunks)

    start = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.time() - start) * 1000)

    return response.content[0].text, {
        "model": MODEL,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_ms": latency_ms,
        "prompt": prompt,
    }
