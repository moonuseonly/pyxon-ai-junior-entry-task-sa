"""Answer generation from retrieved chunks.

Pluggable by design: set GROQ_API_KEY (or extend `_call_llm`) to generate a
real synthesized answer via an LLM. With no key configured, falls back to an
extractive answer (the top-scoring chunk(s), verbatim) so the API is fully
usable and demoable without requiring anyone to hand you a credential.
"""

import os

from app.rag.retriever import RetrievedChunk

_SYSTEM_PROMPT = (
    "You answer questions using only the provided context. If the context "
    "doesn't contain the answer, say so plainly instead of guessing. Answer "
    "in the same language as the question."
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        source = c.metadata.get("filename", "unknown source")
        parts.append(f"[{i}] (source: {source})\n{c.text}")
    return "\n\n".join(parts)


def _call_llm(question: str, context: str) -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    from groq import Groq  # imported lazily so this stays optional

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> dict:
    if not chunks:
        return {"answer": "No relevant content was found for this question.", "mode": "none"}

    context = _build_context(chunks)
    llm_answer = _call_llm(question, context)
    if llm_answer is not None:
        return {"answer": llm_answer, "mode": "llm"}

    # Extractive fallback — no API key configured.
    top = chunks[0]
    source = top.metadata.get("filename", "unknown source")
    return {
        "answer": top.text,
        "mode": "extractive_fallback",
        "note": (
            f"Returned verbatim from the top-matching chunk (source: {source}). "
            "Set GROQ_API_KEY to enable LLM-synthesized answers."
        ),
    }
