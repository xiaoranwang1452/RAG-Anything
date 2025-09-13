"""Example demonstrating micro planner usage."""

import asyncio
from pathlib import Path
import sys

# Allow running without package installation
sys.path.append(str(Path(__file__).parent.parent))

from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig


async def dummy_llm(prompt, **kwargs):
    return "dummy response"

async def dummy_embed(texts):
    return [[0.0] * 1024 for _ in texts]


async def main() -> None:
    config = RAGAnythingConfig(enable_micro_planner=True)
    rag = RAGAnything(
        llm_model_func=dummy_llm,
        embedding_func = EmbeddingFunc(1024, dummy_embed),
        config=config,
    )
    await rag._ensure_lightrag_initialized()
    await rag.aquery("Summarize the document content.")
    await rag.aquery("What does figure 2 show?")


if __name__ == "__main__":
    asyncio.run(main())