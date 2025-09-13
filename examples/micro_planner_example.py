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
    return [[0.0] * 3 for _ in texts]

async def dummy_vlm(prompt, system_prompt=None, messages=None):
    return "dummy vision response"


class DummyLightRAG:
    async def aquery(self, query, param=None):
        return "dummy response"

    async def finalize_storages(self):
        return None

async def main() -> None:
    config = RAGAnythingConfig(enable_micro_planner=True)
    rag = RAGAnything(
        llm_model_func=dummy_llm,
        embedding_func=EmbeddingFunc(3, dummy_embed),
        vision_model_func=dummy_vlm,
        config=config,
    )
    rag.lightrag = DummyLightRAG()

    async def _noop():
        return {"success": True}

    rag._ensure_lightrag_initialized = _noop

    answer1 = await rag.aquery("Summarize the document content.")
    print("Answer 1:", answer1)
    answer2 = await rag.aquery("What does figure 2 show?")
    print("Answer 2:", answer2)


if __name__ == "__main__":
    asyncio.run(main())