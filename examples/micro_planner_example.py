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
    return [[0.0, 0.0, 0.0] for _ in texts]


async def dummy_vlm(prompt, system_prompt=None, messages=None):
    return "dummy vision response"


async def main() -> None:
    config = RAGAnythingConfig(enable_micro_planner=True, working_dir="./tmp_rag_storage")
    rag = RAGAnything(
        llm_model_func=dummy_llm,
        embedding_func=EmbeddingFunc(3, dummy_embed),
        vision_model_func=dummy_vlm,
        config=config,
    )
    await rag._ensure_lightrag_initialized()
    q1 = "Summarize the **document** content with figure ![alt](img.png) and $E=mc^2$."
    answer1 = await rag.aquery(q1)
    print("Answer 1:", answer1)
    q2 = "Compare values in table 1 and image 2."
    answer2 = await rag.aquery(q2)
    print("Answer 2:", answer2)


if __name__ == "__main__":
    asyncio.run(main())