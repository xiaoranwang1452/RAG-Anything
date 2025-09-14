#!/usr/bin/env python
"""Utility script for testing Micro Planner behavior on a document."""

import argparse
import asyncio
import os
from pathlib import Path
import numpy as np

import sys

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Simple console logging."""
    logger.setLevel("INFO")
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def run(file_path: str, working_dir: str, output_dir: str, api_key: str, base_url: str, parser: str):
    """Process document and run sample queries to inspect Micro Planner output."""

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        enable_micro_planner=True,
    )

    # Async LLM and VLM wrappers
    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        history_messages = history_messages or []
        return await openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    async def vision_model_func(prompt, system_prompt=None, history_messages=None, image_data=None, messages=None, **kwargs):
        history_messages = history_messages or []
        if messages:
            return await openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        if image_data:
            return await openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        ],
                    }
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    def rerank_model_func(query: str, documents: list[str], top_n: int | None = None, **kwargs):
        texts = [query] + documents
        embeddings = openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        )
        query_vec = np.array(embeddings[0])
        doc_vecs = [np.array(e) for e in embeddings[1:]]
        scores = [float(np.dot(doc_vec, query_vec)) for doc_vec in doc_vecs]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        return [{"index": i, "relevance_score": scores[i]} for i in ranked]

    rag = RAGAnything(
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        config=config,
        lightrag_kwargs={"rerank_model_func": rerank_model_func},
    )

    if rag.micro_planner:
        rag.micro_planner.evaluator_func = None

    await rag.process_document_complete(
        file_path=file_path,
        output_dir=output_dir,
        parse_method="auto",
        device="cpu",
    )

    # Example queries touching different capabilities
    queries = [
        "Explain what are Figure 2 and 3 doing.",
        "Summarise ‘RAG-Sequence’ vs ‘RAG-Token’ in 3 bullet points each. Quote the defining lines for both, with citations.",
        "Find the figure that explains training/inference differences (if any). If none exists, say so and fall back to the most relevant paragraph. Always cite.",
        "Find the table reporting FEVER results. Return the 2-way and 3-way accuracies, and briefly explain the difference between the two tasks, with citations.",
        "Extract the caption of Figure 2 verbatim. If OCR/parse fails, fall back to the nearest textual caption or paragraph that describes it, and explicitly state the fallback.",
    ]

    for q in queries:
        normalized, intent, plan = rag.micro_planner.plan(q)
        logger.info("\nQuery: %s", q)
        logger.info("Planner intent: %s, tags: %s, confidence: %.2f", intent.intent, intent.tags, intent.confidence)
        logger.info("Planner plan: mode=%s, top_k=%d, rerank_top_k=%d, use_vlm=%s", plan.retrieval_mode, plan.top_k, plan.rerank_top_k, plan.use_vlm)
        result = await rag.aquery(q, mode=plan.retrieval_mode)
        eval_res = rag.micro_planner.evaluate(q, result, {})
        logger.info("Answer: %s", result)
        logger.info("Eval score: %.2f reason: %s", eval_res.get("score", 0.0), eval_res.get("degrade_reason"))


def main():
    parser = argparse.ArgumentParser(description="Micro Planner evaluation helper")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument("--working_dir", "-w", default="./rag_storage", help="Working directory path")
    parser.add_argument("--output", "-o", default="./output", help="Output directory path")
    parser.add_argument("--api-key", default=os.getenv("LLM_BINDING_API_KEY"), help="OpenAI API key")
    parser.add_argument("--base-url", default=os.getenv("LLM_BINDING_HOST"), help="Optional base URL")
    parser.add_argument("--parser", default=os.getenv("PARSER", "mineru"), help="Parser backend")
    args = parser.parse_args()

    if not args.api_key:
        logger.error("OpenAI API key is required")
        return

    os.makedirs(args.output, exist_ok=True)

    asyncio.run(run(args.file_path, args.working_dir, args.output, args.api_key, args.base_url, args.parser))


if __name__ == "__main__":
    configure_logging()
    main()