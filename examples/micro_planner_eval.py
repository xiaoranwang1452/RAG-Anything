#!/usr/bin/env python
"""Utility script for testing Micro Planner behavior on a document."""

import argparse
import asyncio
import os
from pathlib import Path
import math
from collections import Counter

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
        enable_micro_planner=(os.getenv("ENABLE_MICRO_PLANNER","true").lower() in ("1","true","yes")),
    )

    # Async LLM and VLM wrappers
    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        history_messages = history_messages or []
        _safe_llm_keys = {"temperature", "max_tokens", "presence_penalty", "frequency_penalty", "stop", "top_p", "logprobs"}  # all chat-safe
        _llm_kwargs = {k: v for k, v in kwargs.items() if k in _safe_llm_keys}

        return await openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **_llm_kwargs
        )

    async def vision_model_func(prompt, system_prompt=None, history_messages=None, image_data=None, messages=None, **kwargs):
        history_messages = history_messages or []
        if messages:
            _safe_llm_keys = {"temperature", "max_tokens", "presence_penalty", "frequency_penalty", "stop", "top_p", "logprobs"}
            _llm_kwargs = {k: v for k, v in kwargs.items() if k in _safe_llm_keys}

            return await openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **_llm_kwargs,
            )
        if image_data:
            _safe_llm_keys = {"temperature", "max_tokens", "presence_penalty", "frequency_penalty", "stop", "top_p", "logprobs"}
            _llm_kwargs = {k: v for k, v in kwargs.items() if k in _safe_llm_keys}

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
                **_llm_kwargs,
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

    def rerank_model_func(
        query: str, documents: list[str], top_n: int | None = None, **kwargs
    ) -> list[dict[str, float]]:
        """Simple lexical cosine similarity reranker."""

        def tf_counter(text: str) -> Counter:
            return Counter(text.lower().split())

        query_tf = tf_counter(query)
        results = []
        for idx, doc in enumerate(documents):
            doc_tf = tf_counter(doc)
            vocab = set(query_tf) | set(doc_tf)
            q_vec = [query_tf.get(t, 0) for t in vocab]
            d_vec = [doc_tf.get(t, 0) for t in vocab]
            dot = sum(q * d for q, d in zip(q_vec, d_vec))
            q_norm = math.sqrt(sum(q * q for q in q_vec))
            d_norm = math.sqrt(sum(d * d for d in d_vec))
            score = dot / (q_norm * d_norm + 1e-8)
            results.append({"index": idx, "relevance_score": score})

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results

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
        "Tell me what are the insights from Figure 1.",
        "Summarise Table 1 findings",
        "Find the figure that explains training/inference differences (if any). If none exists, say so and fall back to the most relevant paragraph. Always cite.",
        "Find the table reporting FEVER results. Return the 2-way and 3-way accuracies, and briefly explain the difference between the two tasks, with citations.",
        "Extract the caption of Figure 2 verbatim. If OCR/parse fails, fall back to the nearest textual caption or paragraph that describes it, and explicitly state the fallback.",
    ]

    for q in queries:
        logger.info("\nQuery: %s", q)
        if rag.micro_planner:
            normalized, intent, plan = rag.micro_planner.plan(q)
            logger.info("Planner intent: %s, tags: %s, confidence: %.2f", intent.intent, intent.tags, intent.confidence)
            logger.info(
                "Planner plan: mode=%s, top_k=%d, rerank_top_k=%d, use_vlm=%s",
                plan.retrieval_mode,
                plan.top_k,
                plan.rerank_top_k,
                plan.use_vlm,
            )
            result = await rag.aquery(q, mode=plan.retrieval_mode)
            eval_res = rag.micro_planner.evaluate(q, result, {})
            logger.info("Answer: %s", result)
            logger.info("Eval score: %.2f reason: %s", eval_res.get("score", 0.0), eval_res.get("degrade_reason"))
        else:
            result = await rag.aquery(q)
            logger.info("Answer: %s", result)


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
