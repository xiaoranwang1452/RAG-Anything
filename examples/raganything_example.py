#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process documents with RAGAnything using MinerU parser
2. Perform pure text queries using aquery() method
3. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
4. Handle different types of multimodal content (tables, equations) in queries

Updates:
- Loads model configuration (LLM, embeddings, vision) directly from environment/.env
- Ingests every supported file inside a directory (default: ./example_doc) for multimodal processing
"""

import os
import argparse
import asyncio
import json
import logging
import logging.config
from pathlib import Path
import numpy as np

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.azure_openai import azure_openai_complete_if_cache, azure_openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def _env_flag(name: str, default: str = "false") -> bool:
    """Read boolean feature flags from environment."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, separator: str = ",") -> list[str]:
    value = os.getenv(name, "")
    return [item.strip().lower() for item in value.split(separator) if item.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _azure_api_kwargs() -> dict[str, str]:
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
    return {"api_version": api_version} if api_version else {}


def _http_post_json(url: str, payload: dict, timeout: int = 240) -> dict:
    """Minimal helper to POST JSON without external deps."""
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} error from {url}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e}")


def _build_ollama_embedding_func() -> EmbeddingFunc:
    """Create an embedding function that talks to Ollama via HTTP."""

    host = os.getenv("EMBEDDING_BINDING_HOST", os.getenv("LLM_BINDING_HOST", "http://localhost:11434")).rstrip("/")
    model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest").strip()
    url = f"{host}/api/embeddings"
    timeout = _env_int("TIMEOUT", 240)
    max_concurrency = max(1, _env_int("EMBEDDING_MAX_CONCURRENCY", 4))
    embed_semaphore = asyncio.Semaphore(max_concurrency)

    async def embed_batch(texts):
        async def one(text):
            payload = {"model": model, "prompt": text}
            async with embed_semaphore:
                resp = await asyncio.to_thread(_http_post_json, url, payload, timeout)
            embedding = resp.get("embedding")
            if not embedding:
                raise RuntimeError("No embedding returned from Ollama")
            return embedding

        if not texts:
            return []
        return list(await asyncio.gather(*(one(t) for t in texts)))

    dim = os.getenv("EMBEDDING_DIM")
    embedding_dim = int(dim) if dim and dim.isdigit() else None

    return EmbeddingFunc(
        embedding_dim=embedding_dim or 1024,
        max_token_size=_env_int("MAX_EMBED_TOKENS", 8192),
        func=embed_batch,
    )


DEFAULT_LLM_MODEL = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("LLM_MODEL", "gpt-4o-mini")
)
DEFAULT_VISION_MODEL = (
    os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT")
    or os.getenv("VISION_MODEL")
    or "gpt-4o"
)
DEFAULT_EMBEDDING_MODEL = (
    os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
    or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
)
DEFAULT_EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 3072)
DEFAULT_COMPLETION_MAX_TOKENS = _env_int("LLM_COMPLETION_MAX_TOKENS", 512)


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    input_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    Process document with RAGAnything

    Args:
        input_path: Path to a document or directory with documents
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or os.getenv("WORKING_DIR", "./rag_storage"),
            parser=parser,  # Parser selection: mineru or docling
            parse_method="auto",  # Parse method: auto, ocr, or txt
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            enable_micro_planner=True,
        )

        # Define LLM model function
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            kwargs = dict(kwargs)
            kwargs.setdefault("max_tokens", DEFAULT_COMPLETION_MAX_TOKENS)
            kwargs.setdefault("temperature", _env_float("TEMPERATURE", 0.2))
            azure_kwargs = _azure_api_kwargs()
            return await azure_openai_complete_if_cache(
                DEFAULT_LLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **azure_kwargs,
                **kwargs,
            )

        # Define vision model function for image processing
        async def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # If messages format is provided (for multimodal VLM enhanced query), use it directly
            if messages:
                return await azure_openai_complete_if_cache(
                    DEFAULT_VISION_MODEL,
                    None,
                    system_prompt=None,
                    history_messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **_azure_api_kwargs(),
                    **kwargs,
                )
            # Traditional single image format
            elif image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        (
                            {"role": "system", "content": system_prompt}
                            if system_prompt
                            else None
                        ),
                        (
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        },
                                    },
                                ],
                            }
                            if image_data
                            else {"role": "user", "content": prompt}
                        ),
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **_azure_api_kwargs(),
                    **kwargs,
                )
            # Pure text format
            else:
                return await llm_model_func(
                    prompt, system_prompt, history_messages, **kwargs
                )

        # Define embedding function
        embedding_binding = os.getenv("EMBEDDING_BINDING", "").strip().lower()
        if embedding_binding == "ollama":
            embedding_func = _build_ollama_embedding_func()
        else:
            embedding_func = EmbeddingFunc(
                embedding_dim=DEFAULT_EMBEDDING_DIM,
                max_token_size=_env_int("MAX_EMBED_TOKENS", 8192),
                func=lambda texts: azure_openai_embed(
                    texts,
                    model=DEFAULT_EMBEDDING_MODEL,
                    api_key=api_key,
                    base_url=base_url,
                    **_azure_api_kwargs(),
                ),
            )


        # Define rerank model function using embedding similarity
        async def rerank_model_func(
            query: str, documents: list[str], top_n: int | None = None, **kwargs
        ) -> list[dict[str, float]]:
            texts = [query] + documents
            embeddings = await embedding_func(texts)
            query_vec = np.array(embeddings[0])
            doc_vecs = [np.array(e) for e in embeddings[1:]]
            scores = [float(np.dot(doc_vec, query_vec)) for doc_vec in doc_vecs]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            if top_n is not None:
                ranked = ranked[:top_n]
            return [
                {"index": i, "relevance_score": scores[i]} for i in ranked
            ]

        # Define rerank model function using embedding similarity
        async def rerank_model_func(
            query: str, documents: list[str], top_n: int | None = None, **kwargs
        ) -> list[dict[str, float]]:
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
            return [
                {"index": i, "relevance_score": scores[i]} for i in ranked
            ]

        # Initialize RAGAnything with new dataclass structure
        rag = RAGAnything(
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            rerank_model_func=rerank_model_func, 
            config=config,
        )


        # Process document
        await rag.process_document_complete(
            file_path=file_path,
            output_dir=output_dir,
            parse_method="auto",
            device="cpu",
        )

        # Example queries - demonstrating different query approaches
        logger.info("\nQuerying processed document:")

        # 1. Pure text queries using aquery()
        text_queries = [
            "What is the main content of the document?",
            "What are the key topics discussed?",
        ]

        for query in text_queries:
            logger.info(f"\n[Text Query]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"Answer: {result}")
            await asyncio.sleep(2)

        # # 2. Multimodal query with specific multimodal content using aquery_with_multimodal()
        # logger.info(
        #     "\n[Multimodal Query]: Analyzing performance data in context of document"
        # )
        # multimodal_result = await rag.aquery_with_multimodal(
        #     "Compare this performance data with any similar results mentioned in the document",
        #     multimodal_content=[
        #         {
        #             "type": "table",
        #             "table_data": """Method,Accuracy,Processing_Time
        #                         RAGAnything,95.2%,120ms
        #                         Traditional_RAG,87.3%,180ms
        #                         Baseline,82.1%,200ms""",
        #             "table_caption": "Performance comparison results",
        #         }
        #     ],
        #     mode="hybrid",
        # )
        # logger.info(f"Answer: {multimodal_result}")

        # # 3. Another multimodal query with equation content
        # logger.info("\n[Multimodal Query]: Mathematical formula analysis")
        # equation_result = await rag.aquery_with_multimodal(
        #     "Explain this formula and relate it to any mathematical concepts in the document",
        #     multimodal_content=[
        #         {
        #             "type": "equation",
        #             "latex": "F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}",
        #             "equation_caption": "F1-score calculation formula",
        #         }
        #     ],
        #     mode="hybrid",
        # )
        # logger.info(f"Answer: {equation_result}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    parser.add_argument(
        "--input",
        "-i",
        default=os.getenv("INPUT_DIR", "example_doc"),
        help="Path to a document file or directory to process",
    )
    parser.add_argument(
        "--working_dir",
        "-w",
        default=os.getenv("WORKING_DIR", "./rag_storage"),
        help="Working directory path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.getenv("OUTPUT_DIR", "./output"),
        help="Output directory path",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Optional base URL for API",
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_with_rag(
            args.input,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
