#!/usr/bin/env python
"""
Simple example: use Ollama (qwen2.5:3b-instruct) with RAG-Anything

What it does:
- Loads configuration from .env (Ollama endpoints/models)
- Builds minimal chat + embedding functions for Ollama HTTP API
- Parses all documents in ./example_doc
- Runs a query against the built knowledge base and prints the answer

Prerequisites:
- Ollama running locally with models pulled:
  - LLM: qwen2.5:3b-instruct (default from .env LLM_MODEL)
  - Embedding: bge-m3:latest (default from .env EMBEDDING_MODEL)
  - Vision (for image processing): set env VISION_MODEL (e.g. qwen2.5vl:3b 或 llava:7b)
  Example:
    ollama pull qwen2.5:3b-instruct
    ollama pull bge-m3:latest
    ollama pull llava:7b  # or another VLM that supports images

Run:
  # optional: export VISION_MODEL=qwen2.5vl:3b
  python examples/example.py --query "用中文总结文档内容"
"""

import os
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Optional

# Ensure local imports resolve when running from repo root
import sys
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc, logger


load_dotenv(dotenv_path=".env", override=False)


def _http_post_json(url: str, payload: dict, timeout: int = 240) -> dict:
    """Minimal HTTP POST JSON helper using stdlib to avoid extra deps."""
    import urllib.request
    import urllib.error

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


def build_ollama_llm_func():
    """Create a chat completion function for Ollama /api/chat.

    Signature expected by RAGAnything examples:
      (prompt, system_prompt=None, history_messages=[], messages=None, **kwargs) -> str
    """
    host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct").strip()
    url = f"{host}/api/chat"

    async def llm_model_func(prompt: str,
                             system_prompt: Optional[str] = None,
                             history_messages: Optional[List[dict]] = None,
                             messages: Optional[List[dict]] = None,
                             **kwargs) -> str:
        # Prefer explicit messages if provided (for multimodal-style calls; here we only forward text)
        if messages:
            # Filter to text-only contents if any complex structures present
            norm_msgs = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    # Extract text parts only
                    text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                    content = "\n".join([t for t in text_parts if t])
                norm_msgs.append({"role": role, "content": content})
        else:
            norm_msgs = []
            if system_prompt:
                norm_msgs.append({"role": "system", "content": system_prompt})
            if history_messages:
                # Expect objects like {"role": "user"|"assistant", "content": str}
                for h in history_messages:
                    if "role" in h and "content" in h:
                        norm_msgs.append({"role": h["role"], "content": h["content"]})
            norm_msgs.append({"role": "user", "content": prompt or ""})

        payload = {
            "model": model,
            "messages": norm_msgs,
            "stream": False,
            # Map common params if provided
            **({"temperature": kwargs.get("temperature")} if kwargs.get("temperature") is not None else {}),
        }

        # Run blocking HTTP in thread to keep this function awaitable
        resp = await asyncio.to_thread(
            _http_post_json, url, payload, int(os.getenv("TIMEOUT", "240"))
        )
        # ollama chat response shape: {"message": {"role": "assistant", "content": "..."}, ...}
        msg = resp.get("message") or {}
        return msg.get("content", "")

    return llm_model_func


def build_ollama_vision_func():
    """Create a vision-capable function for Ollama using /api/generate with images.

    Looks for `VISION_MODEL` in env, falls back to `qwen2.5vl:3b`.
    If no images are provided, falls back to text-only chat via /api/chat.
    """
    host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434").rstrip("/")
    vlm_model = os.getenv("VISION_MODEL", "qwen2.5vl:3b").strip()
    gen_url = f"{host}/api/generate"
    chat_url = f"{host}/api/chat"

    async def vision_model_func(prompt: str,
                                system_prompt: Optional[str] = None,
                                history_messages: Optional[List[dict]] = None,
                                image_data: Optional[str] = None,
                                messages: Optional[List[dict]] = None,
                                **kwargs) -> str:
        # Prepare prompt and image list
        images: List[str] = []
        final_prompt_parts: List[str] = []

        if system_prompt:
            final_prompt_parts.append(system_prompt)

        if messages:
            # Flatten OpenAI-like message format into a single prompt and extract images
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            t = c.get("text", "")
                            if t:
                                final_prompt_parts.append(t)
                        elif isinstance(c, dict) and c.get("type") == "image_url":
                            url = (c.get("image_url") or {}).get("url", "")
                            if url.startswith("data:image") and "," in url:
                                b64 = url.split(",", 1)[1]
                                images.append(b64)
                else:
                    if isinstance(content, str) and content:
                        final_prompt_parts.append(content)
        else:
            if prompt:
                final_prompt_parts.append(prompt)
            if image_data:
                images.append(image_data)

        final_prompt = "\n".join([p for p in final_prompt_parts if p]) or "Describe the image."

        if images:
            payload = {
                "model": vlm_model,
                "prompt": final_prompt,
                "images": images,
                "stream": False,
            }
            resp = await asyncio.to_thread(
                _http_post_json, gen_url, payload, int(os.getenv("TIMEOUT", "240"))
            )
            return resp.get("response", "")
        else:
            # Fallback to text-only chat if no image present
            norm_msgs = []
            if system_prompt:
                norm_msgs.append({"role": "system", "content": system_prompt})
            if history_messages:
                for h in history_messages:
                    if "role" in h and "content" in h:
                        norm_msgs.append({"role": h["role"], "content": h["content"]})
            norm_msgs.append({"role": "user", "content": final_prompt})
            payload = {"model": os.getenv("LLM_MODEL", "qwen2.5:3b-instruct").strip(), "messages": norm_msgs, "stream": False}
            resp = await asyncio.to_thread(
                _http_post_json, chat_url, payload, int(os.getenv("TIMEOUT", "240"))
            )
            return (resp.get("message") or {}).get("content", "")

    return vision_model_func


def build_ollama_embedding_func() -> EmbeddingFunc:
    """Create an EmbeddingFunc bound to Ollama /api/embeddings."""
    host = os.getenv("EMBEDDING_BINDING_HOST", os.getenv("LLM_BINDING_HOST", "http://localhost:11434")).rstrip("/")
    model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest").strip()
    url = f"{host}/api/embeddings"

    async def embed_batch(texts: List[str]) -> List[List[float]]:
        async def one(t: str) -> List[float]:
            payload = {"model": model, "prompt": t}
            resp = await asyncio.to_thread(
                _http_post_json, url, payload, int(os.getenv("TIMEOUT", "240"))
            )
            emb = resp.get("embedding")
            if not emb:
                raise RuntimeError("No embedding returned from Ollama")
            return emb

        results = await asyncio.gather(*(one(t) for t in texts))
        return list(results)

    # Dimension is optional; if provided in .env use it, else leave None
    dim = os.getenv("EMBEDDING_DIM")
    embedding_dim = int(dim) if dim and dim.isdigit() else None

    return EmbeddingFunc(
        embedding_dim=embedding_dim or 1024,
        max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
        func=embed_batch,
    )


async def run(query: str, docs_dir: str = "example_doc", output_dir: str = "./output"):
    base_dir = Path(__file__).resolve().parent.parent

    # Resolve working_dir to repo root by default for consistency
    default_working_dir = str(base_dir / "rag_storage")
    working_dir = os.getenv("WORKING_DIR", default_working_dir)

    # Use a light config: text-first to avoid image/VLM requirements
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        enable_image_processing=True,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=build_ollama_llm_func(),
        vision_model_func=build_ollama_vision_func(),
        embedding_func=build_ollama_embedding_func(),
    )

    # Collect files under docs_dir (support running from repo root or examples/)
    docs_path = Path(docs_dir)
    if not docs_path.is_absolute():
        if docs_path.exists() and docs_path.is_dir():
            pass
        elif (base_dir / docs_dir).exists() and (base_dir / docs_dir).is_dir():
            docs_path = base_dir / docs_dir
        else:
            raise FileNotFoundError(
                f"Documents folder not found: {docs_dir}. Tried: '{Path.cwd()/docs_path}', '{base_dir/docs_dir}'"
            )

    files = [p for p in docs_path.iterdir() if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in {docs_dir}")

    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = base_dir / output_dir
    os.makedirs(out_path, exist_ok=True)
    logger.info(f"Using documents dir: {docs_path}")
    logger.info(f"Using output dir: {out_path}")
    logger.info(f"Using working dir: {working_dir}")

    # Process each document
    for f in files:
        logger.info(f"Processing document: {f}")
        await rag.process_document_complete(
            file_path=str(f),
            output_dir=str(out_path),
            parse_method=config.parse_method,
        )

    # Ask
    logger.info(f"Query: {query}")
    answer = await rag.aquery(query, mode="hybrid")
    print("\n===== ANSWER =====\n" + str(answer) + "\n===================\n")


def main():
    parser = argparse.ArgumentParser(description="RAG-Anything + Ollama example")
    parser.add_argument("--query", "-q", default="用中文总结文档的主要内容", help="Query to ask")
    parser.add_argument("--docs", default="example_doc", help="Folder of documents to ingest")
    parser.add_argument("--output", default="./output", help="Output directory for artifacts")
    args = parser.parse_args()

    asyncio.run(run(args.query, args.docs, args.output))


if __name__ == "__main__":
    main()
