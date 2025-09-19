#!/usr/bin/env python3
"""
RAGAnything Reflection Example (ENV-driven, LOCAL models)

This example demonstrates the Reflection Layer with all runtime settings
read from .env so you can run against local OpenAI-compatible backends
(e.g., Ollama/vLLM/LM Studio) without touching the code.

.env keys (examples):
  LLM_BINDING_HOST=http://127.0.0.1:11434
  LLM_MODEL=mistral

  EMBEDDING_BINDING_HOST=http://127.0.0.1:11434
  EMBEDDING_MODEL=bge-m3:latest
  EMBEDDING_DIM=1024

  VISION_MODEL=qwen2.5vl:3b
  ENABLE_IMAGE_PROCESSING=true
  VLM_ENHANCED=true
  VLM_MAX_CONCURRENCY=1
  VLM_MAX_IMAGES=2
  MAX_IMAGE_SIDE=384
  IMAGE_JPEG_QUALITY=75
  VLM_RETRIES=2
  VLM_DISABLE_ON_FAILURE=true
  VLM_FALLBACK_TEXT="[Image omitted due to resource limits]"

  WORKING_DIR=./rag_storage
  PARSER=mineru
  PARSE_METHOD=auto

  MAX_TOKENS=2048
  NUM_PREDICT=256
  TEMPERATURE=0
  TIMEOUT=240
  VLM_TIMEOUT=300

  # Reflection thresholds (new)
  REFLECTION_ENABLED=true
  REFLECTION_MAX_ITERS=2
  REFLECTION_MIN_SUPPORT=0.70
  REFLECTION_MAX_CONTRADICTION=0.10
  REFLECTION_MIN_COVERAGE=0.85
  REFLECTION_MIN_ATTRIBUTABLE=0.80
  REFLECTION_TOPK=6
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from raganything.config import ReflectionConfig
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import raganything
print("√ Using raganything from:", Path(raganything.__file__).resolve())


# ---------- helpers ----------
def _getenv_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _getenv_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _getenv_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

from ollama import Client as OllamaClient  # NEW

def setup_env_driven_functions():
    """Build LLM / (fallback-only) VLM / Embedding funcs entirely from .env."""

    load_dotenv(dotenv_path=".env", override=False)

    # ---- Core text LLM ----
    LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:11434/v1")
    # 规范化，确保以 /v1 结尾（OpenAI-兼容端点）
    if not LLM_HOST.rstrip("/").endswith("/v1"):
        LLM_HOST = LLM_HOST.rstrip("/") + "/v1"

    LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "local")  # dummy for local backends

    # ---- Embedding via Ollama SDK (no /v1/embeddings) ----
    EMB_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://127.0.0.1:11434")
    EMB_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
    EMB_DIM   = _getenv_int("EMBEDDING_DIM", 1024)
    ollama = OllamaClient(host=EMB_HOST)

    # ---- Inference params ----
    MAX_TOKENS   = _getenv_int("MAX_TOKENS", 2048)
    TEMPERATURE  = _getenv_float("TEMPERATURE", 0.0)

    # ---- Image/VLM switches（先统一回退到文本）----
    ENABLE_IMAGE = _getenv_bool("ENABLE_IMAGE_PROCESSING", False)

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        history_messages = history_messages or []
        return openai_complete_if_cache(
            LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=LLM_API_KEY,
            base_url=LLM_HOST,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            # 这里不要传 request_timeout，OpenAI SDK v1 不接受该参数
            **kwargs,
        )

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        """
        Multimodal path stub:
        当前本地后端无稳定多模态 OpenAI 兼容实现，统一回退为文本 LLM。
        未来接入真 VLM 时再替换此函数的实现。
        """
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=EMB_DIM,
        max_token_size=8192,
        func=lambda texts: [
            ollama.embeddings(model=EMB_MODEL, prompt=t)["embedding"]
            for t in texts
        ],
    )

    runtime = {
        "llm_host": LLM_HOST,
        "llm_model": LLM_MODEL,
        "embed_host": EMB_HOST,
        "embed_model": EMB_MODEL,
        "embed_dim": EMB_DIM,
        "vision_model": "(fallback to text)",
    }
    return llm_model_func, vision_model_func, embedding_func, runtime


# def setup_env_driven_functions():
#     """Build LLM / VLM / Embedding funcs entirely from .env."""

#     # Load .env once here to allow running without CLI flags
#     load_dotenv(dotenv_path=".env", override=False)

#     # Core text LLM
#     LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:11434")
#     LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
#     LLM_API_KEY = os.getenv("LLM_API_KEY", "local")  # dummy ok for local backends

#     # Embedding
#     EMB_HOST = os.getenv("EMBEDDING_BINDING_HOST", LLM_HOST)
#     EMB_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
#     EMB_DIM = _getenv_int("EMBEDDING_DIM", 1024)
#     EMB_API_KEY = os.getenv("EMBEDDING_API_KEY", "local")

#     # Inference params
#     MAX_TOKENS = _getenv_int("MAX_TOKENS", 2048)
#     TEMPERATURE = _getenv_float("TEMPERATURE", 0.0)
#     TIMEOUT = _getenv_int("TIMEOUT", 240)

#     # Vision
#     ENABLE_IMAGE = _getenv_bool("ENABLE_IMAGE_PROCESSING", False)
#     VISION_MODEL = os.getenv("VISION_MODEL", "")
#     VLM_TIMEOUT = _getenv_int("VLM_TIMEOUT", 300)
#     VLM_DISABLE_ON_FAILURE = _getenv_bool("VLM_DISABLE_ON_FAILURE", True)
#     VLM_FALLBACK_TEXT = os.getenv("VLM_FALLBACK_TEXT", "[image omitted]")

#     # Build funcs
#     def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
#         # ensure list
#         history_messages = history_messages or []
#         return openai_complete_if_cache(
#             LLM_MODEL,
#             prompt,
#             system_prompt=system_prompt,
#             history_messages=history_messages,
#             api_key=LLM_API_KEY,
#             base_url=LLM_HOST,
#             temperature=TEMPERATURE,
#             max_tokens=MAX_TOKENS,
#             # request_timeout=TIMEOUT,
#             **kwargs,
#         )

#     def vision_model_func(
#         prompt,
#         system_prompt=None,
#         history_messages=None,
#         image_data=None,
#         messages=None,
#         **kwargs,
#     ):
#         """Multimodal path. If disabled or unavailable, gracefully fall back."""
#         if not ENABLE_IMAGE or not VISION_MODEL:
#             # Fallback to plain text LLM
#             return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

#         # build messages cleanly (no None items)
#         msgs = []
#         if system_prompt:
#             msgs.append({"role": "system", "content": system_prompt})

#         if messages:
#             msgs.extend(messages)
#         elif image_data:
#             msgs.append(
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},},
#                     ],
#                 }
#             )
#         else:
#             msgs.append({"role": "user", "content": prompt})

#         try:
#             return openai_complete_if_cache(
#                 VISION_MODEL,
#                 "",
#                 messages=msgs,
#                 api_key=LLM_API_KEY,   # most local VLMs share same host/key
#                 base_url=LLM_HOST,
#                 request_timeout=VLM_TIMEOUT,
#                 **kwargs,
#             )
#         except Exception:
#             if VLM_DISABLE_ON_FAILURE:
#                 return VLM_FALLBACK_TEXT
#             raise

#     embedding_func = EmbeddingFunc(
#         embedding_dim=EMB_DIM,
#         max_token_size=8192,
#         func=lambda texts: openai_embed(
#             texts,
#             model=EMB_MODEL,
#             api_key=EMB_API_KEY,
#             base_url=EMB_HOST,
#         ),
#     )

#     # Expose also the runtime we read (for logging)
#     runtime = {
#         "llm_host": LLM_HOST,
#         "llm_model": LLM_MODEL,
#         "embed_host": EMB_HOST,
#         "embed_model": EMB_MODEL,
#         "embed_dim": EMB_DIM,
#         "vision_model": VISION_MODEL if ENABLE_IMAGE else "(disabled)",
#     }
#     return llm_model_func, vision_model_func, embedding_func, runtime


async def demonstrate_reflection_layer():
    """Demonstrate the reflection layer functionality using ENV-driven config."""

    load_dotenv(dotenv_path=".env", override=False)

    parser = argparse.ArgumentParser(description="RAGAnything Reflection Example (ENV-driven)")
    parser.add_argument("document_path", help="Path to document to process")
    parser.add_argument("--question",
                        default="What are the main findings and conclusions presented in this document?",
                        help="Question to ask about the document")
    parser.add_argument("--working-dir", default=os.getenv("WORKING_DIR", "./rag_storage"),
                        help="Override working dir (else use WORKING_DIR from .env)")
    parser.add_argument("--output-dir", default="./output_reflection",
                        help="Output directory for parsed content")
    parser.add_argument("--query-mode", default=os.getenv("QUERY_MODE", "mix"),
                        help="LightRAG query mode (default: mix)")
    args = parser.parse_args()

    # Validate document exists
    document_path = Path(args.document_path)
    if not document_path.exists():
        print(f" Document not found: {document_path}")
        sys.exit(1)

    # Runtime from env
    PARSER = os.getenv("PARSER", "mineru")
    PARSE_METHOD = os.getenv("PARSE_METHOD", "auto")
    ENABLE_TABLE = True
    ENABLE_EQUATION = True
    ENABLE_IMAGE = _getenv_bool("ENABLE_IMAGE_PROCESSING", False)

    # Reflection thresholds from env (two naming styles supported)
    REF_ENABLED = _getenv_bool("REFLECTION_ENABLED", True)
    REF_MAX_ITERS = _getenv_int("REFLECTION_MAX_ITERS", 2)
    REF_TOPK = _getenv_int("REFLECTION_TOPK", 6)
    REF_MIN_SUPPORT = _getenv_float("REFLECTION_MIN_SUPPORT", 0.70)
    REF_MIN_COVERAGE = _getenv_float("REFLECTION_MIN_COVERAGE", 0.85)
    REF_MAX_CONTRA = _getenv_float("REFLECTION_MAX_CONTRADICTION", 0.10)
    REF_MIN_ATTR = _getenv_float("REFLECTION_MIN_ATTRIBUTABLE", 0.80)
    REF_TEMP = _getenv_float("REFLECTION_TEMPERATURE", 0.1)

    print(" RAGAnything Reflection Layer (ENV-driven)")
    print("=" * 60)
    print(f" Document: {document_path}")
    print(f" Question: {args.question}")
    print(f" Parser/Method: {PARSER} / {PARSE_METHOD}")
    print(f" Reflection enabled: {REF_ENABLED}  |  mode={args.query_mode}")
    print()

    try:
        # Setup model functions from ENV
        llm_model_func, vision_model_func, embedding_func, runtime = setup_env_driven_functions()
        print(" Runtime backends:")
        print(f"   LLM     : {runtime['llm_model']} @ {runtime['llm_host']}")
        print(f"   Embedding: {runtime['embed_model']} (dim={runtime['embed_dim']}) @ {runtime['embed_host']}")
        print(f"   Vision  : {runtime['vision_model']}")

        # Build ReflectionConfig using only the fields accepted by the dataclass
        # 注意：不要传 enabled / targeted_topk，因为它们在 dataclass 里是 init=False
        reflection_cfg_kwargs = dict(
            # 老名字（dataclass 接受）
            enable_reflection=REF_ENABLED,
            reflection_top_k=REF_TOPK,
            reflection_temperature=REF_TEMP,
            reflection_support_threshold=REF_MIN_SUPPORT,

            # 新增字段（init=True）
            max_iters=REF_MAX_ITERS,
            min_support=REF_MIN_SUPPORT,
            min_coverage=REF_MIN_COVERAGE,
            max_contradiction=REF_MAX_CONTRA,
            min_attributable=REF_MIN_ATTR,
        )

        reflection_config = ReflectionConfig(**reflection_cfg_kwargs)

        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=args.working_dir,
            parser=PARSER,
            parse_method=PARSE_METHOD,
            enable_image_processing=ENABLE_IMAGE,
            enable_table_processing=ENABLE_TABLE,
            enable_equation_processing=ENABLE_EQUATION,
            reflection=reflection_config,
        )

        # Initialize RAGAnything
        print(" Initializing RAGAnything...")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process document
        print(" Processing document...")
        await rag.process_document_complete(
            file_path=str(document_path),
            output_dir=args.output_dir,
            parse_method=PARSE_METHOD,
            display_stats=True,
        )

        print("\n" + "=" * 60)
        print(" GENERATING ANSWER WITH REFLECTION")
        print("=" * 60)

        # Generate answer with reflection
        result = await rag.answer_with_reflection(
            question=args.question,
            mode=args.query_mode,
        )

        # Display results
        print("\n ORIGINAL QUESTION:")
        print("-" * 30)
        print(result["question"])

        print("\n DRAFT ANSWER:")
        print("-" * 30)
        print(result["draft"])

        print("\n FINAL ANSWER (WITH CITATIONS):")
        print("-" * 30)
        print(result["final"])

        print("\n REFLECTION ANALYSIS:")
        print("-" * 30)
        report = result.get("reflection_report", {}) or {}

        # Print core metrics if present (new schema)
        core_keys = [
            ("global_support", " Support"),
            ("coverage", " Coverage"),
            ("contradiction_ratio", " Contradiction Ratio"),
            ("attributable", " Attribution"),
        ]
        for k, label in core_keys:
            if k in report:
                try:
                    print(f"{label}: {float(report[k]):.2f}")
                except Exception:
                    print(f"{label}: {report[k]}")

        if "targeted_queries" in report:
            tq = report["targeted_queries"]
            print(f" Targeted Queries Used: {len(tq)}")
            for i, q in enumerate(tq[:3], 1):
                print(f"   {i}. {q[:80]}{'...' if len(q) > 80 else ''}")

        # Save detailed results to file
        out_file = Path(args.working_dir) / "reflection_results.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n Detailed results saved to: {out_file}")

    except Exception as e:
        print(f" Error during reflection example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    asyncio.run(demonstrate_reflection_layer())


if __name__ == "__main__":
    main()
