#!/usr/bin/env python3
"""
RAGAnything Reflection Example (ENV-driven, LOCAL models)
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import List
import json
from typing import List, Union

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from raganything.config import ReflectionConfig
from ollama import Client as OllamaClient

print("√ Using raganything from:", Path(__file__).resolve())

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


def setup_env_driven_functions():
    """Build LLM / (fallback-only) VLM / Embedding funcs entirely from .env."""
    import json
    from typing import List, Union
    from dotenv import load_dotenv
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc
    from ollama import Client as OllamaClient

    load_dotenv(dotenv_path=".env", override=False)

    # ---- Core text LLM ----
    LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:11434/v1")
    if not LLM_HOST.rstrip("/").endswith("/v1"):
        LLM_HOST = LLM_HOST.rstrip("/") + "/v1"
    LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "local")

    # ---- Embedding via Ollama SDK ----
    EMB_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://127.0.0.1:11434")
    EMB_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
    # 先用 env，随后用实际探测覆盖
    EMB_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
    ollama = OllamaClient(host=EMB_HOST)

    # 同步小函数（避免阻塞事件循环）
    def _embed_sync(text: str) -> List[float]:
        return ollama.embeddings(model=EMB_MODEL, prompt=text)["embedding"]

    # 宽容地把任意输入转成字符串
    def _to_str(x) -> str:
        if isinstance(x, (list, tuple)):
            return " ".join(str(t) for t in x if t is not None)
        if isinstance(x, dict):
            import json as _json
            return _json.dumps(x, ensure_ascii=False)
        return str(x)

    # 先探测一次真实维度（避免 1024/2048 不匹配）
    try:
        _probe = _embed_sync("hello world")
        EMB_DIM = len(_probe)  # 用真实维度覆盖
        print(f"[embed] Detected embedding dim = {EMB_DIM} from model '{EMB_MODEL}'")
    except Exception as e:
        print(f"[embed] Probe failed, keep env EMBEDDING_DIM={EMB_DIM}: {e}")

    import asyncio

    async def embed_func(text_or_texts: Union[str, List[str], tuple, dict]) -> Union[List[float], List[List[float]]]:
        # 单条：返回 List[float]；批量：返回 List[List[float]]
        if isinstance(text_or_texts, (list, tuple)):
            # 批量：逐条转字符串并并发计算
            items = [ _to_str(t) for t in text_or_texts ]
            # 这里用线程池并发，避免阻塞；也可以分批限制并发
            def _call_many():
                return [ _embed_sync(it) for it in items ]
            return await asyncio.to_thread(_call_many)
        else:
            text = _to_str(text_or_texts)
            return await asyncio.to_thread(_embed_sync, text)

    embedding_func = EmbeddingFunc(
        embedding_dim=EMB_DIM,
        max_token_size=8192,
        func=embed_func,  # 关键：既支持单条也支持批量
    )

    # ---- Inference params ----
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

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
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    runtime = {
        "llm_host": LLM_HOST,
        "llm_model": LLM_MODEL,
        "embed_host": EMB_HOST,
        "embed_model": EMB_MODEL,
        "embed_dim": EMB_DIM,
        "vision_model": "(fallback to text)",
    }
    return llm_model_func, vision_model_func, embedding_func, runtime


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

    document_path = Path(args.document_path)
    if not document_path.exists():
        print(f" Document not found: {document_path}")
        sys.exit(1)

    PARSER = os.getenv("PARSER", "mineru")
    PARSE_METHOD = os.getenv("PARSE_METHOD", "auto")
    ENABLE_TABLE = False
    ENABLE_EQUATION = False
    ENABLE_IMAGE = _getenv_bool("ENABLE_IMAGE_PROCESSING", False)

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
        llm_model_func, vision_model_func, embedding_func, runtime = setup_env_driven_functions()
        print("🔧 Runtime backends:")
        print(f"   LLM     : {runtime['llm_model']} @ {runtime['llm_host']}")
        print(f"   Embedding: {runtime['embed_model']} (dim={runtime['embed_dim']}) @ {runtime['embed_host']}")
        print(f"   Vision  : {runtime['vision_model']}")

        reflection_cfg_kwargs = dict(
            enable_reflection=REF_ENABLED,
            reflection_top_k=REF_TOPK,
            reflection_temperature=REF_TEMP,
            reflection_support_threshold=REF_MIN_SUPPORT,
            max_iters=REF_MAX_ITERS,
            min_support=REF_MIN_SUPPORT,
            min_coverage=REF_MIN_COVERAGE,
            max_contradiction=REF_MAX_CONTRA,
            min_attributable=REF_MIN_ATTR,
        )

        reflection_config = ReflectionConfig(**reflection_cfg_kwargs)

        config = RAGAnythingConfig(
            working_dir=args.working_dir,
            parser=PARSER,
            parse_method=PARSE_METHOD,
            enable_image_processing=ENABLE_IMAGE,
            enable_table_processing=ENABLE_TABLE,
            enable_equation_processing=ENABLE_EQUATION,
            reflection=reflection_config,
        )

        print(" Initializing RAGAnything...")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

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

        result = await rag.answer_with_reflection(
            question=args.question,
            mode=args.query_mode,
        )

        print("\n ORIGINAL QUESTION:")
        print(result["question"])

        print("\n DRAFT ANSWER:")
        print(result["draft"])

        print("\n FINAL ANSWER (WITH CITATIONS):")
        print(result["final"])

        print("\n REFLECTION ANALYSIS:")
        report = result.get("reflection_report", {}) or {}
        print(json.dumps(report, indent=2, ensure_ascii=False))

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
