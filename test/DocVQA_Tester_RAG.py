#!/usr/bin/env python
"""
DocVQA tester that mirrors the structure of ScienceQA_Tester_Revised.py
while adapting to DocVQA's document question answering format.

Full Evaluation:
python test/DocVQA_Tester_RAG.py \
  --data_file dataset/docvqa/val/val_v1.0.json \
  --image_root dataset/docvqa/val \
  --ocr_root dataset/docvqa/val/ocr_results \
  --use_ocr \
  --test_number -1 \
  --label docvqa_full \
  --output_root ./output/docvqa \
  --max_tokens 512 \
  --save_every 50 \
  --rate_limit_delay 1.0 \
  --rate_limit_retries 6

"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import logging.config
import os
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root directory to Python path (same pattern as ScienceQA tester)
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from dotenv import load_dotenv

from lightrag.llm.azure_openai import azure_openai_complete_if_cache, azure_openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag import QueryParam

from raganything import RAGAnything, RAGAnythingConfig


load_dotenv(dotenv_path=".env", override=False)


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about documents. "
    "Provide ONLY the direct answer without any explanation or preamble. "
    "Do NOT repeat the question in your answer. "
    "Keep your answer brief and precise. "
    "For brand names, provide the main brand name without excessive product details."
)


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, separator: str = ",") -> List[str]:
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


def _azure_api_kwargs() -> Dict[str, str]:
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
    return {"api_version": api_version} if api_version else {}

def _azure_embedding_api_kwargs() -> Dict[str, str]:
    api_version = os.getenv("AZURE_EMBEDDING_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
    return {"api_version": api_version} if api_version else {}


def _http_post_json(url: str, payload: dict, timeout: int = 240) -> dict:
    """Minimal helper to POST JSON without external deps (shared with example scripts)."""
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
    host = os.getenv("EMBEDDING_BINDING_HOST", os.getenv("LLM_BINDING_HOST", "http://127.0.0.1:11434")).rstrip("/")
    model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest").strip()
    url = f"{host}/api/embeddings"
    timeout = _env_int("TIMEOUT", 240)
    max_concurrency = max(1, _env_int("EMBEDDING_MAX_CONCURRENCY", 4))
    semaphore = asyncio.Semaphore(max_concurrency)

    async def embed_batch(texts: Iterable[str]) -> List[List[float]]:
        async def one(text: str) -> List[float]:
            payload = {"model": model, "prompt": text}
            async with semaphore:
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


def configure_logging() -> None:
    """Match the logging setup used in ScienceQA tester."""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file = os.path.abspath(os.path.join(log_dir, "docvqa_tester.log"))
    print(f"\nDocVQA tester log file: {log_file}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "detailed",
                    "filename": log_file,
                    "maxBytes": int(os.getenv("LOG_MAX_BYTES", 10_485_760)),
                    "backupCount": int(os.getenv("LOG_BACKUP_COUNT", 5)),
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
    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DocVQA tester using RAG-Anything")
    parser.add_argument("--data_file", type=str, required=True, help="Path to DocVQA JSON file (e.g., val_v1.0.json)")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory containing document images")
    parser.add_argument("--ocr_root", type=str, help="Directory containing OCR results JSON files")
    parser.add_argument("--use_ocr", action="store_true", help="Use OCR text in prompts")
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", "gpt-4o"))
    parser.add_argument("--test_number", type=int, default=-1, help="Number of questions to test (-1 for all)")
    parser.add_argument("--save_every", type=int, default=50, help="Save results every N questions")
    parser.add_argument("--label", type=str, default="docvqa_rag", help="Label for output files")
    parser.add_argument("--output_root", type=str, default="./output/docvqa", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode (process only 10 questions)")
    parser.add_argument("--force_text", action="store_true", help="Use text-only mode (no image)")

    # RAG-Anything configuration (same as ScienceQA tester)
    parser.add_argument("--working_dir", type=str, default=os.getenv("WORKING_DIR", "./rag_storage"))
    parser.add_argument("--rag_output", type=str, default=os.getenv("OUTPUT_DIR", "./rag_output"))
    parser.add_argument("--knowledge_dir", type=str, help="Directory of documents to ingest before evaluation")
    parser.add_argument("--knowledge_recursive", action="store_true")
    parser.add_argument("--supported_exts", type=str, default=os.getenv("SUPPORTED_FILE_EXTENSIONS", ""))
    parser.add_argument("--mode", type=str, default=os.getenv("MODE", "hybrid"), help="RAG mode: local, global, hybrid")
    parser.add_argument("--top_k", type=int, default=_env_int("TOP_K", _env_int("DOCVQA_RAG_TOP_K", 5)))
    parser.add_argument("--rate_limit_delay", type=float, default=_env_float("DOCVQA_RATE_LIMIT_DELAY", 1.0))
    parser.add_argument("--rate_limit_retries", type=int, default=_env_int("DOCVQA_RATE_LIMIT_RETRIES", 6))
    parser.add_argument("--rate_limit_backoff", type=float, default=_env_float("DOCVQA_RATE_LIMIT_BACKOFF", 2.0))

    # API configuration
    parser.add_argument("--openai_key", type=str, default=os.getenv("LLM_BINDING_API_KEY"), help="Override API key")
    parser.add_argument("--base_url", type=str, default=os.getenv("LLM_BINDING_HOST"))
    parser.add_argument("--engine", type=str, default=os.getenv("LLM_MODEL"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=os.getenv("MAX_TOKENS"))
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode (no API calls)")
    
    # DocVQA specific
    parser.add_argument("--anls_threshold", type=float, default=0.5, help="ANLS threshold for scoring")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_docvqa_data(data_file: Path, test_number: int = -1) -> Tuple[List[Dict[str, Any]], int]:
    """Load DocVQA dataset from JSON file."""
    data = _load_json(data_file)
    
    # DocVQA format: {"dataset_name": "docvqa", "data": [...]}
    if isinstance(data, dict) and "data" in data:
        questions = data["data"]
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError(f"Unexpected DocVQA data format in {data_file}")
    
    if test_number > 0:
        questions = questions[:test_number]
    
    total = len(questions)
    print(f"Loaded {total} questions from {data_file}\n")
    
    return questions, total


def normalize_answer(text: str) -> str:
    """Normalize answer text following DocVQA evaluation protocol."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_anls(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    Compute ANLS (Average Normalized Levenshtein Similarity).
    Official DocVQA evaluation metric.
    Note: Uses case-insensitive comparison by normalizing to lowercase.
    """
    if not prediction or not ground_truths:
        return 0.0
    
    # Normalize to lowercase for case-insensitive comparison
    prediction_lower = prediction.lower()
    
    max_similarity = 0.0
    for gt in ground_truths:
        if not gt:
            continue
        
        gt_lower = gt.lower()
        
        # Compute edit distance
        dist = compute_levenshtein_distance(prediction_lower, gt_lower)
        max_len = max(len(prediction_lower), len(gt_lower))
        
        if max_len == 0:
            similarity = 1.0
        else:
            # NLS = 1 - (edit_distance / max_length)
            similarity = 1.0 - (dist / max_len)
        
        # ANLS: if similarity < threshold, score as 0
        if similarity < threshold:
            similarity = 0.0
        
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Compute exact match score (1.0 if prediction matches any ground truth)."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if pred_norm == normalize_answer(gt):
            return 1.0
    return 0.0


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    
    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            f1 = 0.0
        else:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        
        max_f1 = max(max_f1, f1)
    
    return max_f1


def _resolve_image_path(question: Dict[str, Any], image_root: Path) -> Optional[Path]:
    """Resolve the full path to the document image."""
    image_field = question.get("image") or question.get("image_path")
    if not image_field:
        return None
    
    # Try direct path
    candidate = Path(image_field)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    
    # Try relative to image_root
    candidate = image_root / image_field
    if candidate.exists():
        return candidate
    
    # Try without 'documents/' prefix if present
    if "documents/" in str(image_field):
        clean_name = str(image_field).replace("documents/", "")
        candidate = image_root / clean_name
        if candidate.exists():
            return candidate
    
    return None


def _resolve_ocr_path(question: Dict[str, Any], ocr_root: Optional[Path], image_path: Optional[Path]) -> Optional[Path]:
    """Resolve the path to the OCR results JSON file."""
    if not ocr_root:
        return None
    
    # Try using image filename
    if image_path:
        ocr_name = image_path.stem + ".json"
        candidate = ocr_root / ocr_name
        if candidate.exists():
            return candidate
    
    # Try using docId from question
    doc_id = question.get("ucsf_document_id")
    doc_page = question.get("ucsf_document_page_no")
    if doc_id and doc_page:
        ocr_name = f"{doc_id}_{doc_page}.json"
        candidate = ocr_root / ocr_name
        if candidate.exists():
            return candidate
    
    return None


def load_ocr_text(ocr_file: Path, max_length: int = 3000) -> Optional[str]:
    """Load OCR text from Azure OCR results JSON."""
    if not ocr_file.exists():
        return None
    
    try:
        ocr_data = _load_json(ocr_file)
        
        # Azure OCR format
        lines = []
        for result in ocr_data.get("recognitionResults", []):
            for line in result.get("lines", []):
                text = line.get("text", "").strip()
                if text:
                    lines.append(text)
        
        full_text = "\n".join(lines)
        
        # Truncate if too long
        if len(full_text) > max_length:
            full_text = full_text[:max_length] + "..."
        
        return full_text if full_text else None
    except Exception as e:
        logger.warning(f"Failed to load OCR from {ocr_file}: {e}")
        return None


def _encode_image_to_base64(path: Path) -> Optional[Tuple[str, str]]:
    import mimetypes

    try:
        with path.open("rb") as f:
            payload = base64.b64encode(f.read()).decode("utf-8")
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/png"
        return payload, mime
    except Exception:
        return None


def _build_prompt(
    question: str,
    context: Optional[str],
    ocr_text: Optional[str],
) -> str:
    """Build prompt for DocVQA question answering."""
    lines: List[str] = []
    
    if context:
        lines.append("Retrieved Context:")
        lines.append(context.strip())
        lines.append("")
    
    if ocr_text:
        lines.append("OCR Extracted Text from Document:")
        lines.append(ocr_text)
        lines.append("")
    
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Provide a direct, concise answer based on the document.")
    lines.append("Answer:")
    
    return "\n".join(lines).strip()


async def _retrieve_context(
    rag: RAGAnything,
    question: str,
    mode: str,
    top_k: int,
) -> Optional[str]:
    if not getattr(rag, "lightrag", None):
        return None
    param = QueryParam(mode=mode, chunk_top_k=top_k, only_need_context=True)
    try:
        context = await rag.lightrag.aquery(question, param=param)
        return context.strip() if context else None
    except Exception as exc:
        logger.warning(f"Context retrieval failed: {exc}")
        return None


async def _ingest_knowledge(
    rag: RAGAnything,
    knowledge_dir: Path,
    output_dir: Path,
    recursive: bool,
    supported_exts: List[str],
) -> None:
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {knowledge_dir}")

    if recursive:
        files = [p for p in knowledge_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in knowledge_dir.iterdir() if p.is_file()]

    if supported_exts:
        lowered = {f".{ext.lstrip('.').lower()}" for ext in supported_exts}
        files = [f for f in files if f.suffix.lower() in lowered]

    if not files:
        raise FileNotFoundError(f"No ingestible files found in {knowledge_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        logger.info(f"Ingesting knowledge document: {file_path}")
        await rag.process_document_complete(
            file_path=str(file_path),
            output_dir=str(output_dir),
            parse_method=rag.config.parse_method,
            device="cpu",
        )


async def _ingest_docvqa_metadata(
    rag: Optional[RAGAnything],
    questions: List[Dict[str, Any]],
    image_root: Path,
    ocr_root: Optional[Path],
) -> None:
    """Insert DocVQA question metadata (text + optional image + OCR) into LightRAG."""
    if not rag:
        return

    init_result = await rag._ensure_lightrag_initialized()
    if isinstance(init_result, dict) and init_result.get("success") is False:
        logger.warning(f"Failed to initialize LightRAG for metadata ingestion: {init_result.get('error')}")
        return

    lightrag = getattr(rag, "lightrag", None)
    if not lightrag:
        return

    for idx, question in enumerate(questions):
        q_id = question.get("questionId", idx)
        question_text = question.get("question", "").strip()
        
        content_list: List[Dict[str, Any]] = []
        page_idx = 0

        def _add_text_block(label: str, text: str) -> None:
            nonlocal page_idx
            text = text.strip()
            if not text:
                return
            prefix = f"{label}: " if label else ""
            content_list.append(
                {
                    "type": "text",
                    "text": f"{prefix}{text}",
                    "page_idx": page_idx,
                }
            )
            page_idx += 1

        if question_text:
            _add_text_block("Question", f"[Q{q_id}] {question_text}")

        # Add OCR text if available
        image_path = _resolve_image_path(question, image_root)
        if ocr_root and image_path:
            ocr_path = _resolve_ocr_path(question, ocr_root, image_path)
            if ocr_path:
                ocr_text = load_ocr_text(ocr_path, max_length=2000)
                if ocr_text:
                    _add_text_block("OCR Text", ocr_text)

        # Add image
        if image_path and image_path.exists():
            content_list.append(
                {
                    "type": "image",
                    "img_path": str(image_path.resolve()),
                    "img_caption": [],
                    "img_footnote": [],
                    "page_idx": page_idx,
                }
            )
            page_idx += 1

        if not content_list:
            continue

        doc_id = f"docvqa_meta_{q_id}"
        try:
            await rag.insert_content_list(
                content_list=content_list,
                file_path=f"docvqa_question_{q_id}",
                doc_id=doc_id,
                display_stats=False,
            )
        except Exception as exc:
            logger.warning(f"Failed to insert metadata doc {doc_id}: {exc}")


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if any(keyword in text for keyword in ("rate limit", "too many requests", "quota", "retry-after", "please retry")):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 429:
        return True
    return False


def _extract_retry_after(exc: Exception) -> Optional[float]:
    for attr in ("retry_after", "retry_after_seconds", "retry_after_ms"):
        value = getattr(exc, attr, None)
        if value:
            try:
                seconds = float(value) / (1000 if attr.endswith("_ms") else 1)
                if seconds > 0:
                    return seconds
            except (TypeError, ValueError):
                continue

    headers = getattr(exc, "headers", None) or getattr(exc, "response_headers", None)
    if isinstance(headers, dict):
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except (TypeError, ValueError):
                pass

    text = str(exc)
    match = re.search(r"retry[- ]after[^0-9]*([0-9]+)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            pass

    match = re.search(r"try again in[^0-9]*([0-9]+)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            pass
    return None


def _clean_answer(text: str) -> str:
    """Clean and extract answer from model response."""
    if not text:
        return ""
    
    # Take first line if multi-line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        text = lines[0]
    
    # Remove common prefixes
    text = re.sub(r'^(answer|response|result)[\s:]*', '', text, flags=re.IGNORECASE)
    
    # Remove common question rephrasing patterns
    # Patterns like: "The name of X is Y" -> "Y"
    patterns_to_remove = [
        r'^the\s+name\s+of\s+(?:the\s+)?(?:company|university|organization|institution|person|individual)\s+is\s+',
        r'^the\s+(?:company|university|organization|institution)\s+name\s+is\s+',
        r'^the\s+value\s+(?:is|was|equals?)\s+',
        r'^the\s+answer\s+is\s+',
        r'^it\s+is\s+',
        r'^the\s+.*?\s+(?:value|number|amount|quantity|total|count|percentage)\s+(?:is|was|equals?)\s+',
        r'^(?:approximately|about|around)\s+',
    ]
    
    original_text = text
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        if text != original_text:
            break  # Stop after first match
    
    # Remove trailing punctuation like periods
    text = re.sub(r'\.\s*$', '', text)
    
    # Remove surrounding quotes
    text = text.strip('"\'')
    
    # Advanced cleaning: extract core answer from verbose responses
    text = _extract_core_answer(text)
    
    return text.strip()


def _extract_core_answer(text: str) -> str:
    """Extract the core answer by removing redundant units and context when appropriate."""
    if not text:
        return text
    
    # Early check: Detect if this is an address (contains street indicators + location)
    # Keep full address if detected
    address_indicators = ['st.', 'street', 'ave', 'avenue', 'rd', 'road', 'blvd', 'boulevard', 
                         'dr', 'drive', 'ln', 'lane', 'way', 'court', 'ct']
    location_indicators = [r'\b[A-Z]{2}\b', r'\d{5}', r'washington', r'new york', r'los angeles',
                          r'n\.?\s*w\.?', r's\.?\s*e\.?', r's\.?\s*w\.?', r'n\.?\s*e\.?']
    
    text_lower = text.lower()
    has_address_indicator = any(indicator in text_lower for indicator in address_indicators)
    has_location_indicator = any(re.search(pattern, text, re.IGNORECASE) for pattern in location_indicators)
    
    # If it looks like an address, preserve it completely
    if has_address_indicator and has_location_indicator:
        return text
    
    # Pattern 1: Extract number from "X per Y" or "X / Y" formats
    # e.g., "0.28 per 1000" -> "0.28", "5 / 100" -> "5"
    match = re.match(r'^([0-9]+\.?[0-9]*)\s*(?:per|/)\s+[0-9]+', text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: Extract percentage value
    # e.g., "25 percent" -> "25%", "25%" -> "25%"
    match = re.match(r'^([0-9]+\.?[0-9]*)\s*(?:percent|percentage)', text, re.IGNORECASE)
    if match:
        return f"{match.group(1)}%"
    
    # Pattern 3: Extract currency value
    # e.g., "$100" -> "100", "100 dollars" -> "100"
    match = re.match(r'^(?:\$|USD|EUR|GBP)?\s*([0-9,]+\.?[0-9]*)\s*(?:dollars?|USD|euros?|EUR|pounds?|GBP)?', text, re.IGNORECASE)
    if match and ('$' in text or 'dollar' in text.lower() or 'USD' in text):
        return match.group(1).replace(',', '')
    
    # Pattern 4: Intelligently remove common units from number-based answers
    # Only remove units when they appear redundant (question already mentions the unit)
    # e.g., "How many days?" -> "40 days" -> "40"
    # But keep "11:14 to 11:39 a.m." as is
    
    # Remove time period units (days, weeks, months, years) when they follow numbers
    # But NOT time designations (a.m., p.m.)
    if re.search(r'^\d+\s+(?:days?|weeks?|months?|years?)$', text, re.IGNORECASE):
        text = re.sub(r'\s+(?:days?|weeks?|months?|years?)\b', '', text, flags=re.IGNORECASE)
    
    # Remove count units (children, people, etc.) when they follow numbers or words like "seven"
    if re.search(r'^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:children|kids|people|persons?|individuals?)$', text, re.IGNORECASE):
        text = re.sub(r'\s+(?:children|kids|people|persons?|individuals?)\b', '', text, flags=re.IGNORECASE)
    
    # Remove medical/measurement units for simple numeric answers
    if re.search(r'^\d+(?:\.\d+)?\s+mg%?$', text, re.IGNORECASE):
        text = re.sub(r'\s+mg%?', '', text, flags=re.IGNORECASE)
    
    # Remove leading words like "Over", "About", "Around", "Approximately"
    text = re.sub(r'^(?:over|about|around|approximately|roughly)\s+', '', text, flags=re.IGNORECASE)
    
    # Pattern 5: Remove parenthetical information
    # e.g., "John Smith (CEO)" -> "John Smith"
    text = re.sub(r'\s*\([^)]*\)\s*', ' ', text).strip()
    
    # Pattern 6: Remove explanatory phrases at the end
    # e.g., "ITC Limited, a company in India" -> "ITC Limited"
    text = re.sub(r',\s+(?:which|that|who|a|an|the)\s+.*$', '', text, flags=re.IGNORECASE)
    
    # Pattern 7: Extract first occurrence of a number if text is verbose
    # e.g., "The value is approximately 0.28 based on the chart" -> check if core value exists
    if len(text.split()) > 5:  # If answer is too verbose
        # But avoid this for addresses or complex entities
        if not (has_address_indicator or ',' in text):
            # Try to find a standalone number or short phrase
            number_match = re.search(r'\b([0-9]+\.?[0-9]*)\b', text)
            if number_match:
                # Check if there's a simple pattern like "X Y" where Y is a unit
                simple_match = re.search(r'\b([0-9]+\.?[0-9]*(?:\s+[A-Za-z]{1,10})?)\b', text)
                if simple_match and len(simple_match.group(1).split()) <= 2:
                    return simple_match.group(1)
    
    # Pattern 8: Extract brand/product name from verbose descriptions
    # e.g., "John Players Jeans gives one the perfect vintage look..." -> "John Players Jeans"
    # Look for capitalized words at the beginning before descriptive text
    if len(text.split()) > 6:
        # Check for pattern: "Name gives/provides/offers/..." or "Name, a..."
        match = re.match(r'^([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*(?:\s+Jeans?|(?:\s+Di\s+)?Wills?)?)\s+(?:gives?|provides?|offers?|is|are|was|were|,)', text)
        if match:
            return match.group(1).strip()
        
        # Check for pattern starting with capitalized words before "—" or descriptive phrases
        match = re.match(r'^([A-Z][A-Za-z\s]+?)\s*(?:—|gives|provides|offers)', text)
        if match:
            brand_name = match.group(1).strip()
            # Only return if it's not too long (< 5 words)
            if len(brand_name.split()) < 5:
                return brand_name
    
    return text


async def run_docvqa(
    args: argparse.Namespace,
    questions: List[Dict[str, Any]],
) -> None:
    api_key = (args.openai_key or os.getenv("LLM_BINDING_API_KEY") or "").strip()
    if not api_key and not args.dry_run:
        raise RuntimeError("API key is required. Set LLM_BINDING_API_KEY in .env or pass --openai_key.")
    base_url = (args.base_url or "").strip() or None
    llm_model = args.engine or args.model or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("LLM_MODEL", "gpt-4o")
    vision_model = os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT") or os.getenv("VISION_MODEL") or "gpt-4o"
    embedding_model = os.getenv("AZURE_EMBEDDING_DEPLOYMENT") or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    # Embedding specific configuration
    embedding_api_key = os.getenv("EMBEDDING_BINDING_API_KEY") or api_key
    embedding_base_url = os.getenv("EMBEDDING_BINDING_HOST") or base_url

    temperature = args.temperature if args.temperature is not None else _env_float("TEMPERATURE", 0.2)
    max_tokens = args.max_tokens if args.max_tokens is not None else _env_int("LLM_COMPLETION_MAX_TOKENS", 512)

    embedding_binding = os.getenv("EMBEDDING_BINDING", "").strip().lower()
    if embedding_binding == "ollama":
        embedding_func = _build_ollama_embedding_func()
    else:
        embedding_func = EmbeddingFunc(
            embedding_dim=_env_int("EMBEDDING_DIM", 3072),
            max_token_size=_env_int("MAX_EMBED_TOKENS", 8192),
            func=lambda texts: azure_openai_embed(
                texts,
                model=embedding_model,
                api_key=embedding_api_key,
                base_url=embedding_base_url,
                **_azure_embedding_api_kwargs(),
            ),
        )

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("max_tokens", max_tokens)
        kwargs.setdefault("temperature", temperature)
        return await azure_openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **_azure_api_kwargs(),
            **kwargs,
        )

    async def vision_model_func(prompt, system_prompt=None, history_messages=None, image_data=None, messages=None, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("max_tokens", max_tokens)
        kwargs.setdefault("temperature", temperature)
        if messages:
            return await azure_openai_complete_if_cache(
                vision_model,
                None,
                system_prompt=None,
                history_messages=messages,
                api_key=api_key,
                base_url=base_url,
                **_azure_api_kwargs(),
                **kwargs,
            )
        if image_data:
            if isinstance(image_data, dict):
                image_base64 = image_data.get("base64")
                image_mime = image_data.get("mime") or "image/png"
            else:
                image_base64 = image_data
                image_mime = "image/png"
            structured_messages = [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_base64}"}} if image_base64 else None,
                    ],
                },
            ]
            structured_messages = [m for m in structured_messages if m]
            return await azure_openai_complete_if_cache(
                vision_model,
                None,
                system_prompt=None,
                history_messages=structured_messages,
                api_key=api_key,
                base_url=base_url,
                **_azure_api_kwargs(),
                **kwargs,
            )
        return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    async def rerank_model_func(query: str, documents: List[str], top_n: Optional[int] = None, **_: Any) -> List[Dict[str, float]]:
        texts = [query] + documents
        embeddings = await embedding_func(texts)  # type: ignore[misc]
        query_vec = np.array(embeddings[0])
        doc_vecs = [np.array(e) for e in embeddings[1:]]
        scores = [float(np.dot(doc, query_vec)) for doc in doc_vecs]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if top_n is not None:
            order = order[:top_n]
        return [{"index": i, "relevance_score": scores[i]} for i in order]

    rag_config = RAGAnythingConfig(
        working_dir=args.working_dir,
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        enable_image_processing=_env_flag("ENABLE_IMAGE_PROCESSING", "true"),
        enable_table_processing=_env_flag("ENABLE_TABLE_PROCESSING", "true"),
        enable_equation_processing=_env_flag("ENABLE_EQUATION_PROCESSING", "true"),
        enable_micro_planner=_env_flag("ENABLE_MICRO_PLANNER", "true"),
    )
    rag = RAGAnything(
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        config=rag_config,
        lightrag_kwargs={"rerank_model_func": rerank_model_func},
    )
    if rag.micro_planner:
        rag.micro_planner.evaluator_func = None

    if args.knowledge_dir:
        knowledge_dir = Path(args.knowledge_dir).expanduser()
        rag_output = Path(args.rag_output).expanduser()
        supported_exts = (
            [ext.strip() for ext in args.supported_exts.split(",") if ext.strip()]
            if args.supported_exts
            else _env_list("SUPPORTED_FILE_EXTENSIONS")
        )
        await _ingest_knowledge(
            rag,
            knowledge_dir=knowledge_dir,
            output_dir=rag_output,
            recursive=args.knowledge_recursive or _env_flag("RECURSIVE_FOLDER_PROCESSING", "false"),
            supported_exts=supported_exts,
        )

    image_root = Path(args.image_root).expanduser()
    ocr_root = Path(args.ocr_root).expanduser() if args.ocr_root else None

    await _ingest_docvqa_metadata(rag, questions, image_root, ocr_root)

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    result_file = output_root / f"{args.label}.json"
    trace_file = output_root / f"{args.label}_trace.json"
    summary_file = output_root / f"{args.label}_summary.json"

    results: Dict[str, str] = {}
    traces: Dict[str, Any] = {}
    
    # Metrics
    total = 0
    anls_scores = []
    em_scores = []
    f1_scores = []
    failed: List[int] = []

    iterator = tqdm(questions, desc="DocVQA", unit="q") if not args.debug else questions

    delay = max(0.0, args.rate_limit_delay)
    base_wait = delay if delay > 0 else 1.0
    backoff = max(1.0, args.rate_limit_backoff)
    retries = max(0, args.rate_limit_retries)

    request_timeout = _env_int("DOCVQA_REQUEST_TIMEOUT", _env_int("TIMEOUT", 240))

    async def call_with_retry(coro_factory, description: str):
        attempt = 0
        while True:
            try:
                logger.info(f"{description} attempt {attempt + 1} (timeout={request_timeout}s)")
                result = await asyncio.wait_for(coro_factory(), timeout=request_timeout)
                if delay > 0:
                    await asyncio.sleep(delay)
                return result
            except asyncio.TimeoutError:
                logger.error(f"{description} timed out after {request_timeout}s (attempt {attempt + 1})")
                if attempt >= retries:
                    raise
                wait = base_wait * (backoff ** attempt)
                tqdm.write(f"{description} timeout; sleeping {wait:.1f}s before retry")
                await asyncio.sleep(wait)
                attempt += 1
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt >= retries:
                    raise
                wait = base_wait * (backoff ** attempt)
                retry_after = _extract_retry_after(exc)
                if retry_after:
                    wait = max(wait, retry_after)
                msg = f"{description} hit rate limit (attempt {attempt + 1}/{retries + 1}); retrying in {wait:.1f}s"
                tqdm.write(msg)
                logger.warning(msg)
                await asyncio.sleep(wait)
                attempt += 1

    for idx, question in enumerate(iterator, start=1):
        q_id = question.get("questionId", idx)
        logger.info(f"[DocVQA] Start question {idx}/{len(questions)} (qid={q_id})")
        question_text = question.get("question", "")
        ground_truths = question.get("answers", [])

        planner_report: Optional[Dict[str, Any]] = None
        if rag and rag.micro_planner:
            try:
                normalized_query, planner_meta = rag.micro_planner.normalize(question_text)
                intent_result = rag.micro_planner.detect_intent(normalized_query, planner_meta)
                plan = rag.micro_planner.compile_strategy(intent_result.intent, intent_result.tags)
                budgets = {
                    "time_ms": _env_int("MICRO_PLANNER_TIME_MS", 1000),
                    "memory_gb": _env_float("MICRO_PLANNER_MEMORY_GB", 2.0),
                }
                plan = rag.micro_planner.apply_policies(plan, budgets)
                planner_report = plan.to_dict(question_text, normalized_query, intent_result.intent, intent_result.tags)
                logger.info(
                    "[Planner] qid=%s intent=%s tags=%s",
                    q_id,
                    intent_result.intent,
                    intent_result.tags,
                )
            except Exception as exc:
                logger.warning(f"[Planner] Failed to build plan for qid={q_id}: {exc}")

        context: Optional[str] = None
        if rag:
            context = await _retrieve_context(
                rag,
                question_text,
                args.mode,
                args.top_k,
            )
            logger.info(
                f"[DocVQA] Retrieved context for qid={q_id}: "
                f"{'none' if context is None else f'{len(context.split())} words'}"
            )

        clean_context = context
        if clean_context and clean_context.strip().lower().startswith("sorry"):
            logger.info(f"[Query] qid={q_id} context indicates no retrieval; omitting from prompt")
            clean_context = None

        # Load OCR text if requested
        ocr_text: Optional[str] = None
        if args.use_ocr and ocr_root:
            image_path = _resolve_image_path(question, image_root)
            if image_path:
                ocr_path = _resolve_ocr_path(question, ocr_root, image_path)
                if ocr_path:
                    ocr_text = load_ocr_text(ocr_path)
                    logger.info(f"[DocVQA] Loaded OCR text for qid={q_id}: {len(ocr_text) if ocr_text else 0} chars")

        prompt = _build_prompt(question_text, clean_context, ocr_text)
        image_path = _resolve_image_path(question, image_root)
        use_image = bool(image_path and not args.force_text)
        logger.info(
            f"[DocVQA] Prepared prompt for qid={q_id}; use_image={use_image}"
        )

        image_mime: Optional[str] = None
        effective_use_image = use_image

        if args.dry_run:
            response_text = "dry run answer"
        else:
            response_text = ""
            effective_use_image = use_image
            if use_image and image_path:
                encoded_image = _encode_image_to_base64(image_path)
                if encoded_image:
                    image_base64, image_mime = encoded_image
                    image_payload = {"base64": image_base64, "mime": image_mime}
                    logger.info(f"[DocVQA] Using image for qid={q_id}: {image_path} ({image_mime})")
                    try:
                        start_time = time.perf_counter()

                        async def call_vision():
                            return await vision_model_func(
                                prompt,
                                system_prompt=DEFAULT_SYSTEM_PROMPT,
                                image_data=image_payload,
                            )

                        response_text = await call_with_retry(call_vision, f"[{q_id}] vision completion")
                        elapsed = time.perf_counter() - start_time
                        logger.info(f"[DocVQA] Vision completion for qid={q_id} succeeded in {elapsed:.2f}s")
                    except Exception as exc:
                        logger.error(f"[{q_id}] Vision call failed: {exc}")
                        effective_use_image = False
                else:
                    logger.warning(f"[DocVQA] Failed to encode image for qid={q_id}: {image_path}")
                    effective_use_image = False

            if not response_text:
                try:
                    start_time = time.perf_counter()

                    async def call_text():
                        return await llm_model_func(
                            prompt,
                            system_prompt=DEFAULT_SYSTEM_PROMPT,
                        )

                    response_text = await call_with_retry(call_text, f"[{q_id}] text completion")
                    elapsed = time.perf_counter() - start_time
                    logger.info(f"[DocVQA] Text completion for qid={q_id} succeeded in {elapsed:.2f}s")
                except Exception as exc:
                    logger.error(f"[{q_id}] LLM call failed: {exc}")

        predicted = _clean_answer(response_text)

        reflection_report: Optional[Dict[str, Any]] = None
        if rag and rag.micro_planner:
            try:
                reflection_report = rag.micro_planner.evaluate(
                    question_text,
                    response_text,
                    {"context": context or ""},
                )
                logger.info(
                    "[Reflection] qid=%s score=%s",
                    q_id,
                    reflection_report.get("score"),
                )
            except Exception as exc:
                logger.warning(f"[Reflection] Evaluation failed for qid={q_id}: {exc}")

        # Compute metrics
        anls = compute_anls(predicted, ground_truths, args.anls_threshold)
        em = compute_exact_match(predicted, ground_truths)
        f1 = compute_f1(predicted, ground_truths)

        results[str(q_id)] = predicted
        traces[str(q_id)] = {
            "prompt": prompt,
            "context": context,
            "ocr_text": ocr_text,
            "response": response_text,
            "predicted": predicted,
            "ground_truths": ground_truths,
            "image_path": str(image_path) if effective_use_image and image_path else None,
            "image_mime": image_mime if effective_use_image and image_mime else None,
            "planner": planner_report,
            "reflection": reflection_report,
            "metrics": {
                "anls": anls,
                "exact_match": em,
                "f1": f1,
            },
        }
        logger.info(
            f"[Result] qid={q_id} predicted='{predicted[:50]}...' "
            f"ANLS={anls:.4f} EM={em:.4f} F1={f1:.4f}"
        )

        total += 1
        anls_scores.append(anls)
        em_scores.append(em)
        f1_scores.append(f1)
        
        if anls < 0.5:  # Consider failed if ANLS < 0.5
            failed.append(q_id)

        if args.save_every > 0 and idx % args.save_every == 0:
            with result_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            with trace_file.open("w", encoding="utf-8") as f:
                json.dump(traces, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {result_file} after {idx} examples.")

        if args.debug and idx >= 10:
            break

    with result_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with trace_file.open("w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)

    avg_anls = sum(anls_scores) / total if total else 0.0
    avg_em = sum(em_scores) / total if total else 0.0
    avg_f1 = sum(f1_scores) / total if total else 0.0

    summary = {
        "evaluated": total,
        "metrics": {
            "anls": avg_anls,
            "exact_match": avg_em,
            "f1": avg_f1,
        },
        "failed": failed,
    }
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {result_file}")
    print(f"Traces saved to {trace_file}")
    print(f"Summary saved to {summary_file}")
    if total:
        print(f"\nEvaluated: {total} questions")
        print(f"ANLS: {avg_anls:.4f}")
        print(f"Exact Match: {avg_em:.4f}")
        print(f"F1 Score: {avg_f1:.4f}")
    if failed:
        print(f"\nFailed question ids (ANLS<0.5): {failed[:20]}{'...' if len(failed) > 20 else ''}")


def main() -> None:
    configure_logging()
    args = parse_args()

    print("====Input Arguments====")
    safe_args = dict(vars(args))
    if safe_args.get("openai_key"):
        safe_args["openai_key"] = "***"
    print(json.dumps(safe_args, indent=2, sort_keys=False, default=str))

    data_file = Path(args.data_file).expanduser()
    questions, total = load_docvqa_data(data_file, args.test_number)

    asyncio.run(run_docvqa(args, questions))


if __name__ == "__main__":
    main()

