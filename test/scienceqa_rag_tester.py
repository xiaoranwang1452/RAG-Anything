#!/usr/bin/env python
"""
ScienceQA tester with optional RAG-Anything retrieval augmentation.

Pipeline:
- Load ScienceQA problems and split qids
- Optionally process a knowledge base folder via RAG-Anything
- For each qid: build prompt (question + choices + optional caption + optional retrieved context)
- Call LLM/VLM backend and parse a single-letter prediction
- Periodically save results; finally compute accuracy if GT is present

Usage examples:
  python test/scienceqa_rag_tester.py \
    --data_root ScienceQA/data/scienceqa \
    --output_root ./output \
    --label sqa_rag \
    --test_split test \
    --use_rag --kb_root ./kb --rag_top_k 3

  python test/scienceqa_rag_tester.py \
    --problems_json d:/data/problems.json \
    --split_json d:/data/pid_splits.json \
    --image_root d:/data/images \
    --force_text \
    --output_root ./output \
    --label sqa_text
"""

import os
import json
import base64
import argparse
import asyncio
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from dotenv import load_dotenv

# Reuse minimal Ollama helpers
from examples.example import (
    build_ollama_llm_func,
    build_ollama_vision_func,
)

# RAG-Anything
from raganything.raganything import RAGAnything
from raganything.config import Config


def parse_args():
    p = argparse.ArgumentParser(description="ScienceQA tester with optional RAG context")
    # ScienceQA-style inputs
    p.add_argument("--data_root", default=None, help="Root folder with problems.json, pid_splits.json, images/")
    p.add_argument("--problems_json", default=None, help="Explicit path to problems.json")
    p.add_argument("--split_json", default=None, help="Explicit path to split qid list JSON or pid_splits.json")
    p.add_argument("--caption_file", default=None, help="Optional captions.json path; attach problem['caption']")
    p.add_argument("--image_root", default=None, help="Base folder for images, used when problems only have filenames")
    # Output and run control
    p.add_argument("--output_root", default="./output", help="Directory to write results")
    p.add_argument("--label", default="scienceqa_rag", help="Label prefix for result filename")
    p.add_argument("--test_split", default="test", choices=["train","val","minival","test"], help="Split name")
    p.add_argument("--save_every", type=int, default=50, help="Save results every N examples")
    p.add_argument("--test_number", type=int, default=-1, help="Limit number of evaluated qids; -1 for all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--force_text", action="store_true", help="Ignore images and do text-only")
    p.add_argument("--use_caption", action="store_true", help="Append image captions to the prompt if available")
    # RAG augmentation
    p.add_argument("--use_rag", action="store_true", help="Enable RAG retrieval to add context to prompts")
    p.add_argument("--kb_root", default=None, help="Knowledge base root folder (PDFs, markdown, docs)")
    p.add_argument("--kb_glob", default="**/*", help="Glob pattern under kb_root for documents")
    p.add_argument("--rag_top_k", type=int, default=3, help="Top-k retrieved chunks to append to prompt")
    return p.parse_args()


def _encode_image_to_base64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _is_azure_host(url: Optional[str]) -> bool:
    if not url:
        return False
    return "azure.com" in url.lower()


def _post_json_with_headers(url: str, payload: dict, headers: dict, timeout: int = 240) -> dict:
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} error from {url}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e}")


def build_openai_chat_func():
    load_dotenv(dotenv_path=".env", override=False)
    host = os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1/chat/completions").strip()
    api_key = os.getenv("LLM_BINDING_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("LLM_BINDING_API_KEY must be set when using the OpenAI binding.")

    model = os.getenv("LLM_MODEL", "gpt-4o").strip()
    temperature_env = os.getenv("TEMPERATURE", "0")
    num_predict_env = os.getenv("NUM_PREDICT", "")
    timeout = int(os.getenv("TIMEOUT", "240"))

    try:
        temperature = float(temperature_env)
    except ValueError:
        temperature = 0.0

    try:
        max_tokens = int(num_predict_env) if num_predict_env else None
    except ValueError:
        max_tokens = None

    is_azure = _is_azure_host(host)
    headers = {"Content-Type": "application/json"}
    if is_azure:
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    async def _chat_func(prompt: str,
                         system_prompt: Optional[str] = None,
                         history_messages: Optional[List[dict]] = None,
                         messages: Optional[List[dict]] = None,
                         **kwargs) -> str:
        payload_messages: List[dict] = []
        if messages:
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    payload_messages.append({"role": role, "content": content})
                else:
                    payload_messages.append({"role": role, "content": str(content)})
        else:
            if system_prompt:
                payload_messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                for h in history_messages:
                    if "role" in h and "content" in h:
                        payload_messages.append({"role": h["role"], "content": h["content"]})
            payload_messages.append({"role": "user", "content": prompt or ""})

        payload: Dict[str, Any] = {
            "messages": payload_messages,
            "temperature": kwargs.get("temperature", temperature),
        }
        if not is_azure:
            payload["model"] = kwargs.get("model", model)
        if max_tokens is not None:
            payload["max_tokens"] = kwargs.get("max_tokens", max_tokens)

        response = await asyncio.to_thread(
            _post_json_with_headers,
            host,
            payload,
            headers,
            kwargs.get("timeout", timeout),
        )
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned: {response}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if isinstance(c, dict)]
            return "\n".join([t for t in text_parts if t])
        return str(content)

    return _chat_func


def _choices_to_letter_map(choices: List[str]) -> Dict[str, int]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {letters[i]: i for i in range(min(len(choices), len(letters)))}


def _extract_predicted_letter(text: str, choices: List[str]) -> str:
    if not choices:
        for ch in text:
            if "A" <= ch <= "Z":
                return ch
        return text.strip()[:1] or ""
    letter_map = _choices_to_letter_map(choices)
    letters = set(letter_map.keys())
    import re
    m = re.search(r"\b([A-Z])\b", text)
    if m and m.group(1) in letters:
        return m.group(1)
    m = re.search(r"Answer\s*[:：]\s*([A-Z])", text, flags=re.I)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    m = re.search(r"\(([A-Z])\)", text)
    if m and m.group(1) in letters:
        return m.group(1)
    for ch in text:
        if ch in letters:
            return ch
    return ""


class ScienceQARAGPredictor:
    def __init__(self, force_text: bool = False, use_caption: bool = False, use_rag: bool = False,
                 rag_top_k: int = 3, kb_root: Optional[str] = None, kb_glob: str = "**/*"):
        load_dotenv(dotenv_path=".env", override=False)
        binding = (os.getenv("LLM_BINDING", "ollama") or "").strip().lower()
        host = os.getenv("LLM_BINDING_HOST", "")

        if binding in {"openai", "azure_openai"} or _is_azure_host(host):
            self.llm = build_openai_chat_func()
            self.vlm = self.llm
        else:
            self.llm = build_ollama_llm_func()
            self.vlm = build_ollama_vision_func()

        self.force_text = force_text or (os.getenv("DISABLE_VISION", "false").lower() == "true")
        self.use_caption = use_caption
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k

        self.rag: Optional[RAGAnything] = None
        self.kb_root = kb_root
        self.kb_glob = kb_glob

        if self.use_rag and self.kb_root:
            cfg = Config.from_env()
            self.rag = RAGAnything(cfg)
            # Process knowledge base folder
            try:
                # Prefer batch folder processing if available
                self.rag.process_folder_complete(self.kb_root, glob=self.kb_glob)
            except Exception:
                # Fallback: iterate files
                base = Path(self.kb_root)
                for p in base.rglob(self.kb_glob):
                    if p.is_file():
                        try:
                            self.rag.process_document_complete(str(p))
                        except Exception:
                            pass

    async def _retrieve_context(self, question: str) -> List[str]:
        if not (self.use_rag and self.rag):
            return []
        try:
            resp = await self.rag.aquery(question)
            # Expect resp to contain retrieved contexts / passages
            ctxs: List[str] = []
            # Normalize across possible return structures
            if isinstance(resp, dict):
                for k in ["contexts", "chunks", "passages", "nodes", "results"]:
                    v = resp.get(k)
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                ctxs.append(item)
                            elif isinstance(item, dict):
                                for key in ["text", "content", "chunk", "passage"]:
                                    t = item.get(key)
                                    if isinstance(t, str):
                                        ctxs.append(t)
            elif isinstance(resp, list):
                ctxs = [str(x) for x in resp if isinstance(x, (str, dict))]
            # Deduplicate and take top-k
            uniq = []
            seen = set()
            for c in ctxs:
                s = c.strip()
                if s and s not in seen:
                    uniq.append(s)
                    seen.add(s)
            return uniq[: max(1, self.rag_top_k)]
        except Exception:
            return []

    async def apredict(self, problem: Dict[str, Any], image_root: Optional[str] = None,
                        qid: Optional[str] = None, split: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        question = problem.get("question") or problem.get("hint") or ""
        choices: List[str] = problem.get("choices") or []

        # Build prompt skeleton
        prompt_lines = [
            "You are solving a multiple-choice science question.",
            "Choose the single best option and answer ONLY with the letter.",
        ]
        if question:
            prompt_lines.append(f"Question: {question}")
        if choices:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for i, c in enumerate(choices):
                if i >= len(letters):
                    break
                prompt_lines.append(f"{letters[i]}. {c}")
            prompt_lines.append("Output format: a single letter like A, B, C, ...")
        if self.use_caption and problem.get("caption"):
            prompt_lines.append(f"Image caption: {problem['caption']}")

        # RAG retrieval
        retrieved: List[str] = await self._retrieve_context(question)
        if retrieved:
            prompt_lines.append("Context:")
            for i, ctx in enumerate(retrieved, 1):
                prompt_lines.append(f"[{i}] {ctx}")

        prompt = "\n".join(prompt_lines)

        # Attach image if present
        image_b64: Optional[str] = None
        if not self.force_text:
            img_name = problem.get("image") or problem.get("image_path")
            if img_name:
                candidate_paths: List[Path] = []
                if image_root:
                    if split and qid:
                        candidate_paths.append(Path(image_root) / split / str(qid) / img_name)
                    if qid:
                        candidate_paths.append(Path(image_root) / str(qid) / img_name)
                    candidate_paths.append(Path(image_root) / img_name)
                candidate_paths.append(Path(img_name))
                img_path: Optional[Path] = None
                for p in candidate_paths:
                    if p.exists() and p.is_file():
                        img_path = p
                        break
                if img_path:
                    image_b64 = _encode_image_to_base64(str(img_path))

        if image_b64:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }]
            answer_text = await self.vlm(prompt=None, messages=messages)
        else:
            answer_text = await self.llm(prompt)

        pred_letter = _extract_predicted_letter(answer_text.strip(), choices)
        all_messages = {
            "prompt": prompt,
            "image_attached": bool(image_b64),
            "raw_answer": answer_text,
            "pred_letter": pred_letter,
            "retrieved": retrieved,
        }
        return pred_letter, all_messages


def _load_data(args) -> Tuple[Dict[str, Any], List[str]]:
    problems_path = args.problems_json
    split_path = args.split_json
    if args.data_root:
        root = Path(args.data_root)
        if not root.is_absolute():
            maybe = BASE_DIR / root
            if maybe.exists():
                root = maybe
        if not problems_path:
            problems_path = str(root / "problems.json")
        if not split_path and (root / "pid_splits.json").exists():
            split_path = str(root / "pid_splits.json")

    if not problems_path:
        default_p = BASE_DIR / "ScienceQA/data/scienceqa/problems.json"
        if default_p.exists():
            problems_path = str(default_p)
    if not split_path:
        default_s = BASE_DIR / "ScienceQA/data/scienceqa/pid_splits.json"
        if default_s.exists():
            split_path = str(default_s)

    if not problems_path:
        raise FileNotFoundError("problems.json not provided and no default found.")

    with open(problems_path, "r", encoding="utf-8") as f:
        problems_raw = json.load(f)
    problems: Dict[str, Any] = {str(k): v for k, v in problems_raw.items()}

    qids: List[str] = []
    if split_path:
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        if isinstance(split, dict) and all(isinstance(v, list) for v in split.values()):
            qids = [str(q) for q in split.get(args.test_split, [])]
        elif isinstance(split, dict) and "qids" in split:
            qids = [str(q) for q in split["qids"]]
        else:
            qids = [str(q) for q in split]
    else:
        maybe = Path(problems_path).parent / "pid_splits.json"
        if maybe.exists():
            with open(maybe, "r", encoding="utf-8") as f:
                pid_splits = json.load(f)
            qids = [str(q) for q in pid_splits.get(args.test_split, [])]
        else:
            qids = list(problems.keys())

    # Optional captions merge
    if args.caption_file:
        try:
            with open(args.caption_file, "r", encoding="utf-8") as f:
                cap = json.load(f)
            cap_map = cap.get("captions", {}) if isinstance(cap, dict) else {}
            for k in problems:
                if k in cap_map:
                    problems[k]["caption"] = cap_map[k]
        except Exception:
            pass

    qids = [qid for qid in qids if qid in problems]
    if args.test_number > 0:
        qids = qids[: args.test_number]
    return problems, qids


def main():
    args = parse_args()
    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    # Resolve defaults for convenience
    default_root = BASE_DIR / "ScienceQA/data/scienceqa"
    default_caps = BASE_DIR / "ScienceQA/data/captions.json"
    if args.data_root is None and default_root.exists():
        args.data_root = str(default_root)
    if args.caption_file is None and default_caps.exists():
        args.caption_file = str(default_caps)
    if args.image_root is None and args.data_root:
        maybe_img = Path(args.data_root) / "images"
        if maybe_img.exists():
            args.image_root = str(maybe_img)

    problems, qids = _load_data(args)

    os.makedirs(args.output_root, exist_ok=True)
    result_file = f"{args.output_root}/{args.label}_{args.test_split}.json"

    predictor = ScienceQARAGPredictor(
        force_text=args.force_text,
        use_caption=args.use_caption,
        use_rag=args.use_rag,
        rag_top_k=args.rag_top_k,
        kb_root=args.kb_root,
        kb_glob=args.kb_glob,
    )

    correct = 0
    results: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}
    failed: List[str] = []

    for i, qid in enumerate(qids):
        if args.debug and i > 10:
            break
        if args.test_number > 0 and i >= args.test_number:
            break

        problem = problems[qid]
        gt_answer = problem.get("answer")
        choices: List[str] = problem.get("choices") or []
        gt_letter = None
        if gt_answer is not None and choices:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if isinstance(gt_answer, int):
                if 0 <= gt_answer < len(choices):
                    gt_letter = letters[gt_answer]
            elif isinstance(gt_answer, str):
                if gt_answer.upper() in letters:
                    gt_letter = gt_answer.upper()

        try:
            pred, all_messages = asyncio.run(
                predictor.apredict(problem, image_root=args.image_root, qid=qid, split=args.test_split)
            )
        except Exception as e:
            pred, all_messages = "", {"error": str(e)}

        outputs[qid] = all_messages
        results[qid] = pred

        if gt_letter is not None:
            if pred == gt_letter:
                correct += 1
            else:
                failed.append(qid)

        if (i + 1) % args.save_every == 0:
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {result_file} after {i + 1} examples.")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {result_file} after {len(qids)} examples.")

    if any(problems[qid].get("answer") is not None for qid in qids):
        denom = sum(1 for qid in qids if problems[qid].get("answer") is not None)
        print(f"Number of correct answers: {correct}/{denom}")
        acc = correct / denom if denom else 0.0
        print(f"Accuracy: {acc:.4f}")
        print(f"Failed question ids: {failed}")
        print(f"Number of failed questions: {len(failed)}")
    else:
        print("No ground-truth answers present; only predictions saved.")


if __name__ == "__main__":
    main()