#!/usr/bin/env python
"""
ScienceQA tester: evaluate a VLM/LLM on ScienceQA-style JSON files.

It mirrors the workflow you described:
- Parse args and print them
- Load problems and a list of test qids
- For each qid, build a prompt (and attach image if present)
- Call the vision/text model to predict
- Save periodic results and final accuracy

Assumptions about dataset format (flexible):
- problems.json: a dict keyed by qid (str or int) -> problem dict
  Typical fields: "question", "choices" (list[str]), "answer" (int or str),
  optional "image" (filename) or "image_path".
- split.json: a list of qids for evaluation (str or int)

Models:
- Supports Ollama HTTP endpoints via the helpers in examples/example.py
  - LLM for text-only
  - VLM for image + text
- When LLM_BINDING is set to OpenAI/Azure in .env, calls the chat completion API directly.

Run example:
  python test/scienceQA_tester.py \
    --problems_json /path/to/problems.json \
    --split_json /path/to/test.json \
    --image_root /path/to/images \
    --output_root ./output \
    --label scienceqa \
    --save_every 50 \
    --test_number 200

Environment:
- Ensure Ollama is running and the .env contains model names or defaults are ok.
"""

import os
import json
import base64
import argparse
import random
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Allow importing from repo root
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
BASE_DIR = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv

# Reuse the minimal Ollama helpers
from examples.example import (
    build_ollama_llm_func,
    build_ollama_vision_func,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ScienceQA tester for RAG-Anything models")
    # ScienceQA-style inputs
    parser.add_argument("--data_root", default=None, help="If provided, read problems/pid_splits from here")
    parser.add_argument("--problems_json", default=None, help="Optional explicit path to problems.json")
    parser.add_argument("--split_json", default=None, help="Optional explicit path to a split qid list JSON (overrides pid_splits)")
    parser.add_argument("--caption_file", default=None, help="Optional captions.json path; will add problem['caption']")
    parser.add_argument("--image_root", default=None, help="Base folder for images, if problems only have filenames")
    parser.add_argument("--output_root", default="./output", help="Directory to write results")
    parser.add_argument("--label", default="scienceqa", help="Label prefix for result filename")
    parser.add_argument("--test_split", default="test", choices=["train","val","minival","test"], help="Split name")
    parser.add_argument("--save_every", type=int, default=50, help="Save results every N examples")
    parser.add_argument("--test_number", type=int, default=-1, help="Limit number of evaluated qids; -1 for all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force_text", action="store_true", help="Ignore images and do text-only")
    parser.add_argument("--use_caption", action="store_true", help="Append image captions to the prompt if available")
    # Few-shot placeholder (kept for interface compatibility)
    parser.add_argument("--shot_number", type=int, default=0, help="Number of few-shot examples (unused placeholder)")
    parser.add_argument("--shot_qids", default=None, help="Comma-separated qids for shots (unused placeholder)")
    return parser.parse_args()


def load_data(args) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Load problems and qids. Returns (problems, qids, shot_qids).

    - problems: dict keyed by str qid
    - qids: list[str] order to evaluate
    - shot_qids: empty list for now (placeholder to match interface)
    """
    # Resolve default paths if data_root is provided
    problems_path = args.problems_json
    split_path = args.split_json

    if args.data_root:
        root = Path(args.data_root)
        if not root.is_absolute():
            if (BASE_DIR / root).exists():
                root = BASE_DIR / root
        if not problems_path:
            problems_path = str(root / "problems.json")
        if not split_path and (root / "pid_splits.json").exists():
            split_path = str(root / "pid_splits.json")

    # If not specified, try ScienceQA defaults in repo
    if not problems_path:
        default_p1 = Path("ScienceQA/data/scienceqa/problems.json")
        default_p2 = BASE_DIR / "ScienceQA/data/scienceqa/problems.json"
        if default_p1.exists():
            problems_path = str(default_p1)
        elif default_p2.exists():
            problems_path = str(default_p2)
    if not split_path:
        default_s1 = Path("ScienceQA/data/scienceqa/pid_splits.json")
        default_s2 = BASE_DIR / "ScienceQA/data/scienceqa/pid_splits.json"
        if default_s1.exists():
            split_path = str(default_s1)
        elif default_s2.exists():
            split_path = str(default_s2)

    if not problems_path:
        raise FileNotFoundError(
            "problems.json not provided and no default found. "
            "Pass --data_root or --problems_json explicitly."
        )

    with open(problems_path, "r") as f:
        problems_raw = json.load(f)

    # Normalize keys to str
    problems: Dict[str, Any] = {str(k): v for k, v in problems_raw.items()}

    qids: List[str] = []
    if split_path:
        with open(split_path, "r") as f:
            split = json.load(f)
        # If a pid_splits.json, pick by test_split
        if isinstance(split, dict) and all(isinstance(v, list) for v in split.values()):
            qids = [str(q) for q in split.get(args.test_split, [])]
        elif isinstance(split, dict) and "qids" in split:
            qids = [str(q) for q in split["qids"]]
        else:
            qids = [str(q) for q in split]
    else:
        # Attempt to read pid_splits.json next to problems
        maybe = Path(problems_path).parent / "pid_splits.json"
        if maybe.exists():
            with open(maybe, "r") as f:
                pid_splits = json.load(f)
            qids = [str(q) for q in pid_splits.get(args.test_split, [])]
        else:
            # Fallback: evaluate all problems
            qids = list(problems.keys())

    # Optional: attach captions
    if args.caption_file:
        try:
            with open(args.caption_file, "r") as f:
                cap = json.load(f)
            cap_map = cap.get("captions", {}) if isinstance(cap, dict) else {}
            for k in problems:
                if k in cap_map:
                    problems[k]["caption"] = cap_map[k]
        except Exception:
            pass

    # Filter to only those present in problems and apply test_number limit
    qids = [qid for qid in qids if qid in problems]
    if args.test_number > 0:
        qids = qids[: args.test_number]

    print(f"number of test problems: {len(qids)}\n")

    # Few-shot placeholders to mirror interface (not used in prediction path)
    shot_qids: List[str] = []
    # If we have a pid_splits file, we can use its train set to sample few-shots
    try:
        pid_splits_path = Path(split_path) if split_path else (Path(problems_path).parent / "pid_splits.json")
        if pid_splits_path and pid_splits_path.exists():
            with open(pid_splits_path, "r") as f:
                pid_splits = json.load(f)
            train_qids = [str(q) for q in pid_splits.get("train", [])]
            if args.shot_qids:
                # Support comma-separated list
                cand = [s.strip() for s in str(args.shot_qids).split(",") if s.strip()]
                # validate
                shot_qids = [q for q in cand if q in train_qids]
            elif args.shot_number and args.shot_number > 0:
                import random as _rnd
                k = max(0, min(int(args.shot_number), len(train_qids)))
                shot_qids = _rnd.sample(train_qids, k) if k > 0 else []
    except Exception:
        shot_qids = []

    print("training question ids for prompting:", shot_qids, "\n")
    return problems, qids, shot_qids


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
    """Create an async chat completion helper for OpenAI/Azure OpenAI APIs."""

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
            # gpt-4o may return a list of content parts; join text fields
            text_parts = [c.get("text", "") for c in content if isinstance(c, dict)]
            return "\n".join([t for t in text_parts if t])
        return str(content)

    return _chat_func


def _choices_to_letter_map(choices: List[str]) -> Dict[str, int]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {letters[i]: i for i in range(min(len(choices), len(letters)))}


def _extract_predicted_letter(text: str, choices: List[str]) -> str:
    """Try to parse a single-choice letter (A/B/...). Robust to extra text."""
    if not choices:
        # fallback: return first capital letter found
        for ch in text:
            if "A" <= ch <= "Z":
                return ch
        return text.strip()[:1] or ""

    letter_map = _choices_to_letter_map(choices)
    letters = set(letter_map.keys())
    # 1) Look for standalone letter in parentheses or start/end
    import re
    m = re.search(r"\b([A-Z])\b", text)
    if m and m.group(1) in letters:
        return m.group(1)
    # 2) Look for patterns like "Answer: C" or "(C)"
    m = re.search(r"Answer\s*[:：]\s*([A-Z])", text, flags=re.I)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    m = re.search(r"\(([A-Z])\)", text)
    if m and m.group(1) in letters:
        return m.group(1)
    # 3) As a fallback, select the first letter mentioned left-to-right
    for ch in text:
        if ch in letters:
            return ch
    # 4) Give up: empty
    return ""


class SimpleScienceQAPredictor:
    """Minimal predictor using the configured LLM/VLM backend."""

    def __init__(self, force_text: bool = False, use_caption: bool = False):
        load_dotenv(dotenv_path=".env", override=False)
        binding = (os.getenv("LLM_BINDING", "ollama") or "").strip().lower()
        host = os.getenv("LLM_BINDING_HOST", "")

        if binding in {"openai", "azure_openai"} or _is_azure_host(host):
            self.llm = build_openai_chat_func()
            # gpt-4o style deployments handle both text and vision inputs
            self.vlm = self.llm
        else:
            self.llm = build_ollama_llm_func()
            self.vlm = build_ollama_vision_func()

        self.force_text = force_text or (os.getenv("DISABLE_VISION", "false").lower() == "true")
        self.use_caption = use_caption

    async def apredict(
        self,
        problem: Dict[str, Any],
        image_root: Optional[str] = None,
        qid: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        question = problem.get("question") or problem.get("hint") or ""
        choices: List[str] = problem.get("choices") or []

        # Build prompt
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
        # Append caption if requested and available
        if self.use_caption and problem.get("caption"):
            prompt_lines.append(f"Image caption: {problem['caption']}")
        prompt = "\n".join(prompt_lines)

        # Determine image path if any
        image_b64: Optional[str] = None
        if not self.force_text:
            img_name = problem.get("image") or problem.get("image_path")
            if img_name:
                candidate_paths: List[Path] = []
                if image_root:
                    # Try common ScienceQA layouts
                    # 1) <image_root>/<split>/<qid>/<image>
                    if split and qid:
                        candidate_paths.append(Path(image_root) / split / str(qid) / img_name)
                    # 2) <image_root>/<qid>/<image>
                    if qid:
                        candidate_paths.append(Path(image_root) / str(qid) / img_name)
                    # 3) <image_root>/<image>
                    candidate_paths.append(Path(image_root) / img_name)
                # 4) direct path in problem
                candidate_paths.append(Path(img_name))

                img_path: Optional[Path] = None
                for p in candidate_paths:
                    if p.exists() and p.is_file():
                        img_path = p
                        break
                if img_path:
                    image_b64 = _encode_image_to_base64(str(img_path))

        # Call model
        if image_b64:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }
            ]
            answer_text = await self.vlm(prompt=None, messages=messages)
        else:
            answer_text = await self.llm(prompt)

        pred_letter = _extract_predicted_letter(answer_text.strip(), choices)

        all_messages = {
            "prompt": prompt,
            "image_attached": bool(image_b64),
            "raw_answer": answer_text,
            "pred_letter": pred_letter,
        }
        return pred_letter, all_messages


def main():
    args = parse_args()
    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    # Resolve defaults for ScienceQA convenience BEFORE loading data
    default_data_root = BASE_DIR / "ScienceQA/data/scienceqa"
    default_captions = BASE_DIR / "ScienceQA/data/captions.json"
    if args.data_root is None and default_data_root.exists():
        args.data_root = str(default_data_root)
    if args.caption_file is None and default_captions.exists():
        args.caption_file = str(default_captions)
    # Default image_root if not provided
    if args.image_root is None and args.data_root:
        maybe_img = Path(args.data_root) / "images"
        if maybe_img.exists():
            args.image_root = str(maybe_img)

    problems, qids, shot_qids = load_data(args)  # problems, test qids, shot example ids (unused)

    result_file = f"{args.output_root}/{args.label}_{args.test_split}.json"
    os.makedirs(args.output_root, exist_ok=True)

    predictor = SimpleScienceQAPredictor(force_text=args.force_text, use_caption=args.use_caption)
    correct = 0
    results: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}
    failed: List[str] = []

    # Iterate
    for i, qid in enumerate(qids):
        if args.debug and i > 10:
            break
        if args.test_number > 0 and i >= args.test_number:
            break

        problem = problems[qid]

        # Normalize ground-truth answer to letter if possible
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
            pred, all_messages = asyncio.run(predictor.apredict(problem, image_root=args.image_root, qid=qid, split=args.test_split))
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
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {result_file} after {i + 1} examples.")

    with open(result_file, "w") as f:
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
