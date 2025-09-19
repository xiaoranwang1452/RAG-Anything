from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional


@dataclass
class FaithfulDecodingConfig:
    """Config for faithfulness-aware decoding.

    Uses N-best candidate sampling and an LLM-based verifier to select
    the most faithful answer with respect to the retrieved context.
    """

    num_candidates: int = 3
    temperature: float = 0.2
    min_accept_score: float = 0.6

    # Verifier prompts
    verifier_system_prompt: str = (
        "You are a strict faithfulness verifier for RAG answers.\n"
        "Given a question, a retrieval context, and an answer, judge whether\n"
        "the answer's claims are directly supported by the context.\n"
        "- Score from 0.0 to 1.0 (1.0 = fully supported, 0.0 = hallucinated).\n"
        "- List unsupported or dubious sentences from the answer.\n"
        "Return STRICT JSON: {\"score\": float, \"unsupported\": [string,...]}"
    )

    # Optional rewrite if best score is low
    enable_rewrite_on_low_score: bool = True
    rewrite_temperature: float = 0.2
    rewrite_threshold: float = 0.5


class FaithfulDecodingEngine:
    """Faithfulness-aware decoding via candidate reranking and verification."""

    def __init__(self, llm_model_func: Callable):
        self.llm_model_func = llm_model_func

    async def generate(
        self,
        query: str,
        system_prompt: str,
        context: str,
        config: Optional[FaithfulDecodingConfig] = None,
    ) -> str:
        cfg = config or FaithfulDecodingConfig()

        # 1) Sample N candidates under the same LightRAG-built system prompt
        candidates: List[str] = []
        for _ in range(max(1, cfg.num_candidates)):
            text = await self.llm_model_func(
                query,
                system_prompt=system_prompt,
                temperature=cfg.temperature,
            )
            candidates.append(text or "")

        # 2) Score faithfulness of each candidate
        scored = []
        for ans in candidates:
            score, unsupported = await self._score_faithfulness(
                query, context, ans, cfg
            )
            scored.append({
                "answer": ans,
                "score": score,
                "unsupported": unsupported,
            })

        # 3) Pick the highest-scored
        best = max(scored, key=lambda x: x.get("score", 0.0)) if scored else {"answer": "", "score": 0.0}

        # 4) Optional rewrite if low score
        if (
            cfg.enable_rewrite_on_low_score
            and (best.get("score") or 0.0) < cfg.rewrite_threshold
        ):
            try:
                rewritten = await self._rewrite_with_constraints(
                    query=query,
                    context=context,
                    draft=best.get("answer", ""),
                    temperature=cfg.rewrite_temperature,
                )
                # Re-score rewritten
                r_score, _ = await self._score_faithfulness(query, context, rewritten, cfg)
                if r_score > (best.get("score") or 0.0):
                    best = {"answer": rewritten, "score": r_score, "unsupported": []}
            except Exception:
                # Fall back to best without rewrite on error
                pass

        return best.get("answer", "")

    async def _score_faithfulness(
        self,
        query: str,
        context: str,
        answer: str,
        cfg: FaithfulDecodingConfig,
    ) -> tuple[float, List[str]]:
        user = (
            "Question:\n" + query.strip() +
            "\n\nContext (evidence):\n" + context.strip() +
            "\n\nAnswer to verify:\n" + answer.strip() +
            "\n\nReturn STRICT JSON only."
        )
        raw = await self.llm_model_func(
            user, system_prompt=cfg.verifier_system_prompt, temperature=0.0
        )
        data = self._safe_json(raw)
        score = float(data.get("score") or 0.0)
        unsupported = list(data.get("unsupported") or [])
        # Clamp score
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        return score, unsupported

    async def _rewrite_with_constraints(
        self,
        query: str,
        context: str,
        draft: str,
        temperature: float = 0.2,
    ) -> str:
        system = (
            "You are a careful RAG writer using faithful decoding."
            " Only include claims supported by the provided context."
            " Prefer quoting or paraphrasing evidence; if evidence is insufficient,"
            " state limitations instead of inventing facts."
        )
        user = (
            f"Question:\n{query}\n\n"
            f"Context (evidence):\n{context}\n\n"
            f"Draft Answer:\n{draft}\n\n"
            "Rewrite the answer to be faithful and grounded.\n"
            "- Correct or remove unsupported statements.\n"
            "- Keep clear and concise; cite short quotes when helpful.\n"
        )
        return await self.llm_model_func(user, system_prompt=system, temperature=temperature)

    def _safe_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(text[s:e+1])
                except Exception:
                    pass
        return {"score": 0.0, "unsupported": ["Failed to parse verifier JSON"]}

