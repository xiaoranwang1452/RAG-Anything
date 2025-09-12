from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ReflectionConfig:
    """Configuration for the reflection pipeline."""

    targeted_top_k: int = 4
    """How many additional chunks to retrieve when issues are found (3–5 recommended)."""

    extended_context_max_tokens: int = 3000
    """Max tokens for merged/extended context budget."""

    review_temperature: float = 0.1
    rewrite_temperature: float = 0.2


@dataclass
class ReflectionReview:
    """Structured review result for a draft answer."""

    passed: bool
    missing_points: List[str] = field(default_factory=list)
    citation_issues: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Optional diagnostics
    sentence_support: List[Dict[str, Any]] = field(default_factory=list)
    cross_evidence_conflicts: List[str] = field(default_factory=list)
    coverage_gaps: List[str] = field(default_factory=list)


class ReflectionEngine:
    """Implements reflection pipeline: review → targeted retrieval → rewrite."""

    def __init__(self, lightrag, llm_model_func, tokenizer, config: ReflectionConfig | None = None):
        self.lightrag = lightrag
        self.llm_model_func = llm_model_func
        self.tokenizer = tokenizer
        self.config = config or ReflectionConfig()

    # ------------------------ Public API ------------------------
    async def review(self, query: str, draft: str, context: str) -> ReflectionReview:
        """Run multi-check review and return a structured JSON verdict."""
        system_prompt = (
            "You are a rigorous RAG reflection reviewer."
            " Check sentence-level evidence support, cross-evidence consistency,"
            " query coverage completeness, and perform a retrieval-augmented verification thought."
            " Return STRICT JSON with keys: passed (bool), missing_points (list of strings),"
            " citation_issues (list of strings), keywords (list of strings),"
            " sentence_support (list of {sentence, supported, evidence_snippets}),"
            " cross_evidence_conflicts (list), coverage_gaps (list)."
        )

        user_prompt = (
            "Query:\n" + query.strip() +
            "\n\nContext (evidence):\n" + context.strip() +
            "\n\nDraft Answer:\n" + draft.strip() +
            "\n\nInstructions:\n"
            "1) For each main sentence in the draft, decide if it is directly supported by the provided context;"
            " extract short evidence snippets if supported.\n"
            "2) Identify contradictions or conflicts among evidence.\n"
            "3) Check if the draft fully covers the query; note uncovered aspects as coverage_gaps.\n"
            "4) If any claims are unsupported or coverage is incomplete, propose 3-8 compact retrieval keywords.\n"
            "5) Set passed=true only if claims are supported AND coverage is adequate AND no contradictions.\n"
            "Output only JSON." 
        )

        raw = await self.llm_model_func(
            user_prompt,
            system_prompt=system_prompt,
            temperature=self.config.review_temperature,
        )
        data = self._safe_parse_json(raw)
        return self._coerce_review(data)

    async def targeted_retrieval(
        self, query: str, review: ReflectionReview
    ) -> List[Dict[str, Any]]:
        """Retrieve a few extra supporting chunks based on missing points/keywords."""
        # Build focused query text
        focus = ", ".join(review.missing_points[:6])
        kws = ", ".join(review.keywords[:10])
        augmented = f"{query}\nMissing aspects: {focus}\nKeywords: {kws}"

        try:
            results = await self.lightrag.chunks_vdb.query(
                augmented, top_k=max(1, self.config.targeted_top_k)
            )
        except Exception:
            results = []
        return results or []

    def merge_context(
        self, base_context: str, extra_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Merge base context with targeted evidence under a token budget.

        Returns a tuple of (extended_context, citations_metadata).
        citations_metadata example: [{"ref": "S1", "source": file_path, "id": id}]
        """
        pieces = []
        citations: List[Dict[str, Any]] = []

        # Start with base context
        pieces.append("[Base Context]\n" + base_context.strip())

        # Add extra evidence with inline numbered labels
        for idx, item in enumerate(extra_chunks, start=1):
            ref = f"S{idx}"
            content = item.get("content") or item.get("text") or ""
            meta_source = item.get("file_path") or item.get("full_doc_id") or item.get("source_id") or "unknown"
            pieces.append(f"[Evidence {ref}]\n{content}".strip())
            citations.append({"ref": ref, "source": meta_source, "id": item.get("id")})

        # Token-aware truncation
        merged = "\n\n".join(pieces)
        tokens = self._count_tokens(merged)
        if tokens <= self.config.extended_context_max_tokens:
            return merged, citations

        # Truncate from the end (keep base context and as many evidences as possible)
        kept = [pieces[0]]
        running = self._count_tokens(kept[0])
        kept_citations: List[Dict[str, Any]] = []
        for (p, c) in zip(pieces[1:], citations):
            t = self._count_tokens(p)
            if running + t + 10 <= self.config.extended_context_max_tokens:
                kept.append(p)
                kept_citations.append(c)
                running += t
            else:
                break
        return "\n\n".join(kept), kept_citations

    async def rewrite(
        self,
        query: str,
        draft: str,
        extended_context: str,
        review: ReflectionReview,
        citations: List[Dict[str, Any]],
    ) -> str:
        """Rewrite the answer in the new context, fixing support/gaps and citing evidence."""
        # Build citation legend
        legend_lines = [f"[{c['ref']}] source: {c['source']}" for c in citations]
        legend = "\n".join(legend_lines)

        system_prompt = (
            "You are a careful RAG writer using faithful decoding."
            " Only make claims that are explicitly supported by the provided context."
            " Prefer quoting or paraphrasing evidence."
            " If evidence is insufficient, state that limitation explicitly."
        )

        missing = "; ".join(review.missing_points)
        issues = "; ".join(review.citation_issues)
        user_prompt = (
            f"Query:\n{query}\n\n"  # noqa: E501
            f"Extended Context and Evidence (use these, keep within facts):\n{extended_context}\n\n"  # noqa: E501
            f"Citation Legend:\n{legend}\n\n"
            f"Original Draft:\n{draft}\n\n"
            f"Detected Missing Points: {missing or 'None'}\n"
            f"Detected Citation/Consistency Issues: {issues or 'None'}\n\n"
            "Rewrite the answer to be better grounded, accurate, and complete.\n"
            "Rules:\n"
            "- Fix unsupported claims; remove or correct them.\n"
            "- Cover missing aspects succinctly.\n"
            "- Cite evidence using [S1], [S2], ... inline where appropriate.\n"
            "- Do not invent facts not supported by evidence.\n"
            "- Keep the answer clear and concise.\n"
        )

        result = await self.llm_model_func(
            user_prompt,
            system_prompt=system_prompt,
            temperature=self.config.rewrite_temperature,
        )
        return result

    async def verify_after_rewrite(self, query: str, answer: str) -> bool:
        """Optional final verifier: re-query chunks DB for claims consistency.

        Lightweight: asks LLM to self-check without changing the answer. Returns True on pass.
        """
        system_prompt = (
            "You are a final verifier. Check if the answer is self-consistent,"
            " cautious, and free of obvious hallucinations given a brief re-retrieval."
            " Return 'PASS' or 'FAIL' only."
        )
        # Simple prompt; we avoid another heavy retrieval here for efficiency
        user_prompt = f"Question: {query}\nAnswer to verify:\n{answer}\n\nRespond with PASS or FAIL."
        raw = await self.llm_model_func(user_prompt, system_prompt=system_prompt, temperature=0.0)
        return "PASS" in raw.upper()

    # ------------------------ Helpers ------------------------
    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            # Try to extract JSON substring
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
        return {
            "passed": False,
            "missing_points": [],
            "citation_issues": ["Failed to parse reviewer JSON"],
            "keywords": [],
            "sentence_support": [],
            "cross_evidence_conflicts": [],
            "coverage_gaps": [],
        }

    def _coerce_review(self, data: Dict[str, Any]) -> ReflectionReview:
        return ReflectionReview(
            passed=bool(data.get("passed") or data.get("pass") or False),
            missing_points=list(data.get("missing_points") or []),
            citation_issues=list(data.get("citation_issues") or []),
            keywords=list(data.get("keywords") or []),
            sentence_support=list(data.get("sentence_support") or []),
            cross_evidence_conflicts=list(data.get("cross_evidence_conflicts") or []),
            coverage_gaps=list(data.get("coverage_gaps") or []),
        )

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text)) if self.tokenizer else len(text.split())
        except Exception:
            return len(text.split())
