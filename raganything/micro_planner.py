from __future__ import annotations
"""Lightweight micro planner for query normalization and strategy selection."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Callable
import unicodedata
import re


@dataclass
class IntentResult:
    """Result of intent detection and metadata extraction."""

    intent: str
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPlan:
    """Compiled plan for executing a query."""

    retrieval_mode: str = "mix"
    top_k: int = 5
    rerank_top_k: int = 5
    use_vlm: bool = False
    model_size: str = "base"
    degrade_reasons: List[str] = field(default_factory=list)
    feature_flags: Dict[str, bool] = field(
        default_factory=lambda: {"image": True, "table": True, "equation": True}
    )


class MicroPlanner:
    """Simple planner that prepares queries before passing to LightRAG."""

    def __init__(self, evaluator_func: Callable[[str], Any] | None = None):
        self.feature_flags = {"image": True, "table": True, "equation": True}
        self.evaluator_func = evaluator_func

    def normalize(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Normalize text and extract simple metadata.

        Strips Markdown image/table markup and LaTeX equations while recording
        counts of removed items.
        """

        metadata = {"images": 0, "tables": 0}

        image_pattern = r"!\[[^\]]*\]\([^\)]+\)"
        metadata["images"] += len(re.findall(image_pattern, text))
        text = re.sub(image_pattern, " ", text)

        latex_table = r"\\begin\{table\}"
        metadata["tables"] += len(re.findall(latex_table, text))

        markdown_table = r"\|.*?\|"
        metadata["tables"] += len(re.findall(markdown_table, text))

        # Remove LaTeX inline/math environments
        text = re.sub(r"\$[^$]+\$", " ", text)
        text = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", " ", text, flags=re.DOTALL)

        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        return normalized, metadata

    def detect_intent(self, text: str, metadata: Dict[str, int]) -> IntentResult:
        """Lightweight intent detector using keyword heuristics."""

        visual_keywords = ["figure", "image", "photo", "visual"]
        table_keywords = ["table", "spreadsheet"]
        tags: List[str] = []

        visual_score = sum(text.count(k) for k in visual_keywords) + metadata.get("images", 0)
        table_score = sum(text.count(k) for k in table_keywords) + metadata.get("tables", 0)

        if visual_score > 0 and table_score > 0:
            intent = "multimodal"
        elif visual_score > 0:
            intent = "visual"
        elif table_score > 0:
            intent = "table"
        else:
            intent = "text"

        if any(k in text for k in ["compare", "difference", "versus"]):
            tags.append("compare")
        if any(k in text for k in ["summarize", "summary", "overview"]):
            tags.append("summarize")
        if any(k in text for k in ["analyze", "analysis"]):
            tags.append("analyze")

        confidence = min(0.5 + 0.1 * (visual_score + table_score), 0.99)
        return IntentResult(intent=intent, tags=tags, confidence=confidence, metadata=metadata)

    def compile_strategy(self, intent: str, tags: List[str]) -> QueryPlan:
        """Build a query plan from detected intent."""

        plan = QueryPlan()
        if intent == "visual":
            plan.use_vlm = True
            plan.retrieval_mode = "local"
        elif intent == "table":
            plan.retrieval_mode = "local"
        elif intent == "multimodal":
            plan.use_vlm = True
            plan.retrieval_mode = "hybrid"
        else:
            plan.retrieval_mode = "mix"
        return plan

    def apply_policies(self, plan: QueryPlan, budgets: Dict[str, Any]) -> QueryPlan:
        """Apply resource-aware policies to the plan."""

        time_ms = budgets.get("time_ms", 1000)
        memory_gb = budgets.get("memory_gb", 2.0)

        if time_ms < 500:
            plan.top_k = 3
            plan.rerank_top_k = min(plan.rerank_top_k, 2)
            plan.degrade_reasons.append("low_time_budget")

        if memory_gb < 1.0:
            plan.feature_flags["image"] = False
            plan.feature_flags["equation"] = False
            plan.model_size = "small"
            plan.degrade_reasons.append("low_memory_budget")
            self.feature_flags.update(plan.feature_flags)
        return plan

    def evaluate(self, query: str, answer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight evaluation using LLM or lexical overlap."""

        result = {"score": 0.0, "degrade_reason": None}
        if not answer:
            result["degrade_reason"] = "empty_answer"
            return result

        score = 0.0
        if self.evaluator_func:
            prompt = (
                "Question: {q}\nAnswer: {a}\nOn a scale from 0 to 1, how relevant is the answer to the question?"
            ).format(q=query, a=answer)
            try:
                resp = self.evaluator_func(prompt)
                if isinstance(resp, (int, float)):
                    score = float(resp)
                else:
                    match = re.search(r"0\.\d+|1\.0", str(resp))
                    score = float(match.group()) if match else 0.0
            except Exception:
                score = 0.0
        else:
            q_terms = set(re.findall(r"\w+", query.lower()))
            a_terms = set(re.findall(r"\w+", answer.lower()))
            score = len(q_terms & a_terms) / max(1, len(q_terms))

        result["score"] = score
        if score < 0.2:
            result["degrade_reason"] = "low_relevance"
        return result

    def plan(
        self, query: str, budgets: Dict[str, Any] | None = None
    ) -> Tuple[str, IntentResult, QueryPlan]:
        """Entry point combining normalization, intent detection and policies."""

        normalized, metadata = self.normalize(query)
        intent = self.detect_intent(normalized, metadata)
        plan = self.compile_strategy(intent.intent, intent.tags)
        if budgets:
            plan = self.apply_policies(plan, budgets)
        return normalized, intent, plan