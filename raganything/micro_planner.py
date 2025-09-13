from __future__ import annotations
"""Lightweight micro planner for query normalization and strategy selection."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import unicodedata
import re


@dataclass
class IntentResult:
    """Result of intent detection."""

    intent: str
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class QueryPlan:
    """Compiled plan for executing a query."""

    retrieval_mode: str = "mix"
    top_k: int = 5
    rerank_top_k: int = 5
    use_vlm: bool = False
    feature_flags: Dict[str, bool] = field(
        default_factory=lambda: {"image": True, "table": True, "equation": True}
    )


class MicroPlanner:
    """Simple planner that prepares queries before passing to LightRAG."""

    def __init__(self):
        self.feature_flags = {"image": True, "table": True, "equation": True}

    def normalize(self, text: str) -> str:
        """Normalize input text using Unicode NFKC and whitespace collapse."""

        normalized = unicodedata.normalize("NFKC", text).lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def detect_intent(self, text: str) -> IntentResult:
        """Very small rule based intent detector."""

        lower = text.lower()
        intent = "text"
        tags: List[str] = []
        if any(k in lower for k in ["figure", "image", "photo", "visual"]):
            intent = "visual"
        if "table" in lower:
            intent = "table"
        if any(k in lower for k in ["compare", "difference"]):
            tags.append("compare")
        return IntentResult(intent=intent, tags=tags, confidence=0.6)

    def compile_strategy(self, intent: str, tags: List[str]) -> QueryPlan:
        """Build a query plan from detected intent."""

        plan = QueryPlan()
        if intent == "visual":
            plan.use_vlm = True
            plan.retrieval_mode = "local"
        elif intent == "table":
            plan.retrieval_mode = "local"
        else:
            plan.retrieval_mode = "mix"
        return plan

    def apply_policies(self, plan: QueryPlan, budgets: Dict[str, Any]) -> QueryPlan:
        """Apply simple resource aware policies to the plan."""

        time_ms = budgets.get("time_ms", 1000)
        if time_ms < 500:
            plan.top_k = 3
        memory_gb = budgets.get("memory_gb", 2.0)
        if memory_gb < 1.0:
            plan.feature_flags["image"] = False
            plan.feature_flags["equation"] = False
            self.feature_flags.update(plan.feature_flags)
        return plan

    def evaluate(self, query: str, answer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder evaluation hook."""

        if not answer:
            return {"degrade_reason": "empty_answer"}
        return {"degrade_reason": None}

    def plan(
        self, query: str, budgets: Dict[str, Any] | None = None
    ) -> Tuple[str, QueryPlan]:
        """Entry point combining normalization, intent detection and policies."""

        normalized = self.normalize(query)
        intent = self.detect_intent(normalized)
        plan = self.compile_strategy(intent.intent, intent.tags)
        if budgets:
            plan = self.apply_policies(plan, budgets)
        return normalized, plan
