"""
Reflection Layer for RAGAnything

Implements a pluggable reflection layer that performs sentence-level verification
of generated answers using targeted retrieval and LLM-based analysis.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
try:
    from lightrag import QueryParam
except Exception:
    # Minimal fallback to avoid import crash; actual class comes from lightrag
    class QueryParam:
        def __init__(self, **kwargs):
            for k,v in kwargs.items():
                setattr(self, k, v)

from lightrag.utils import logger


@dataclass
class ReflectionConfig:
    """Configuration for ReflectionLayer"""
    enable_reflection: bool = True
    max_sentences: int = 10
    support_threshold: float = 0.7
    coverage_threshold: float = 0.8
    contradiction_threshold: float = 0.6


@dataclass
class ReflectionReport:
    """Report structure for reflection analysis"""
    support: Dict[str, Any] = None
    coverage: Dict[str, Any] = None
    contradiction: Dict[str, Any] = None
    attributable: Dict[str, Any] = None
    targeted_queries: List[str] = None

    def __post_init__(self):
        if self.support is None:
            self.support = {}
        if self.coverage is None:
            self.coverage = {}
        if self.contradiction is None:
            self.contradiction = {}
        if self.attributable is None:
            self.attributable = {}
        if self.targeted_queries is None:
            self.targeted_queries = []


class ReflectionLayer:
    """
    Reflection Layer for RAGAnything

    Performs sentence-level verification of generated answers through:
    1. Sentence-level support analysis
    2. Coverage analysis
    3. Contradiction detection
    4. Attributable rewriting with citations
    """

    def __init__(
        self,
        lightrag,
        llm_model_func,
        config=None,
    ):
        """
        Initialize ReflectionLayer

        Args:
            lightrag: LightRAG instance for retrieval
            llm_model_func: LLM function for analysis
            config: ReflectionConfig instance
        """
        self.lightrag = lightrag
        self.llm_model_func = llm_model_func
        self.config = config or {}
        self.logger = logger

    async def run(
        self,
        question: str,
        draft_answer: str,
        query_mode: str = "hybrid",
    ) -> Tuple[str, ReflectionReport]:
        """
        Run complete reflection process

        Args:
            question: Original user question
            draft_answer: Initial draft answer
            query_mode: Query mode for retrieval

        Returns:
            Tuple of (final_answer, reflection_report)
        """
        self.logger.info("Starting reflection process")

        # Initialize report
        report = ReflectionReport()

        try:
            # Step 1: Sentence-level support analysis
            self.logger.info("Performing sentence-level support analysis")
            support_result, support_queries = await self._analyze_sentence_support(
                question, draft_answer, query_mode
            )
            report.support = support_result
            report.targeted_queries.extend(support_queries)

            # Step 2: Coverage analysis
            self.logger.info("Performing coverage analysis")
            coverage_result = await self._analyze_coverage(
                question, draft_answer, support_result
            )
            report.coverage = coverage_result

            # Step 3: Contradiction detection
            self.logger.info("Performing contradiction detection")
            contradiction_result = await self._detect_contradictions(
                draft_answer, support_result
            )
            report.contradiction = contradiction_result

            # Step 4: Attributable rewriting
            self.logger.info("Performing attributable rewriting")
            final_answer, attributable_result = await self._rewrite_with_attribution(
                draft_answer, support_result, contradiction_result
            )
            report.attributable = attributable_result

            self.logger.info("Reflection process completed successfully")
            return final_answer, report

        except Exception as e:
            self.logger.error(f"Reflection process failed: {str(e)}")
            # Return original answer with error report
            report.support = {"error": str(e)}
            return draft_answer, report

    async def _analyze_sentence_support(
        self,
        question: str,
        draft_answer: str,
        query_mode: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Analyze sentence-level support using targeted retrieval

        Args:
            question: Original question
            draft_answer: Draft answer to analyze
            query_mode: Query mode for retrieval

        Returns:
            Tuple of (support_analysis, targeted_queries)
        """
        # Extract sentences from draft answer
        sentences = self._extract_sentences(draft_answer)

        support_analysis = {
            "sentences": [],
            "overall_support_score": 0.0,
        }

        targeted_queries = []

        for i, sentence in enumerate(sentences):
            try:
                # Generate targeted query for this sentence
                targeted_query = await self._generate_targeted_query(question, sentence)
                targeted_queries.append(targeted_query)

                # Retrieve relevant context for this sentence
                context = await self._retrieve_context_for_sentence(
                    targeted_query, query_mode
                )

                # Analyze support using LLM
                sentence_support = await self._analyze_single_sentence_support(
                    sentence, context, targeted_query
                )

                support_analysis["sentences"].append({
                    "sentence": sentence,
                    "sentence_id": i,
                    "targeted_query": targeted_query,
                    "support_score": sentence_support.get("support_score", 0.0),
                    "evidence": sentence_support.get("evidence", []),
                    "reasoning": sentence_support.get("reasoning", ""),
                    "context_used": len(context) if context else 0,
                })

            except Exception as e:
                self.logger.warning(f"Failed to analyze sentence {i}: {str(e)}")
                support_analysis["sentences"].append({
                    "sentence": sentence,
                    "sentence_id": i,
                    "support_score": 0.0,
                    "error": str(e),
                })

        # Calculate overall support score
        valid_scores = [s["support_score"] for s in support_analysis["sentences"] if "error" not in s]
        if valid_scores:
            support_analysis["overall_support_score"] = sum(valid_scores) / len(valid_scores)

        return support_analysis, targeted_queries

    async def _analyze_coverage(
        self,
        question: str,
        draft_answer: str,
        support_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze coverage of the answer

        Args:
            question: Original question
            draft_answer: Draft answer
            support_result: Results from support analysis

        Returns:
            Coverage analysis result
        """
        try:
            # Use LLM to analyze coverage
            coverage_prompt = self._get_coverage_prompt(question, draft_answer, support_result)

            coverage_response = await self.llm_model_func(
                coverage_prompt,
                system_prompt=self._get_coverage_system_prompt(),
            )

            # Parse JSON response
            coverage_result = self._parse_json_response(coverage_response, "coverage")

            return coverage_result

        except Exception as e:
            self.logger.error(f"Coverage analysis failed: {str(e)}")
            return {
                "coverage_score": 0.0,
                "missing_aspects": [],
                "error": str(e),
            }

    async def _detect_contradictions(
        self,
        draft_answer: str,
        support_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Detect contradictions in the draft answer

        Args:
            draft_answer: Draft answer to analyze
            support_result: Support analysis results

        Returns:
            Contradiction detection result
        """
        try:
            # Use LLM to detect contradictions
            contradiction_prompt = self._get_contradiction_prompt(draft_answer, support_result)

            contradiction_response = await self.llm_model_func(
                contradiction_prompt,
                system_prompt=self._get_contradiction_system_prompt(),
            )

            # Parse JSON response
            contradiction_result = self._parse_json_response(contradiction_response, "contradiction")

            return contradiction_result

        except Exception as e:
            self.logger.error(f"Contradiction detection failed: {str(e)}")
            return {
                "has_contradictions": False,
                "contradictions": [],
                "error": str(e),
            }

    async def _rewrite_with_attribution(
        self,
        draft_answer: str,
        support_result: Dict[str, Any],
        contradiction_result: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Rewrite answer with proper attribution and citations

        Args:
            draft_answer: Original draft answer
            support_result: Support analysis results
            contradiction_result: Contradiction detection results

        Returns:
            Tuple of (rewritten_answer, attribution_analysis)
        """
        try:
            # Use LLM to rewrite with attribution
            rewrite_prompt = self._get_rewrite_prompt(
                draft_answer, support_result, contradiction_result
            )

            rewritten_answer = await self.llm_model_func(
                rewrite_prompt,
                system_prompt=self._get_rewrite_system_prompt(),
            )

            # Count citations added
            citation_count = len(re.findall(r'\[[\w\-:]+\]', rewritten_answer))

            attribution_analysis = {
                "original_length": len(draft_answer),
                "rewritten_length": len(rewritten_answer),
                "citations_added": citation_count,
                "rewrite_successful": citation_count > 0,
            }

            return rewritten_answer, attribution_analysis

        except Exception as e:
            self.logger.error(f"Attribution rewriting failed: {str(e)}")
            attribution_analysis = {
                "error": str(e),
                "rewrite_successful": False,
            }
            return draft_answer, attribution_analysis

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        # Simple sentence splitting - can be enhanced with more sophisticated NLP
        sentences = re.split(r'[.!?]+', text.strip())
        # Filter out empty sentences and clean up
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences

    async def _generate_targeted_query(self, question: str, sentence: str) -> str:
        """Generate targeted query for a specific sentence"""
        # Create a focused query based on the sentence content
        # Extract key terms and concepts
        key_terms = self._extract_key_terms(sentence)

        # Combine original question context with sentence-specific terms
        if key_terms:
            targeted_query = f"{question} {' '.join(key_terms[:5])}"
        else:
            targeted_query = sentence

        return targeted_query

    def _extract_key_terms(self, sentence: str) -> List[str]:
        """Extract key terms from sentence for targeted querying"""
        # Simple key term extraction - can be enhanced with NLP
        words = re.findall(r'\b[A-Za-z]{3,}\b', sentence)
        # Filter common words (basic stopword removal)
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        key_terms = [word for word in words if word.lower() not in stopwords]
        return key_terms

    async def _retrieve_context_for_sentence(
        self,
        targeted_query: str,
        query_mode: str,
    ) -> Optional[str]:
        """Retrieve context for sentence verification"""
        try:
            # Try to use only_need_context if available
            query_param = QueryParam(
                mode=query_mode,
                only_need_context=True,
                top_k=getattr(self.config, 'reflection_top_k', 5),
            )

            context = await self.lightrag.aquery(targeted_query, param=query_param)
            return context

        except Exception as e:
            # Fallback to normal query if only_need_context is not available
            self.logger.debug(f"Context-only query failed, falling back to normal query: {str(e)}")
            try:
                query_param = QueryParam(mode=query_mode)
                result = await self.lightrag.aquery(targeted_query, param=query_param)
                return result
            except Exception as e2:
                self.logger.error(f"Context retrieval failed completely: {str(e2)}")
                return None

    async def _analyze_single_sentence_support(
        self,
        sentence: str,
        context: Optional[str],
        targeted_query: str,
    ) -> Dict[str, Any]:
        """Analyze support for a single sentence"""
        try:
            # Use LLM to analyze sentence support
            support_prompt = self._get_sentence_support_prompt(sentence, context, targeted_query)

            support_response = await self.llm_model_func(
                support_prompt,
                system_prompt=self._get_sentence_support_system_prompt(),
            )

            # Parse JSON response
            support_result = self._parse_json_response(support_response, "sentence_support")

            return support_result

        except Exception as e:
            self.logger.error(f"Single sentence support analysis failed: {str(e)}")
            return {
                "support_score": 0.0,
                "evidence": [],
                "reasoning": f"Analysis failed: {str(e)}",
            }

    def _parse_json_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Clean response and extract JSON
            response = response.strip()

            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, try to parse the whole response
                return json.loads(response)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response for {analysis_type}: {str(e)}")
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": response,
            }

    def _get_sentence_support_prompt(self, sentence: str, context: Optional[str], query: str) -> str:
        """Get prompt for sentence support analysis"""
        from raganything.prompt import PROMPTS
        return PROMPTS["REFLECTION_SENTENCE_SUPPORT"].format(
            sentence=sentence,
            context=context or "No relevant context found",
            query=query,
        )

    def _get_sentence_support_system_prompt(self) -> str:
        """Get system prompt for sentence support analysis"""
        from raganything.prompt import PROMPTS
        return PROMPTS["REFLECTION_SENTENCE_SUPPORT_SYS"]

    def _get_coverage_prompt(self, question: str, answer: str, support_result: Dict[str, Any]) -> str:
        """Get prompt for coverage analysis"""
        from raganything.prompt import PROMPTS
        return PROMPTS["REFLECTION_COVERAGE"].format(
            question=question,
            answer=answer,
            support_summary=json.dumps(support_result, indent=2),
        )

    def _get_coverage_system_prompt(self) -> str:
        """Get system prompt for coverage analysis"""
        return "You are an expert at analyzing answer coverage and completeness."

    def _get_contradiction_prompt(self, answer: str, support_result: Dict[str, Any]) -> str:
        """Get prompt for contradiction detection"""
        from raganything.prompt import PROMPTS
        return PROMPTS["REFLECTION_CONTRADICTION"].format(
            answer=answer,
            support_summary=json.dumps(support_result, indent=2),
        )

    def _get_contradiction_system_prompt(self) -> str:
        """Get system prompt for contradiction detection"""
        return "You are an expert at detecting logical contradictions and inconsistencies in text."

    def _get_rewrite_prompt(
        self,
        answer: str,
        support_result: Dict[str, Any],
        contradiction_result: Dict[str, Any],
    ) -> str:
        """Get prompt for rewriting with attribution"""
        from raganything.prompt import PROMPTS
        return PROMPTS["REFLECTION_ATTRIBUTABLE_REWRITE"].format(
            answer=answer,
            support_summary=json.dumps(support_result, indent=2),
            contradiction_summary=json.dumps(contradiction_result, indent=2),
        )

    def _get_rewrite_system_prompt(self) -> str:
        """Get system prompt for rewriting with attribution"""
        return "You are an expert at rewriting text with proper citations and attributions. Always add citations in the format [doc_id:span] to factual sentences."