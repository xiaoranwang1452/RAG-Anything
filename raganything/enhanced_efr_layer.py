"""
Enhanced Evidence Fusion & Re-ranking (EFR) Layer

This module implements the enhanced EFR layer with advanced algorithms:
- MMMORRF: Multi-Modal Multi-Language Evidence Fusion
- Hybrid Reranking: Lightweight + LLM-based reranking
- RARE: Robustness Assessment and Conflict Detection
- RA-RAG: Source Reliability Estimation
- Adaptive MMR: Dynamic diversity balancing
"""

import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta

from .efr_layer import EFRLayer, RetrievalResult, EFRConfig


@dataclass
class EnhancedEFRConfig(EFRConfig):
    """Enhanced EFR Configuration"""
    # Modality weights
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 1.0, "image": 1.2, "table": 1.1, "equation": 1.3
    })
    
    # Language weights
    language_weights: Dict[str, float] = field(default_factory=lambda: {
        "en": 1.0, "zh": 1.1, "es": 0.9, "fr": 0.9
    })
    
    # Reranking thresholds
    lightweight_threshold: int = 10
    llm_threshold: int = 50
    
    # Robustness parameters
    robustness_threshold: float = 0.6
    complexity_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "max_length": 200, "min_length": 10
    })
    complexity_patterns: List[str] = field(default_factory=lambda: [
        r"\b(compare|contrast|analyze|evaluate|explain)\b",
        r"\b(what|how|why|when|where)\b.*\?",
        r"\b(advantages|disadvantages|pros|cons)\b"
    ])
    
    # Reliability parameters
    reliability_weights: Dict[str, float] = field(default_factory=lambda: {
        "consistency": 0.3, "accuracy": 0.3, "authority": 0.2, "freshness": 0.2
    })
    
    # MMR parameters
    mmr_lambda: float = 0.7
    adaptive_mmr: bool = True
    complexity_mmr_threshold: float = 0.8


class RobustnessChecker:
    """Robustness assessment and conflict detection"""
    
    def __init__(self, config: EnhancedEFRConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def check_query_robustness(self, query: str) -> Tuple[bool, float]:
        """Check query robustness and complexity"""
        complexity_score = 0.0
        
        # Length-based complexity
        if len(query) > self.config.complexity_thresholds["max_length"]:
            complexity_score += 0.3
        elif len(query) < self.config.complexity_thresholds["min_length"]:
            complexity_score += 0.1
        
        # Pattern-based complexity
        for pattern in self.config.complexity_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                complexity_score += 0.2
        
        # Question mark count
        question_marks = query.count('?')
        complexity_score += min(question_marks * 0.1, 0.3)
        
        is_robust = complexity_score >= self.config.robustness_threshold
        return is_robust, complexity_score
    
    def check_document_conflicts(self, results: List[RetrievalResult]) -> List[Tuple[int, int, float]]:
        """Check for conflicts between documents"""
        conflicts = []
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                conflict_score = self._calculate_conflict_score(results[i], results[j])
                if conflict_score > 0.5:  # Threshold for significant conflict
                    conflicts.append((i, j, conflict_score))
        
        return conflicts
    
    def _calculate_conflict_score(self, result1: RetrievalResult, result2: RetrievalResult) -> float:
        """Calculate conflict score between two results"""
        # Simple keyword-based conflict detection
        content1 = result1.content.lower()
        content2 = result2.content.lower()
        
        # Check for contradictory keywords
        contradiction_pairs = [
            ("increase", "decrease"), ("high", "low"), ("good", "bad"),
            ("positive", "negative"), ("success", "failure"), ("yes", "no")
        ]
        
        conflict_score = 0.0
        for pos_word, neg_word in contradiction_pairs:
            if (pos_word in content1 and neg_word in content2) or \
               (neg_word in content1 and pos_word in content2):
                conflict_score += 0.3
        
        return min(conflict_score, 1.0)


class SourceReliabilityEstimator:
    """Source reliability estimation"""
    
    def __init__(self, config: EnhancedEFRConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def estimate_reliability(self, result: RetrievalResult) -> float:
        """Estimate source reliability score"""
        consistency = self._calculate_consistency(result)
        accuracy = self._get_historical_accuracy(result)
        authority = self._get_authority_score(result)
        freshness = self._get_freshness_score(result)
        
        reliability = (
            consistency * self.config.reliability_weights["consistency"] +
            accuracy * self.config.reliability_weights["accuracy"] +
            authority * self.config.reliability_weights["authority"] +
            freshness * self.config.reliability_weights["freshness"]
        )
        
        return min(max(reliability, 0.0), 1.0)
    
    def _calculate_consistency(self, result: RetrievalResult) -> float:
        """Calculate content consistency score"""
        # Simple consistency based on content structure
        content = result.content
        
        # Check for structured content (tables, lists, etc.)
        structure_indicators = ["|", "-", "•", "1.", "2.", "3."]
        structure_count = sum(1 for indicator in structure_indicators if indicator in content)
        
        # Normalize to 0-1 range
        return min(structure_count / 5.0, 1.0)
    
    def _get_historical_accuracy(self, result: RetrievalResult) -> float:
        """Get historical accuracy score (placeholder)"""
        # In a real implementation, this would check historical accuracy
        return 0.8  # Default value
    
    def _get_authority_score(self, result: RetrievalResult) -> float:
        """Get authority score based on source"""
        source = result.source.lower()
        
        authority_map = {
            "academic": 0.9,
            "journal": 0.8,
            "conference": 0.7,
            "book": 0.8,
            "website": 0.5,
            "blog": 0.3,
            "unknown": 0.5
        }
        
        for source_type, score in authority_map.items():
            if source_type in source:
                return score
        
        return 0.5  # Default
    
    def _get_freshness_score(self, result: RetrievalResult) -> float:
        """Get freshness score based on document age"""
        # Simple freshness calculation (placeholder)
        # In real implementation, would use actual timestamps
        return 0.8  # Default value


class HybridReranker:
    """Hybrid reranking with lightweight and LLM-based approaches"""
    
    def __init__(self, config: EnhancedEFRConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def rerank(self, results: List[RetrievalResult], query: str, llm_func=None) -> List[RetrievalResult]:
        """Apply hybrid reranking"""
        if len(results) <= self.config.lightweight_threshold:
            return self._lightweight_rerank(results, query)
        elif len(results) <= self.config.llm_threshold and llm_func:
            return await self._llm_rerank(results, query, llm_func)
        else:
            return await self._hybrid_rerank(results, query, llm_func)
    
    def _lightweight_rerank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Lightweight reranking using keyword overlap"""
        query_words = set(query.lower().split())
        
        for result in results:
            content_words = set(result.content.lower().split())
            overlap = len(query_words.intersection(content_words))
            overlap_ratio = overlap / len(query_words) if query_words else 0
            
            # Adjust score based on overlap
            result.score = result.score * (1 + overlap_ratio * 0.5)
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    async def _llm_rerank(self, results: List[RetrievalResult], query: str, llm_func) -> List[RetrievalResult]:
        """LLM-based reranking"""
        if not llm_func:
            return self._lightweight_rerank(results, query)
        
        # Create ranking prompt
        prompt = f"Rank the following documents by relevance to the query: '{query}'\n\n"
        for i, result in enumerate(results):
            prompt += f"{i+1}. {result.content[:200]}...\n"
        
        prompt += "\nReturn only the ranking numbers in order of relevance."
        
        try:
            response = await llm_func(prompt)
            # Parse response and reorder results
            # This is a simplified implementation
            return results
        except Exception as e:
            self.logger.warning(f"LLM reranking failed: {e}, falling back to lightweight")
            return self._lightweight_rerank(results, query)
    
    async def _hybrid_rerank(self, results: List[RetrievalResult], query: str, llm_func) -> List[RetrievalResult]:
        """Hybrid reranking combining lightweight and LLM approaches"""
        # First apply lightweight filtering
        filtered_results = self._lightweight_rerank(results, query)
        
        # Then apply LLM reranking to top results
        top_results = filtered_results[:self.config.llm_threshold]
        reranked_top = await self._llm_rerank(top_results, query, llm_func)
        
        # Combine results
        return reranked_top + filtered_results[self.config.llm_threshold:]


class EnhancedEFRLayer(EFRLayer):
    """Enhanced EFR Layer with advanced algorithms"""
    
    def __init__(self, config: EnhancedEFRConfig):
        super().__init__(config)
        self.robustness_checker = RobustnessChecker(config)
        self.reliability_estimator = SourceReliabilityEstimator(config)
        self.reranker = HybridReranker(config)
        self.logger = logging.getLogger(__name__)
    
    async def process_retrieval_results(self, results: List[Dict[str, Any]], query: str, llm_func=None) -> List[RetrievalResult]:
        """Process retrieval results with enhanced EFR layer"""
        if not self.config.enable_efr:
            return self._convert_to_retrieval_results(results)
        
        self.logger.info(f"Processing {len(results)} results with Enhanced EFR layer")
        
        # Step 1: Robustness check
        is_robust, complexity_score = self.robustness_checker.check_query_robustness(query)
        self.logger.debug(f"Query robustness: {is_robust}, complexity: {complexity_score}")
        
        # Step 2: Convert to RetrievalResult objects
        retrieval_results = self._convert_to_retrieval_results(results)
        
        # Step 3: Source reliability estimation
        for result in retrieval_results:
            result.reliability_score = self.reliability_estimator.estimate_reliability(result)
        
        # Step 4: Enhanced weighted RRF
        weighted_results = self._apply_enhanced_weighted_rrf(retrieval_results, query)
        
        # Step 5: Hybrid reranking
        reranked_results = await self.reranker.rerank(weighted_results, query, llm_func)
        
        # Step 6: Enhanced recency and trust scoring
        enhanced_results = self._apply_enhanced_scoring(reranked_results, query)
        
        # Step 7: Adaptive MMR diversification
        if self.config.adaptive_mmr:
            diversified_results = self._apply_adaptive_mmr_diversification(enhanced_results, query, complexity_score)
        else:
            diversified_results = enhanced_results
        
        # Step 8: Final score calculation
        final_results = self._calculate_final_scores(diversified_results, query)
        
        # Limit results
        final_results = final_results[:self.config.max_results]
        
        self.logger.info(f"Enhanced EFR processing completed, returning {len(final_results)} results")
        return final_results
    
    def _apply_enhanced_weighted_rrf(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply enhanced weighted RRF"""
        for result in results:
            # Get weights
            modality_weight = self.config.modality_weights.get(result.modality, 1.0)
            language_weight = self.config.language_weights.get(result.language, 1.0)
            reliability_weight = result.reliability_score
            
            # Apply weighted RRF
            result.score = result.score * modality_weight * language_weight * reliability_weight
        
        return results
    
    def _apply_enhanced_scoring(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply enhanced recency and trust scoring"""
        for result in results:
            # Trust score based on reliability
            trust_score = result.reliability_score
            
            # Recency score (placeholder - would use actual timestamps)
            recency_score = 0.8  # Default value
            
            # Combine scores
            result.score = result.score * (0.7 * trust_score + 0.3 * recency_score)
        
        return results
    
    def _apply_adaptive_mmr_diversification(self, results: List[RetrievalResult], query: str, complexity_score: float) -> List[RetrievalResult]:
        """Apply adaptive MMR diversification"""
        if not results:
            return results
        
        # Adaptive lambda based on query complexity
        if complexity_score > self.config.complexity_mmr_threshold:
            lambda_val = self.config.mmr_lambda * 0.8  # More diversity for complex queries
        else:
            lambda_val = self.config.mmr_lambda * 1.2  # More relevance for simple queries
        
        # Simple MMR implementation
        selected = [results[0]]  # Start with highest scoring
        remaining = results[1:]
        
        while remaining and len(selected) < self.config.max_results:
            best_score = -1
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Calculate MMR score
                relevance = candidate.score
                diversity = self._calculate_diversity(candidate, selected)
                mmr_score = lambda_val * relevance - (1 - lambda_val) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _calculate_diversity(self, candidate: RetrievalResult, selected: List[RetrievalResult]) -> float:
        """Calculate diversity score"""
        if not selected:
            return 0.0
        
        # Simple diversity based on content similarity
        max_similarity = 0.0
        candidate_words = set(candidate.content.lower().split())
        
        for selected_result in selected:
            selected_words = set(selected_result.content.lower().split())
            similarity = len(candidate_words.intersection(selected_words)) / len(candidate_words.union(selected_words))
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_final_scores(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Calculate final scores"""
        if not results:
            return results
        
        # Normalize scores
        max_score = max(result.score for result in results)
        if max_score > 0:
            for result in results:
                result.score = result.score / max_score
        
        return sorted(results, key=lambda x: x.score, reverse=True)
