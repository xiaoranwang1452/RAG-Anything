"""
Enhanced Evidence Fusion & Re-ranking (EFR) Layer V2

This module implements an advanced EFR layer based on recent research:
- MMMORRF: Multimodal Multilingual MOdularized Reciprocal Rank Fusion
- RA-RAG: Retrieval-Augmented Generation with Estimation of Source Reliability
- Adaptive control strategies for dynamic parameter adjustment
- Weighted majority voting for result fusion

Author: AI Assistant
Date: 2024
"""

import asyncio
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from lightrag import QueryParam
from lightrag.utils import logger as lightrag_logger

from .efr_layer import EFRLayer, RetrievalResult, EFRConfig


@dataclass
class EnhancedEFRConfigV2(EFRConfig):
    """Enhanced EFR Configuration V2 based on recent research"""
    
    # MMMORRF: Multimodal fusion weights
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 1.0, "image": 1.2, "table": 1.1, "equation": 1.3, "audio": 1.1
    })
    
    # Language weights for multilingual support
    language_weights: Dict[str, float] = field(default_factory=lambda: {
        "en": 1.0, "zh": 1.1, "es": 0.9, "fr": 0.9, "de": 0.9, "ja": 1.0
    })
    
    # RA-RAG: Source reliability parameters
    reliability_weights: Dict[str, float] = field(default_factory=lambda: {
        "consistency": 0.3, "accuracy": 0.3, "authority": 0.2, "freshness": 0.2
    })
    
    # Adaptive control parameters
    adaptive_threshold: float = 0.7
    complexity_adaptation: bool = True
    dynamic_weight_adjustment: bool = True
    
    # Weighted majority voting
    voting_threshold: float = 0.6
    consensus_weight: float = 0.8
    
    # Performance optimization
    enable_parallel_processing: bool = True
    max_workers: int = 4
    cache_size: int = 1000


class SourceReliabilityEstimatorV2:
    """Advanced source reliability estimation based on RA-RAG"""
    
    def __init__(self, config: EnhancedEFRConfigV2):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def estimate_reliability(self, result: RetrievalResult) -> float:
        """Estimate source reliability using multiple factors"""
        reliability_score = 0.0
        
        # Consistency check
        consistency_score = self._check_consistency(result)
        reliability_score += consistency_score * self.config.reliability_weights["consistency"]
        
        # Accuracy assessment
        accuracy_score = self._assess_accuracy(result)
        reliability_score += accuracy_score * self.config.reliability_weights["accuracy"]
        
        # Authority evaluation
        authority_score = self._evaluate_authority(result)
        reliability_score += authority_score * self.config.reliability_weights["authority"]
        
        # Freshness check
        freshness_score = self._check_freshness(result)
        reliability_score += freshness_score * self.config.reliability_weights["freshness"]
        
        return min(1.0, max(0.0, reliability_score))
    
    def _check_consistency(self, result: RetrievalResult) -> float:
        """Check content consistency"""
        content = result.content.lower()
        
        # Check for contradictory phrases
        contradiction_phrases = [
            "however", "but", "although", "despite", "nevertheless",
            "on the other hand", "in contrast", "whereas"
        ]
        
        contradiction_count = sum(1 for phrase in contradiction_phrases if phrase in content)
        consistency_score = max(0.0, 1.0 - (contradiction_count * 0.2))
        
        return consistency_score
    
    def _assess_accuracy(self, result: RetrievalResult) -> float:
        """Assess content accuracy based on metadata and content quality"""
        accuracy_score = 0.5  # Base score
        
        # Check metadata quality
        metadata = result.metadata
        if metadata.get("source_type") == "academic":
            accuracy_score += 0.3
        elif metadata.get("source_type") == "peer_reviewed":
            accuracy_score += 0.4
        
        # Check content structure
        content = result.content
        if len(content) > 100:  # Substantial content
            accuracy_score += 0.1
        if any(indicator in content.lower() for indicator in ["according to", "research shows", "studies indicate"]):
            accuracy_score += 0.1
        
        return min(1.0, accuracy_score)
    
    def _evaluate_authority(self, result: RetrievalResult) -> float:
        """Evaluate source authority"""
        authority_score = 0.5  # Base score
        
        metadata = result.metadata
        source = result.source.lower()
        
        # Academic sources
        if any(domain in source for domain in [".edu", ".ac.", "arxiv", "scholar"]):
            authority_score += 0.3
        
        # Reputable domains
        if any(domain in source for domain in [".gov", ".org", "wikipedia"]):
            authority_score += 0.2
        
        # Check for author information
        if metadata.get("author") or metadata.get("institution"):
            authority_score += 0.1
        
        return min(1.0, authority_score)
    
    def _check_freshness(self, result: RetrievalResult) -> float:
        """Check content freshness"""
        freshness_score = 0.5  # Base score
        
        metadata = result.metadata
        if "date" in metadata:
            # Simple freshness check (would need actual date parsing in production)
            freshness_score += 0.3
        
        # Recent content indicators
        content = result.content.lower()
        recent_indicators = ["2024", "2023", "recent", "latest", "new", "current"]
        if any(indicator in content for indicator in recent_indicators):
            freshness_score += 0.2
        
        return min(1.0, freshness_score)


class MMMORRFFusion:
    """Multimodal Multilingual MOdularized Reciprocal Rank Fusion"""
    
    def __init__(self, config: EnhancedEFRConfigV2):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fuse_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply MMMORRF fusion to results"""
        if not results:
            return results
        
        # Group results by modality
        modality_groups = self._group_by_modality(results)
        
        # Apply modality-specific weights
        weighted_results = []
        for modality, group_results in modality_groups.items():
            weight = self.config.modality_weights.get(modality, 1.0)
            for result in group_results:
                result.score *= weight
                weighted_results.append(result)
        
        # Apply language weights
        for result in weighted_results:
            lang_weight = self.config.language_weights.get(result.language, 1.0)
            result.score *= lang_weight
        
        # Apply reciprocal rank fusion
        fused_results = self._reciprocal_rank_fusion(weighted_results)
        
        return fused_results
    
    def _group_by_modality(self, results: List[RetrievalResult]) -> Dict[str, List[RetrievalResult]]:
        """Group results by modality"""
        groups = {}
        for result in results:
            modality = result.modality
            if modality not in groups:
                groups[modality] = []
            groups[modality].append(result)
        return groups
    
    def _reciprocal_rank_fusion(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply reciprocal rank fusion"""
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Apply RRF formula: 1 / (k + rank)
        k = 60  # Standard RRF parameter
        for i, result in enumerate(sorted_results):
            rrf_score = 1.0 / (k + i + 1)
            result.score = rrf_score
        
        return sorted_results


class WeightedMajorityVoting:
    """Weighted majority voting for result consensus"""
    
    def __init__(self, config: EnhancedEFRConfigV2):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def vote_on_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply weighted majority voting"""
        if len(results) < 2:
            return results
        
        # Group similar results
        consensus_groups = self._group_by_consensus(results)
        
        # Apply voting weights
        voted_results = []
        for group in consensus_groups:
            if len(group) >= 2:  # Only vote on groups with multiple results
                consensus_score = self._calculate_consensus_score(group)
                if consensus_score >= self.config.voting_threshold:
                    # Create consensus result
                    consensus_result = self._create_consensus_result(group, consensus_score)
                    voted_results.append(consensus_result)
                else:
                    # Keep individual results
                    voted_results.extend(group)
            else:
                voted_results.extend(group)
        
        return voted_results
    
    def _group_by_consensus(self, results: List[RetrievalResult]) -> List[List[RetrievalResult]]:
        """Group results by content similarity"""
        groups = []
        used_indices = set()
        
        for i, result in enumerate(results):
            if i in used_indices:
                continue
            
            group = [result]
            used_indices.add(i)
            
            for j, other_result in enumerate(results[i+1:], i+1):
                if j in used_indices:
                    continue
                
                similarity = self._calculate_similarity(result, other_result)
                if similarity > 0.7:  # Similarity threshold
                    group.append(other_result)
                    used_indices.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_similarity(self, result1: RetrievalResult, result2: RetrievalResult) -> float:
        """Calculate content similarity between two results"""
        content1 = result1.content.lower()
        content2 = result2.content.lower()
        
        # Simple word overlap similarity
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_consensus_score(self, group: List[RetrievalResult]) -> float:
        """Calculate consensus score for a group of results"""
        if len(group) < 2:
            return 1.0
        
        # Calculate average reliability
        avg_reliability = sum(r.reliability_score for r in group) / len(group)
        
        # Calculate group size factor
        size_factor = min(1.0, len(group) / 5.0)  # Normalize to max 5 results
        
        return (avg_reliability + size_factor) / 2.0
    
    def _create_consensus_result(self, group: List[RetrievalResult], consensus_score: float) -> RetrievalResult:
        """Create a consensus result from a group"""
        # Use the highest scoring result as base
        base_result = max(group, key=lambda x: x.score)
        
        # Create consensus content
        consensus_content = self._merge_contents(group)
        
        # Create consensus result
        consensus_result = RetrievalResult(
            content=consensus_content,
            score=base_result.score * consensus_score,
            metadata=base_result.metadata.copy(),
            modality=base_result.modality,
            language=base_result.language,
            source=base_result.source,
            reliability_score=consensus_score
        )
        
        return consensus_result
    
    def _merge_contents(self, group: List[RetrievalResult]) -> str:
        """Merge contents from a group of results"""
        contents = [r.content for r in group]
        
        # Simple merging strategy (could be more sophisticated)
        if len(contents) == 1:
            return contents[0]
        
        # Take the longest content as base and add unique information from others
        base_content = max(contents, key=len)
        unique_info = []
        
        for content in contents:
            if content != base_content:
                # Extract unique sentences (simplified)
                sentences = content.split('.')
                for sentence in sentences:
                    if sentence.strip() and sentence not in base_content:
                        unique_info.append(sentence.strip())
        
        if unique_info:
            base_content += " " + ". ".join(unique_info[:3])  # Add up to 3 unique sentences
        
        return base_content


class AdaptiveController:
    """Adaptive control for dynamic parameter adjustment"""
    
    def __init__(self, config: EnhancedEFRConfigV2):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.performance_history = []
    
    def adjust_parameters(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Dynamically adjust parameters based on query complexity and results"""
        adjustments = {}
        
        # Analyze query complexity
        complexity_score = self._analyze_query_complexity(query)
        
        # Analyze result quality
        quality_score = self._analyze_result_quality(results)
        
        # Adjust fusion weights based on complexity
        if complexity_score > self.config.adaptive_threshold:
            adjustments["fusion_weight"] = 1.2
            adjustments["reliability_threshold"] = 0.8
        else:
            adjustments["fusion_weight"] = 1.0
            adjustments["reliability_threshold"] = 0.6
        
        # Adjust voting parameters based on quality
        if quality_score < 0.5:
            adjustments["voting_threshold"] = 0.5
            adjustments["consensus_weight"] = 0.6
        else:
            adjustments["voting_threshold"] = self.config.voting_threshold
            adjustments["consensus_weight"] = self.config.consensus_weight
        
        return adjustments
    
    def _analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity"""
        complexity_indicators = [
            "compare", "contrast", "analyze", "evaluate", "explain",
            "advantages", "disadvantages", "pros", "cons", "benefits", "limitations"
        ]
        
        query_lower = query.lower()
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        
        # Length factor
        length_factor = min(1.0, len(query.split()) / 20.0)
        
        # Question complexity
        question_factor = 1.0 if "?" in query else 0.5
        
        return min(1.0, (complexity_count * 0.3 + length_factor * 0.4 + question_factor * 0.3))
    
    def _analyze_result_quality(self, results: List[RetrievalResult]) -> float:
        """Analyze result quality"""
        if not results:
            return 0.0
        
        # Average reliability score
        avg_reliability = sum(r.reliability_score for r in results) / len(results)
        
        # Content diversity
        content_lengths = [len(r.content) for r in results]
        length_variance = np.var(content_lengths) if len(content_lengths) > 1 else 0
        diversity_score = min(1.0, length_variance / 1000.0)  # Normalize variance
        
        # Score distribution
        scores = [r.score for r in results]
        score_variance = np.var(scores) if len(scores) > 1 else 0
        distribution_score = min(1.0, score_variance * 10)  # Normalize variance
        
        return (avg_reliability + diversity_score + distribution_score) / 3.0


class EnhancedEFRLayerV2(EFRLayer):
    """Enhanced EFR Layer V2 with advanced algorithms"""
    
    def __init__(self, config: EnhancedEFRConfigV2):
        super().__init__(config)
        self.config = config
        self.reliability_estimator = SourceReliabilityEstimatorV2(config)
        self.mmmorf_fusion = MMMORRFFusion(config)
        self.majority_voting = WeightedMajorityVoting(config)
        self.adaptive_controller = AdaptiveController(config)
        self.logger = logging.getLogger(__name__)
    
    async def process_retrieval_results(self, results: List[Dict[str, Any]], query: str, llm_func=None) -> List[RetrievalResult]:
        """Process retrieval results with enhanced EFR layer V2"""
        if not self.config.enable_efr:
            return self._convert_to_retrieval_results(results)
        
        self.logger.info(f"Processing {len(results)} results with Enhanced EFR Layer V2")
        
        # Step 1: Convert to RetrievalResult objects
        retrieval_results = self._convert_to_retrieval_results(results)
        
        # Step 2: Source reliability estimation (RA-RAG)
        for result in retrieval_results:
            result.reliability_score = self.reliability_estimator.estimate_reliability(result)
        
        # Step 3: Adaptive parameter adjustment
        adaptive_params = self.adaptive_controller.adjust_parameters(query, retrieval_results)
        
        # Step 4: MMMORRF fusion
        fused_results = self.mmmorf_fusion.fuse_results(retrieval_results, query)
        
        # Step 5: Weighted majority voting
        voted_results = self.majority_voting.vote_on_results(fused_results, query)
        
        # Step 6: Apply adaptive filtering
        filtered_results = self._apply_adaptive_filtering(voted_results, adaptive_params)
        
        # Step 7: Final ranking
        final_results = self._final_ranking(filtered_results, query)
        
        # Step 8: Limit results
        limited_results = final_results[:self.config.max_results]
        
        self.logger.info(f"EFR V2 processed {len(results)} -> {len(limited_results)} results")
        
        return limited_results
    
    def _apply_adaptive_filtering(self, results: List[RetrievalResult], adaptive_params: Dict[str, Any]) -> List[RetrievalResult]:
        """Apply adaptive filtering based on dynamic parameters"""
        reliability_threshold = adaptive_params.get("reliability_threshold", 0.3)  # Lower threshold
        
        filtered_results = []
        for result in results:
            if result.reliability_score >= reliability_threshold:
                # Apply fusion weight
                fusion_weight = adaptive_params.get("fusion_weight", 1.0)
                result.score *= fusion_weight
                filtered_results.append(result)
        
        # If no results pass the threshold, keep the top results anyway
        if not filtered_results and results:
            self.logger.warning(f"🔧 All results filtered out, keeping top {min(3, len(results))} results")
            # Sort by score and keep top results
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
            filtered_results = sorted_results[:min(3, len(results))]
            # Reset reliability scores to ensure they pass
            for result in filtered_results:
                result.reliability_score = max(0.5, result.reliability_score)
        
        return filtered_results
    
    def _final_ranking(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Final ranking based on combined scores"""
        for result in results:
            # Combine score and reliability
            combined_score = (result.score * 0.7 + result.reliability_score * 0.3)
            result.score = combined_score
        
        # Sort by combined score
        return sorted(results, key=lambda x: x.score, reverse=True)
