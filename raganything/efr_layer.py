"""
Evidence Fusion & Re-ranking (EFR) Layer

This module implements the base EFR layer for evidence fusion and re-ranking
in the RAGAnything framework.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import logging


@dataclass
class RetrievalResult:
    """Base class for retrieval results"""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality: str = "text"
    language: str = "en"
    source: str = "unknown"
    reliability_score: float = 1.0


@dataclass
class EFRConfig:
    """Configuration for EFR Layer"""
    enable_efr: bool = True
    rerank_threshold: float = 0.5
    max_results: int = 10
    fusion_strategy: str = "weighted"  # weighted, simple, advanced
    
    def __post_init__(self):
        if self.fusion_strategy not in ["weighted", "simple", "advanced"]:
            raise ValueError("fusion_strategy must be one of: weighted, simple, advanced")


class EFRLayer:
    """Base EFR Layer implementation"""
    
    def __init__(self, config: EFRConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_retrieval_results(self, results: List[Dict[str, Any]], query: str) -> List[RetrievalResult]:
        """
        Process retrieval results with EFR layer
        
        Args:
            results: List of raw retrieval results
            query: User query
            
        Returns:
            List of processed RetrievalResult objects
        """
        if not self.config.enable_efr:
            return self._convert_to_retrieval_results(results)
        
        self.logger.info(f"Processing {len(results)} results with EFR layer")
        
        # Convert to RetrievalResult objects
        retrieval_results = self._convert_to_retrieval_results(results)
        
        # Apply fusion strategy
        if self.config.fusion_strategy == "weighted":
            processed_results = self._weighted_fusion(retrieval_results, query)
        elif self.config.fusion_strategy == "simple":
            processed_results = self._simple_fusion(retrieval_results, query)
        else:  # advanced
            processed_results = self._advanced_fusion(retrieval_results, query)
        
        # Apply re-ranking
        reranked_results = self._rerank(processed_results, query)
        
        # Limit results
        final_results = reranked_results[:self.config.max_results]
        
        self.logger.info(f"EFR processing completed, returning {len(final_results)} results")
        return final_results
    
    def _convert_to_retrieval_results(self, results: List[Dict[str, Any]]) -> List[RetrievalResult]:
        """Convert raw results to RetrievalResult objects"""
        retrieval_results = []
        
        for result in results:
            retrieval_result = RetrievalResult(
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                modality=result.get("modality", "text"),
                language=result.get("language", "en"),
                source=result.get("source", "unknown"),
                reliability_score=result.get("reliability_score", 1.0)
            )
            retrieval_results.append(retrieval_result)
        
        return retrieval_results
    
    def _weighted_fusion(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply weighted fusion strategy"""
        # Simple weighted fusion based on scores and metadata
        for result in results:
            # Adjust score based on modality and language
            modality_weight = self._get_modality_weight(result.modality)
            language_weight = self._get_language_weight(result.language)
            
            result.score = result.score * modality_weight * language_weight * result.reliability_score
        
        return results
    
    def _simple_fusion(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply simple fusion strategy"""
        # Simple score normalization
        if results:
            max_score = max(result.score for result in results)
            if max_score > 0:
                for result in results:
                    result.score = result.score / max_score
        
        return results
    
    def _advanced_fusion(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply advanced fusion strategy"""
        # More sophisticated fusion logic
        return self._weighted_fusion(results, query)
    
    def _rerank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply re-ranking to results"""
        # Sort by score in descending order
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def _get_modality_weight(self, modality: str) -> float:
        """Get weight for different modalities"""
        weights = {
            "text": 1.0,
            "image": 1.2,
            "table": 1.1,
            "equation": 1.3
        }
        return weights.get(modality, 1.0)
    
    def _get_language_weight(self, language: str) -> float:
        """Get weight for different languages"""
        weights = {
            "en": 1.0,
            "zh": 1.1,
            "es": 0.9,
            "fr": 0.9
        }
        return weights.get(language, 1.0)
