"""
EFR Layer - Evidence Fusion & Re-Ranking Layer

This module implements the core EFR (Evidence Fusion & Re-Ranking) layer
that enhances retrieval results through:
1. Weighted RRF to fuse multiple candidate lists
2. On-demand listwise re-ranking
3. MMR diversification after re-rank
4. Recency & source-trust scoring

Author: AI Assistant
Date: 2024
"""

import asyncio
import math
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Unified retrieval result with various scores"""
    content: str
    score: float
    source: str
    chunk_id: Optional[str] = None
    entity_name: Optional[str] = None
    relation_type: Optional[str] = None
    doc_id: Optional[str] = None
    created_at: Optional[float] = None
    source_type: Optional[str] = None
    
    # EFR scores
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    mmr_score: float = 0.0
    recency_score: float = 0.0
    source_trust_score: float = 0.0
    final_score: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EFRConfig:
    """Configuration for EFR Layer"""
    # RRF parameters
    rrf_k: int = 60
    rrf_weights: Dict[str, float] = field(default_factory=lambda: {
        "vector": 1.0,
        "entity": 0.8,
        "relation": 0.7,
        "chunk": 1.0
    })
    
    # Re-ranking parameters
    enable_reranking: bool = True
    rerank_top_k: int = 10
    
    # MMR parameters
    enable_mmr: bool = True
    mmr_top_k: int = 5
    mmr_lambda: float = 0.7
    
    # Scoring parameters
    enable_recency_scoring: bool = True
    enable_source_trust_scoring: bool = True
    recency_decay_factor: float = 0.1
    source_trust_weights: Dict[str, float] = field(default_factory=lambda: {
        "academic": 1.0,
        "news": 0.8,
        "blog": 0.6,
        "unknown": 0.5
    })


class EFRLayer:
    """Evidence Fusion & Re-Ranking Layer"""
    
    def __init__(self, config: EFRConfig, rerank_func: Optional[callable] = None):
        self.config = config
        self.rerank_func = rerank_func
        self.logger = logging.getLogger(__name__)
    
    async def process_retrieval_results(
        self,
        query: str,
        retrieval_lists: Dict[str, List[Dict[str, Any]]],
        **kwargs
    ) -> List[RetrievalResult]:
        """Process retrieval results through EFR pipeline"""
        self.logger.info(f"EFR processing query: {query[:100]}...")
        
        # Step 1: Convert to RetrievalResult objects
        all_results = self._convert_to_retrieval_results(retrieval_lists)
        
        # Step 2: Apply Weighted RRF
        rrf_results = await self._apply_weighted_rrf(all_results)
        
        # Step 3: Apply listwise re-ranking
        if self.config.enable_reranking:
            reranked_results = await self._apply_listwise_rerank(query, rrf_results)
        else:
            reranked_results = rrf_results
        
        # Step 4: Apply MMR diversification
        if self.config.enable_mmr:
            diversified_results = await self._apply_mmr_diversification(query, reranked_results)
        else:
            diversified_results = reranked_results
        
        # Step 5: Apply recency and source-trust scoring
        scored_results = await self._apply_recency_and_trust_scoring(diversified_results)
        
        # Step 6: Calculate final scores
        final_results = self._calculate_final_scores(scored_results)
        
        # Sort by final score
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        
        self.logger.info(f"EFR processing complete. Returned {len(final_results)} results")
        return final_results
    
    def _convert_to_retrieval_results(self, retrieval_lists: Dict[str, List[Dict[str, Any]]]) -> List[RetrievalResult]:
        """Convert retrieval lists to RetrievalResult objects"""
        results = []
        
        for source_type, items in retrieval_lists.items():
            for item in items:
                result = RetrievalResult(
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    source=source_type,
                    chunk_id=item.get("chunk_id"),
                    entity_name=item.get("entity_name"),
                    relation_type=item.get("relation_type"),
                    doc_id=item.get("doc_id"),
                    created_at=item.get("created_at"),
                    source_type=source_type,
                    metadata=item.get("metadata", {})
                )
                results.append(result)
        
        return results
    
    async def _apply_weighted_rrf(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply Weighted Reciprocal Rank Fusion"""
        self.logger.info("Applying Weighted RRF...")
        
        # Group results by source type
        source_groups = defaultdict(list)
        for result in results:
            source_groups[result.source].append(result)
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        
        for source_type, source_results in source_groups.items():
            weight = self.config.rrf_weights.get(source_type, 1.0)
            
            # Sort by original score
            source_results.sort(key=lambda x: x.score, reverse=True)
            
            for rank, result in enumerate(source_results, 1):
                rrf_score = weight / (self.config.rrf_k + rank)
                rrf_scores[result] += rrf_score
        
        # Update RRF scores
        for result in results:
            result.rrf_score = rrf_scores[result]
        
        # Sort by RRF score
        results.sort(key=lambda x: x.rrf_score, reverse=True)
        
        return results
    
    async def _apply_listwise_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply listwise re-ranking"""
        self.logger.info("Applying listwise re-ranking...")
        
        if not self.rerank_func or len(results) <= 1:
            return results
        
        # Prepare documents for reranking
        documents = [result.content for result in results[:self.config.rerank_top_k]]
        
        try:
            # Call rerank function
            rerank_results = await self.rerank_func(query, documents)
            
            # Update rerank scores
            for i, result in enumerate(results[:self.config.rerank_top_k]):
                if i < len(rerank_results):
                    result.rerank_score = rerank_results[i].get("relevance_score", 0.5)
                else:
                    result.rerank_score = 0.5
            
            # Sort by rerank score
            results[:self.config.rerank_top_k].sort(key=lambda x: x.rerank_score, reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Reranking failed: {e}")
            # Fallback: use original scores
            for result in results:
                result.rerank_score = result.rrf_score
        
        return results
    
    async def _apply_mmr_diversification(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply MMR diversification"""
        self.logger.info("Applying MMR diversification...")
        
        if len(results) <= self.config.mmr_top_k:
            return results
        
        selected = []
        remaining = results.copy()
        
        # Select first result (highest score)
        if remaining:
            selected.append(remaining.pop(0))
        
        # MMR selection
        while len(selected) < self.config.mmr_top_k and remaining:
            best_idx = 0
            best_score = -1
            
            for i, candidate in enumerate(remaining):
                # Calculate MMR score
                relevance = candidate.rrf_score
                max_similarity = 0.0
                
                for selected_result in selected:
                    similarity = self._calculate_similarity(candidate.content, selected_result.content)
                    max_similarity = max(max_similarity, similarity)
                
                mmr_score = self.config.mmr_lambda * relevance - (1 - self.config.mmr_lambda) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            # Add best candidate
            selected.append(remaining.pop(best_idx))
        
        # Update MMR scores
        for result in selected:
            result.mmr_score = 1.0
        
        return selected
    
    async def _apply_recency_and_trust_scoring(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply recency and source-trust scoring"""
        self.logger.info("Applying recency and source-trust scoring...")
        
        current_time = time.time()
        
        for result in results:
            # Recency scoring
            if self.config.enable_recency_scoring and result.created_at:
                age_days = (current_time - result.created_at) / (24 * 3600)
                result.recency_score = math.exp(-self.config.recency_decay_factor * age_days)
            else:
                result.recency_score = 1.0
            
            # Source-trust scoring
            if self.config.enable_source_trust_scoring:
                source_type = result.source_type or "unknown"
                result.source_trust_score = self.config.source_trust_weights.get(source_type, 0.5)
            else:
                result.source_trust_score = 1.0
        
        return results
    
    def _calculate_final_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Calculate final scores"""
        self.logger.info("Calculating final scores...")
        
        for result in results:
            # Weighted combination of all scores
            result.final_score = (
                result.rrf_score * 0.4 +
                result.rerank_score * 0.3 +
                result.mmr_score * 0.1 +
                result.recency_score * 0.1 +
                result.source_trust_score * 0.1
            )
        
        return results
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple implementation)"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0