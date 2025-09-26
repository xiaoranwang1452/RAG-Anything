"""
Enhanced EFR Layer with Advanced Techniques

This module implements an enhanced version of the Evidence Fusion & Re-Ranking layer
incorporating advanced techniques from recent research papers:

1. MMMORRF: Multimodal Multilingual MOdularized Reciprocal Rank Fusion
2. LLM-based Reranker Analysis: Hybrid reranking strategies
3. RARE: Retrieval-Aware Robustness Evaluation
4. RA-RAG: Source Reliability Estimation

Author: AI Assistant
Date: 2024
"""

import asyncio
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import hashlib
import re

from .efr_layer import EFRLayer, EFRConfig, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class EnhancedEFRConfig(EFRConfig):
    """Enhanced EFR configuration with advanced features"""
    
    # MMMORRF-inspired multimodal weights
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 1.0,
        "image": 0.8,
        "audio": 0.7,
        "table": 0.9,
        "equation": 0.8,
        "video": 0.6,
    })
    
    # Multilingual support
    enable_multilingual: bool = True
    language_weights: Dict[str, float] = field(default_factory=lambda: {
        "zh": 1.0,  # Chinese
        "en": 1.0,  # English
        "ja": 0.9,  # Japanese
        "ko": 0.9,  # Korean
        "fr": 0.8,  # French
        "de": 0.8,  # German
    })
    
    # Hybrid reranking configuration
    enable_hybrid_reranking: bool = True
    lightweight_rerank_threshold: int = 20  # Use lightweight for >20 docs
    llm_rerank_threshold: int = 10  # Use LLM for <=10 docs
    
    # Robustness evaluation
    enable_robustness_check: bool = True
    robustness_threshold: float = 0.5
    enable_conflict_resolution: bool = True
    
    # Source reliability estimation
    enable_source_reliability: bool = True
    reliability_weights: Dict[str, float] = field(default_factory=lambda: {
        "consistency": 0.4,  # Cross-source consistency
        "accuracy": 0.3,     # Historical accuracy
        "authority": 0.2,    # Source authority
        "freshness": 0.1,    # Temporal freshness
    })
    
    # Advanced MMR configuration
    enable_adaptive_mmr: bool = True
    mmr_adaptation_factor: float = 0.1  # How much to adapt based on query complexity


class RobustnessChecker:
    """Robustness evaluation for queries and documents"""
    
    def __init__(self):
        self.complexity_indicators = [
            r'\b(compare|contrast|difference|similarity)\b',
            r'\b(why|how|what if|explain)\b',
            r'\b(multiple|several|various|different)\b',
            r'\b(relationship|connection|correlation)\b',
        ]
    
    async def check_query_robustness(self, query: str) -> float:
        """Check query robustness based on complexity indicators"""
        complexity_score = 0.0
        
        # Check for complexity indicators
        for pattern in self.complexity_indicators:
            if re.search(pattern, query.lower()):
                complexity_score += 0.25
        
        # Check query length and structure
        if len(query.split()) > 10:
            complexity_score += 0.2
        
        if '?' in query:
            complexity_score += 0.1
        
        # Robustness is inverse of complexity
        robustness = max(0.0, 1.0 - complexity_score)
        return robustness
    
    async def check_document_conflicts(self, documents: List[Dict]) -> List[Dict]:
        """Check for conflicts between documents"""
        conflicts = []
        
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i+1:], i+1):
                conflict_score = await self._calculate_conflict_score(doc1, doc2)
                if conflict_score > 0.7:  # High conflict threshold
                    conflicts.append({
                        'doc1': doc1,
                        'doc2': doc2,
                        'conflict_score': conflict_score
                    })
        
        return conflicts
    
    async def _calculate_conflict_score(self, doc1: Dict, doc2: Dict) -> float:
        """Calculate conflict score between two documents"""
        # Simple semantic conflict detection
        content1 = doc1.get('content', '').lower()
        content2 = doc2.get('content', '').lower()
        
        # Check for contradictory keywords
        contradictions = [
            ('yes', 'no'), ('true', 'false'), ('correct', 'incorrect'),
            ('increase', 'decrease'), ('positive', 'negative'),
            ('support', 'oppose'), ('agree', 'disagree')
        ]
        
        conflict_score = 0.0
        for pos, neg in contradictions:
            if (pos in content1 and neg in content2) or (neg in content1 and pos in content2):
                conflict_score += 0.2
        
        return min(1.0, conflict_score)


class SourceReliabilityEstimator:
    """Source reliability estimation based on RA-RAG principles"""
    
    def __init__(self, config: EnhancedEFRConfig):
        self.config = config
        self.source_history = {}  # Track historical performance
    
    async def estimate_reliability(self, sources: List[Dict]) -> Dict[str, float]:
        """Estimate reliability of different sources"""
        reliability_scores = {}
        
        for source in sources:
            source_id = source.get('source_id', source.get('chunk_id', 'unknown'))
            
            # 1. Cross-source consistency
            consistency_score = await self._calculate_consistency(source, sources)
            
            # 2. Historical accuracy
            accuracy_score = await self._get_historical_accuracy(source_id)
            
            # 3. Source authority
            authority_score = self._get_authority_score(source)
            
            # 4. Temporal freshness
            freshness_score = self._get_freshness_score(source)
            
            # Weighted combination
            reliability_scores[source_id] = (
                consistency_score * self.config.reliability_weights['consistency'] +
                accuracy_score * self.config.reliability_weights['accuracy'] +
                authority_score * self.config.reliability_weights['authority'] +
                freshness_score * self.config.reliability_weights['freshness']
            )
        
        return reliability_scores
    
    async def _calculate_consistency(self, source: Dict, all_sources: List[Dict]) -> float:
        """Calculate consistency with other sources"""
        if len(all_sources) <= 1:
            return 1.0
        
        content = source.get('content', '').lower()
        consistency_scores = []
        
        for other_source in all_sources:
            if other_source == source:
                continue
            
            other_content = other_source.get('content', '').lower()
            
            # Simple semantic similarity
            similarity = self._calculate_semantic_similarity(content, other_content)
            consistency_scores.append(similarity)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _get_historical_accuracy(self, source_id: str) -> float:
        """Get historical accuracy of a source"""
        # In a real implementation, this would query a database
        # For now, return a default score
        return self.source_history.get(source_id, 0.7)
    
    def _get_authority_score(self, source: Dict) -> float:
        """Get authority score based on source type"""
        source_type = source.get('source_type', 'unknown')
        authority_map = {
            'academic': 1.0,
            'official': 0.9,
            'news': 0.7,
            'blog': 0.5,
            'forum': 0.3,
            'unknown': 0.6,
        }
        return authority_map.get(source_type, 0.6)
    
    def _get_freshness_score(self, source: Dict) -> float:
        """Get freshness score based on creation time"""
        created_at = source.get('created_at')
        if not created_at:
            return 0.5
        
        current_time = time.time()
        age_days = (current_time - created_at) / (24 * 3600)
        
        # Exponential decay
        return math.exp(-0.1 * age_days)


class HybridReranker:
    """Hybrid reranking combining lightweight and LLM-based approaches"""
    
    def __init__(self, config: EnhancedEFRConfig, llm_reranker=None, lightweight_reranker=None):
        self.config = config
        self.llm_reranker = llm_reranker
        self.lightweight_reranker = lightweight_reranker
    
    async def rerank(self, query: str, documents: List[str], **kwargs) -> List[Dict]:
        """Hybrid reranking strategy"""
        if not documents:
            return []
        
        # Decide which reranker to use based on document count
        if len(documents) > self.config.lightweight_rerank_threshold:
            # Use lightweight reranker for large document sets
            return await self._lightweight_rerank(query, documents, **kwargs)
        elif len(documents) <= self.config.llm_rerank_threshold and self.llm_reranker:
            # Use LLM reranker for small, high-quality sets
            return await self._llm_rerank(query, documents, **kwargs)
        else:
            # Use lightweight first, then LLM for top results
            return await self._hybrid_rerank(query, documents, **kwargs)
    
    async def _lightweight_rerank(self, query: str, documents: List[str], **kwargs) -> List[Dict]:
        """Lightweight reranking using simple heuristics"""
        if self.lightweight_reranker:
            return await self.lightweight_reranker(query, documents, **kwargs)
        
        # Fallback to simple scoring
        results = []
        query_words = set(query.lower().split())
        
        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words.intersection(doc_words))
            score = overlap / len(query_words) if query_words else 0.0
            
            results.append({
                'index': i,
                'relevance_score': score,
                'content': doc
            })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results
    
    async def _llm_rerank(self, query: str, documents: List[str], **kwargs) -> List[Dict]:
        """LLM-based reranking"""
        if self.llm_reranker:
            return await self.llm_reranker(query, documents, **kwargs)
        else:
            return await self._lightweight_rerank(query, documents, **kwargs)
    
    async def _hybrid_rerank(self, query: str, documents: List[str], **kwargs) -> List[Dict]:
        """Hybrid reranking: lightweight first, then LLM"""
        # Step 1: Lightweight reranking to get top candidates
        lightweight_results = await self._lightweight_rerank(query, documents, **kwargs)
        top_candidates = lightweight_results[:self.config.llm_rerank_threshold]
        
        # Step 2: LLM reranking on top candidates
        if self.llm_reranker and len(top_candidates) > 1:
            top_docs = [result['content'] for result in top_candidates]
            llm_results = await self._llm_rerank(query, top_docs, **kwargs)
            return llm_results
        else:
            return top_candidates


class EnhancedEFRLayer(EFRLayer):
    """Enhanced EFR Layer with advanced techniques"""
    
    def __init__(self, config: EnhancedEFRConfig = None, rerank_func: callable = None):
        super().__init__(config or EnhancedEFRConfig(), rerank_func)
        self.enhanced_config = self.config  # Type hint for enhanced config
        
        # Initialize advanced components
        self.robustness_checker = RobustnessChecker()
        self.reliability_estimator = SourceReliabilityEstimator(self.enhanced_config)
        self.hybrid_reranker = HybridReranker(
            self.enhanced_config,
            llm_reranker=rerank_func,
            lightweight_reranker=self._create_lightweight_reranker()
        )
    
    def _create_lightweight_reranker(self):
        """Create a lightweight reranker for hybrid approach"""
        async def lightweight_rerank(query: str, documents: List[str], **kwargs):
            results = []
            query_words = set(query.lower().split())
            
            for i, doc in enumerate(documents):
                doc_words = set(doc.lower().split())
                overlap = len(query_words.intersection(doc_words))
                score = overlap / len(query_words) if query_words else 0.0
                
                # Add length penalty for very short documents
                if len(doc.split()) < 10:
                    score *= 0.8
                
                results.append({
                    'index': i,
                    'relevance_score': score,
                    'content': doc
                })
            
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results
        
        return lightweight_rerank
    
    async def process_retrieval_results(
        self,
        query: str,
        retrieval_lists: Dict[str, List[Dict[str, Any]]],
        **kwargs
    ) -> List[RetrievalResult]:
        """Enhanced processing with advanced techniques"""
        logger.info(f"Enhanced EFR processing query: {query[:100]}...")
        
        # Step 1: Robustness check
        if self.enhanced_config.enable_robustness_check:
            robustness_score = await self.robustness_checker.check_query_robustness(query)
            logger.info(f"Query robustness score: {robustness_score:.3f}")
            
            if robustness_score < self.enhanced_config.robustness_threshold:
                logger.info("Low robustness query, using conservative processing")
                return await self._conservative_processing(query, retrieval_lists, **kwargs)
        
        # Step 2: Convert to RetrievalResult objects with enhanced metadata
        all_results = self._convert_to_enhanced_retrieval_results(retrieval_lists)
        
        # Step 3: Source reliability estimation
        if self.enhanced_config.enable_source_reliability:
            reliability_scores = await self.reliability_estimator.estimate_reliability(all_results)
            for result in all_results:
                source_id = result.chunk_id or result.entity_name or 'unknown'
                result.metadata['reliability_score'] = reliability_scores.get(source_id, 0.5)
        
        # Step 4: Enhanced Weighted RRF with modality awareness
        rrf_results = await self._apply_enhanced_weighted_rrf(all_results)
        
        # Step 5: Hybrid reranking
        if self.enhanced_config.enable_hybrid_reranking:
            reranked_results = await self._apply_hybrid_rerank(query, rrf_results)
        else:
            reranked_results = await self._apply_listwise_rerank(query, rrf_results)
        
        # Step 6: Enhanced recency and trust scoring
        scored_results = await self._apply_enhanced_scoring(reranked_results)
        
        # Step 7: Adaptive MMR diversification
        if self.enhanced_config.enable_adaptive_mmr:
            final_results = await self._apply_adaptive_mmr_diversification(query, scored_results)
        else:
            final_results = await self._apply_mmr_diversification(query, scored_results)
        
        # Step 8: Calculate enhanced final scores
        final_results = self._calculate_enhanced_final_scores(final_results)
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        
        logger.info(f"Enhanced EFR processing completed: {len(final_results)} final results")
        return final_results
    
    def _convert_to_enhanced_retrieval_results(
        self, 
        retrieval_lists: Dict[str, List[Dict[str, Any]]]
    ) -> List[RetrievalResult]:
        """Convert with enhanced metadata extraction"""
        results = []
        
        for source_type, items in retrieval_lists.items():
            if not items:
                continue
            
            for i, item in enumerate(items):
                # Extract enhanced metadata
                content = (
                    item.get("content") or 
                    item.get("text") or 
                    item.get("chunk_content") or 
                    str(item)
                )
                
                # Detect modality
                modality = self._detect_modality(content, item)
                
                # Detect language
                language = self._detect_language(content)
                
                # Enhanced metadata
                enhanced_metadata = {
                    **item.get("metadata", {}),
                    "modality": modality,
                    "language": language,
                    "source_type": source_type,
                    "original_rank": i + 1,
                }
                
                result = RetrievalResult(
                    content=content,
                    score=item.get("score", 1.0 / (i + 1)),
                    source=source_type,
                    chunk_id=item.get("chunk_id") or item.get("id"),
                    entity_name=item.get("entity_name"),
                    relation_type=item.get("relation_type"),
                    doc_id=item.get("doc_id") or item.get("full_doc_id"),
                    created_at=item.get("created_at"),
                    source_type=item.get("source_type", "unknown"),
                    metadata=enhanced_metadata
                )
                
                results.append(result)
        
        return results
    
    def _detect_modality(self, content: str, item: Dict) -> str:
        """Detect content modality"""
        # Check for explicit modality indicators
        if item.get("type") in ["image", "table", "equation", "audio", "video"]:
            return item.get("type")
        
        # Heuristic detection based on content
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in ["table", "row", "column", "|"]):
            return "table"
        elif any(keyword in content_lower for keyword in ["equation", "formula", "\\frac", "\\sum"]):
            return "equation"
        elif any(keyword in content_lower for keyword in ["image", "figure", "chart", "graph"]):
            return "image"
        else:
            return "text"
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection"""
        # Simple heuristic based on character patterns
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', content))
        korean_chars = len(re.findall(r'[\uac00-\ud7af]', content))
        
        total_chars = len(content)
        
        if chinese_chars / total_chars > 0.3:
            return "zh"
        elif japanese_chars / total_chars > 0.1:
            return "ja"
        elif korean_chars / total_chars > 0.1:
            return "ko"
        else:
            return "en"
    
    async def _apply_enhanced_weighted_rrf(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Enhanced RRF with modality and language awareness"""
        logger.info("Applying enhanced weighted RRF...")
        
        # Group results by content similarity
        content_groups = defaultdict(list)
        for result in results:
            key = result.chunk_id or hash(result.content) % 1000000
            content_groups[key].append(result)
        
        rrf_results = []
        for group in content_groups.values():
            if not group:
                continue
            
            # Calculate enhanced weighted RRF score
            total_rrf_score = 0.0
            for result in group:
                # Base source weight
                source_weight = self.enhanced_config.rrf_weights.get(result.source, 1.0)
                
                # Modality weight
                modality = result.metadata.get("modality", "text")
                modality_weight = self.enhanced_config.modality_weights.get(modality, 1.0)
                
                # Language weight
                language = result.metadata.get("language", "en")
                language_weight = self.enhanced_config.language_weights.get(language, 1.0)
                
                # Reliability weight
                reliability = result.metadata.get("reliability_score", 0.5)
                
                # Combined weight
                combined_weight = source_weight * modality_weight * language_weight * reliability
                
                # RRF score calculation
                rank = max(1, int(1.0 / result.score)) if result.score > 0 else 1
                rrf_score = combined_weight / (self.enhanced_config.rrf_k + rank)
                
                result.rrf_score = rrf_score
                total_rrf_score += rrf_score
            
            # Select best result from group
            best_result = max(group, key=lambda x: x.rrf_score)
            best_result.rrf_score = total_rrf_score
            rrf_results.append(best_result)
        
        rrf_results.sort(key=lambda x: x.rrf_score, reverse=True)
        logger.info(f"Enhanced RRF processing: {len(results)} -> {len(rrf_results)} results")
        return rrf_results
    
    async def _apply_hybrid_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply hybrid reranking strategy"""
        logger.info("Applying hybrid reranking...")
        
        if not self.hybrid_reranker:
            return await self._apply_listwise_rerank(query, results)
        
        # Prepare documents for reranking
        documents = [r.content for r in results]
        
        # Use hybrid reranker
        rerank_results = await self.hybrid_reranker.rerank(query, documents)
        
        # Update results with rerank scores
        for i, result in enumerate(results):
            if i < len(rerank_results):
                result.rerank_score = rerank_results[i].get('relevance_score', result.rrf_score)
            else:
                result.rerank_score = result.rrf_score
        
        logger.info("Hybrid reranking completed")
        return results
    
    async def _apply_enhanced_scoring(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply enhanced recency and trust scoring"""
        logger.info("Applying enhanced scoring...")
        
        current_time = time.time()
        
        for result in results:
            # Enhanced recency scoring
            if self.enhanced_config.enable_recency and result.created_at:
                age_days = (current_time - result.created_at) / (24 * 3600)
                result.recency_score = math.exp(-self.enhanced_config.recency_decay_factor * age_days)
            else:
                result.recency_score = 0.5
            
            # Enhanced source trust scoring
            if self.enhanced_config.enable_source_trust:
                base_trust = self.enhanced_config.source_trust_weights.get(
                    result.source_type, 
                    self.enhanced_config.source_trust_weights["unknown"]
                )
                
                # Adjust based on reliability score
                reliability = result.metadata.get("reliability_score", 0.5)
                result.source_trust_score = base_trust * reliability
            else:
                result.source_trust_score = 1.0
        
        logger.info("Enhanced scoring completed")
        return results
    
    async def _apply_adaptive_mmr_diversification(
        self, 
        query: str, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Apply adaptive MMR based on query complexity"""
        logger.info("Applying adaptive MMR diversification...")
        
        if len(results) <= self.enhanced_config.mmr_top_k:
            return results
        
        # Calculate query complexity
        query_complexity = await self._calculate_query_complexity(query)
        
        # Adapt MMR lambda based on complexity
        base_lambda = self.enhanced_config.mmr_lambda
        adapted_lambda = base_lambda + (query_complexity * self.enhanced_config.mmr_adaptation_factor)
        adapted_lambda = max(0.1, min(0.9, adapted_lambda))  # Clamp between 0.1 and 0.9
        
        logger.info(f"Query complexity: {query_complexity:.3f}, adapted lambda: {adapted_lambda:.3f}")
        
        # Apply MMR with adapted lambda
        selected = []
        remaining = results.copy()
        
        if remaining:
            selected.append(remaining.pop(0))
        
        while len(selected) < self.enhanced_config.mmr_top_k and remaining:
            best_idx = 0
            best_mmr_score = -1
            
            for i, candidate in enumerate(remaining):
                # Calculate relevance score
                relevance = (
                    candidate.rrf_score * 0.3 +
                    candidate.rerank_score * 0.4 +
                    candidate.recency_score * self.enhanced_config.recency_weight +
                    candidate.source_trust_score * 0.2 +
                    candidate.metadata.get("reliability_score", 0.5) * 0.1
                )
                
                # Calculate max similarity
                max_similarity = 0.0
                for selected_result in selected:
                    similarity = self._calculate_content_similarity(
                        candidate.content, 
                        selected_result.content
                    )
                    max_similarity = max(max_similarity, similarity)
                
                # Adaptive MMR score
                mmr_score = adapted_lambda * relevance - (1 - adapted_lambda) * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        logger.info(f"Adaptive MMR diversification: {len(results)} -> {len(selected)} results")
        return selected
    
    async def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score"""
        complexity_indicators = [
            r'\b(compare|contrast|difference|similarity)\b',
            r'\b(why|how|what if|explain)\b',
            r'\b(multiple|several|various|different)\b',
            r'\b(relationship|connection|correlation)\b',
            r'\b(analyze|evaluate|assess)\b',
        ]
        
        complexity_score = 0.0
        for pattern in complexity_indicators:
            if re.search(pattern, query.lower()):
                complexity_score += 0.2
        
        # Length factor
        if len(query.split()) > 15:
            complexity_score += 0.2
        elif len(query.split()) > 10:
            complexity_score += 0.1
        
        return min(1.0, complexity_score)
    
    def _calculate_enhanced_final_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Calculate enhanced final composite scores"""
        logger.info("Calculating enhanced final scores...")
        
        for result in results:
            # Enhanced weighted combination
            result.final_score = (
                result.rrf_score * 0.25 +           # RRF fusion score
                result.rerank_score * 0.35 +        # Reranking score
                result.recency_score * self.enhanced_config.recency_weight +  # Recency
                result.source_trust_score * 0.15 +  # Source trust
                result.metadata.get("reliability_score", 0.5) * 0.1  # Reliability
            )
            
            # Apply minimum score threshold
            if result.final_score < self.enhanced_config.min_rerank_score:
                result.final_score = 0.0
        
        logger.info("Enhanced final score calculation completed")
        return results
    
    async def _conservative_processing(
        self, 
        query: str, 
        retrieval_lists: Dict[str, List[Dict[str, Any]]], 
        **kwargs
    ) -> List[RetrievalResult]:
        """Conservative processing for low robustness queries"""
        logger.info("Using conservative processing for low robustness query")
        
        # Use simpler processing pipeline
        all_results = self._convert_to_retrieval_results(retrieval_lists)
        
        # Simple RRF without advanced features
        rrf_results = await self._apply_weighted_rrf(all_results)
        
        # Limit results and apply basic scoring
        limited_results = rrf_results[:10]  # Conservative limit
        
        for result in limited_results:
            result.final_score = result.rrf_score * 0.8  # Conservative scoring
        
        return limited_results


# Factory function for enhanced EFR layer
def create_enhanced_efr_layer(
    config: EnhancedEFRConfig = None,
    rerank_func: callable = None
) -> EnhancedEFRLayer:
    """Factory function to create enhanced EFR layer"""
    return EnhancedEFRLayer(config=config, rerank_func=rerank_func)

