"""
Optimized Evidence Fusion & Re-ranking (EFR) Layer

This module implements the optimized EFR layer with performance improvements:
- Parallel processing for faster execution
- Intelligent caching to reduce redundant computations
- Adaptive strategies based on query complexity
- Performance monitoring and optimization
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

from .efr_layer import EFRLayer, RetrievalResult, EFRConfig
from .enhanced_efr_layer import EnhancedEFRConfig, EnhancedEFRLayer


@dataclass
class OptimizedEFRConfig(EnhancedEFRConfig):
    """Optimized EFR Configuration"""
    # Performance optimization
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Adaptive strategies
    enable_adaptive_strategies: bool = True
    performance_threshold: float = 0.5  # seconds
    complexity_adaptation: bool = True
    
    # Monitoring
    enable_performance_monitoring: bool = True
    log_performance_metrics: bool = True


class PerformanceMonitor:
    """Performance monitoring and optimization"""
    
    def __init__(self, config: OptimizedEFRConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "total_queries": 0,
            "total_time": 0.0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "parallel_processing_used": 0,
            "adaptive_strategies_used": 0
        }
    
    def start_timer(self) -> float:
        """Start performance timer"""
        return time.time()
    
    def end_timer(self, start_time: float, operation: str = "processing") -> float:
        """End timer and log performance"""
        duration = time.time() - start_time
        self.metrics["total_time"] += duration
        self.metrics["total_queries"] += 1
        self.metrics["avg_response_time"] = self.metrics["total_time"] / self.metrics["total_queries"]
        
        if self.config.log_performance_metrics:
            self.logger.info(f"{operation} completed in {duration:.3f}s")
        
        return duration
    
    def log_cache_hit(self):
        """Log cache hit"""
        self.metrics["cache_hits"] += 1
    
    def log_cache_miss(self):
        """Log cache miss"""
        self.metrics["cache_misses"] += 1
    
    def log_parallel_processing(self):
        """Log parallel processing usage"""
        self.metrics["parallel_processing_used"] += 1
    
    def log_adaptive_strategy(self):
        """Log adaptive strategy usage"""
        self.metrics["adaptive_strategies_used"] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        cache_hit_rate = 0.0
        if self.metrics["cache_hits"] + self.metrics["cache_misses"] > 0:
            cache_hit_rate = self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
        
        return {
            "total_queries": self.metrics["total_queries"],
            "avg_response_time": self.metrics["avg_response_time"],
            "cache_hit_rate": cache_hit_rate,
            "parallel_processing_rate": self.metrics["parallel_processing_used"] / max(self.metrics["total_queries"], 1),
            "adaptive_strategy_rate": self.metrics["adaptive_strategies_used"] / max(self.metrics["total_queries"], 1)
        }


class IntelligentCache:
    """Intelligent caching system"""
    
    def __init__(self, config: OptimizedEFRConfig):
        self.config = config
        self.cache = {}
        self.cache_order = []
        self.logger = logging.getLogger(__name__)
    
    def _generate_cache_key(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate cache key for query and results"""
        # Create a hash of query and result signatures
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        result_signatures = [f"{r.get('content', '')[:50]}_{r.get('score', 0):.3f}" for r in results[:5]]
        results_hash = hashlib.md5("".join(result_signatures).encode()).hexdigest()[:8]
        return f"{query_hash}_{results_hash}"
    
    def get(self, query: str, results: List[Dict[str, Any]]) -> Optional[List[RetrievalResult]]:
        """Get cached results"""
        if not self.config.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(query, results)
        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for query: {query[:50]}...")
            return self.cache[cache_key]
        
        return None
    
    def set(self, query: str, results: List[Dict[str, Any]], processed_results: List[RetrievalResult]):
        """Set cached results"""
        if not self.config.enable_caching:
            return
        
        cache_key = self._generate_cache_key(query, results)
        
        # Implement LRU eviction
        if len(self.cache) >= self.config.cache_size:
            oldest_key = self.cache_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
        
        self.cache[cache_key] = processed_results
        self.cache_order.append(cache_key)
        
        self.logger.debug(f"Cached results for query: {query[:50]}...")


class AdaptiveStrategyManager:
    """Adaptive strategy management"""
    
    def __init__(self, config: OptimizedEFRConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def should_use_parallel_processing(self, results_count: int, query_complexity: float) -> bool:
        """Determine if parallel processing should be used"""
        if not self.config.enable_adaptive_strategies:
            return self.config.enable_parallel_processing
        
        # Use parallel processing for complex queries or large result sets
        return (query_complexity > 0.5 or results_count > 20) and self.config.enable_parallel_processing
    
    def should_use_caching(self, query: str, results_count: int) -> bool:
        """Determine if caching should be used"""
        if not self.config.enable_adaptive_strategies:
            return self.config.enable_caching
        
        # Use caching for repeated patterns or large result sets
        return len(query) > 10 and results_count > 5 and self.config.enable_caching
    
    def get_optimal_workers(self, results_count: int, query_complexity: float) -> int:
        """Get optimal number of workers"""
        if not self.config.enable_adaptive_strategies:
            return self.config.max_workers
        
        # Adaptive worker count based on complexity and result count
        base_workers = min(self.config.max_workers, 4)
        if query_complexity > 0.7 and results_count > 50:
            return min(base_workers + 2, self.config.max_workers)
        elif query_complexity > 0.5 or results_count > 20:
            return min(base_workers + 1, self.config.max_workers)
        else:
            return max(1, base_workers - 1)


class OptimizedEFRLayer(EnhancedEFRLayer):
    """Optimized EFR Layer with performance improvements"""
    
    def __init__(self, config: OptimizedEFRConfig):
        super().__init__(config)
        self.config = config
        self.performance_monitor = PerformanceMonitor(config)
        self.cache = IntelligentCache(config)
        self.strategy_manager = AdaptiveStrategyManager(config)
        self.logger = logging.getLogger(__name__)
    
    async def process_retrieval_results(self, results: List[Dict[str, Any]], query: str, llm_func=None) -> List[RetrievalResult]:
        """Process retrieval results with optimized EFR layer"""
        if not self.config.enable_efr:
            return self._convert_to_retrieval_results(results)
        
        start_time = self.performance_monitor.start_timer()
        self.logger.info(f"Processing {len(results)} results with Optimized EFR layer")
        
        # Check cache first
        cached_results = self.cache.get(query, results)
        if cached_results is not None:
            self.performance_monitor.log_cache_hit()
            self.performance_monitor.end_timer(start_time, "cached processing")
            return cached_results
        
        self.performance_monitor.log_cache_miss()
        
        # Determine processing strategy
        query_complexity = self._calculate_query_complexity(query)
        use_parallel = self.strategy_manager.should_use_parallel_processing(len(results), query_complexity)
        use_caching = self.strategy_manager.should_use_caching(query, len(results))
        
        if use_parallel:
            self.performance_monitor.log_parallel_processing()
            processed_results = await self._parallel_process_results(results, query, llm_func, query_complexity)
        else:
            processed_results = await self._sequential_process_results(results, query, llm_func, query_complexity)
        
        # Cache results if enabled
        if use_caching:
            self.cache.set(query, results, processed_results)
        
        # Log performance metrics
        duration = self.performance_monitor.end_timer(start_time, "optimized processing")
        
        if self.config.enable_performance_monitoring:
            self._log_performance_metrics(duration, len(results), query_complexity)
        
        return processed_results
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score"""
        complexity = 0.0
        
        # Length factor
        if len(query) > 100:
            complexity += 0.3
        elif len(query) > 50:
            complexity += 0.1
        
        # Question count
        question_count = query.count('?')
        complexity += min(question_count * 0.1, 0.3)
        
        # Keyword complexity
        complex_keywords = ['analyze', 'compare', 'evaluate', 'explain', 'discuss', 'critique']
        for keyword in complex_keywords:
            if keyword.lower() in query.lower():
                complexity += 0.1
        
        return min(complexity, 1.0)
    
    async def _parallel_process_results(self, results: List[Dict[str, Any]], query: str, llm_func, complexity: float) -> List[RetrievalResult]:
        """Process results in parallel"""
        # Convert to RetrievalResult objects first
        retrieval_results = self._convert_to_retrieval_results(results)
        
        # Determine optimal worker count
        optimal_workers = self.strategy_manager.get_optimal_workers(len(results), complexity)
        
        # Split results into chunks for parallel processing
        chunk_size = max(1, len(retrieval_results) // optimal_workers)
        chunks = [retrieval_results[i:i + chunk_size] for i in range(0, len(retrieval_results), chunk_size)]
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            tasks = []
            for chunk in chunks:
                task = asyncio.create_task(self._process_chunk_parallel(chunk, query, llm_func))
                tasks.append(task)
            
            # Wait for all tasks to complete
            processed_chunks = await asyncio.gather(*tasks)
        
        # Combine results
        final_results = []
        for chunk_results in processed_chunks:
            final_results.extend(chunk_results)
        
        return final_results
    
    async def _sequential_process_results(self, results: List[Dict[str, Any]], query: str, llm_func, complexity: float) -> List[RetrievalResult]:
        """Process results sequentially"""
        # Use the parent class method for sequential processing
        return await super().process_retrieval_results(results, query, llm_func)
    
    async def _process_chunk_parallel(self, chunk: List[RetrievalResult], query: str, llm_func) -> List[RetrievalResult]:
        """Process a chunk of results in parallel"""
        # Apply reliability estimation
        for result in chunk:
            result.reliability_score = self.reliability_estimator.estimate_reliability(result)
        
        # Apply weighted RRF
        weighted_results = self._apply_enhanced_weighted_rrf(chunk, query)
        
        # Apply scoring
        scored_results = self._apply_enhanced_scoring(weighted_results, query)
        
        return scored_results
    
    def _log_performance_metrics(self, duration: float, result_count: int, complexity: float):
        """Log performance metrics"""
        if self.config.log_performance_metrics:
            self.logger.info(f"Performance metrics - Duration: {duration:.3f}s, "
                           f"Results: {result_count}, Complexity: {complexity:.2f}")
            
            summary = self.performance_monitor.get_performance_summary()
            self.logger.info(f"Performance summary: {summary}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return self.performance_monitor.get_performance_summary()
    
    def reset_performance_metrics(self):
        """Reset performance metrics"""
        self.performance_monitor.metrics = {
            "total_queries": 0,
            "total_time": 0.0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "parallel_processing_used": 0,
            "adaptive_strategies_used": 0
        }
