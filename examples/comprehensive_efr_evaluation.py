#!/usr/bin/env python3
"""
Comprehensive EFR Layer Evaluation Framework

This script provides a comprehensive evaluation of EFR Layer performance including:
1. Performance Metrics (Time, Memory, CPU)
2. Content Quality Metrics (Relevance, Completeness, Accuracy)
3. User Experience Metrics (Readability, Structure, Citations)
4. System Efficiency Metrics (Cache Hit Rate, Processing Steps)
5. Comparative Analysis (Standard vs Enhanced vs Optimized)

Author: AI Assistant
Date: 2024
"""

import asyncio
import time
import os
import sys
import json
import psutil
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from raganything import RAGAnything, RAGAnythingConfig
from raganything.efr_layer import EFRConfig
from raganything.enhanced_efr_layer import EnhancedEFRConfig
from raganything.optimized_efr_layer import OptimizedEFRConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation"""
    response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    processing_steps: int = 0
    cache_hit_rate: float = 0.0
    api_calls_count: int = 0
    tokens_used: int = 0


@dataclass
class ContentQualityMetrics:
    """Content quality metrics for evaluation"""
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    coherence_score: float = 0.0
    citation_quality: float = 0.0
    factual_consistency: float = 0.0


@dataclass
class UserExperienceMetrics:
    """User experience metrics for evaluation"""
    readability_score: float = 0.0
    structure_score: float = 0.0
    information_density: float = 0.0
    answer_directness: float = 0.0
    technical_depth: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    query: str
    system_type: str
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    content_quality: ContentQualityMetrics = field(default_factory=ContentQualityMetrics)
    user_experience: UserExperienceMetrics = field(default_factory=UserExperienceMetrics)
    success: bool = False
    error_message: str = ""
    raw_response: str = ""


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for EFR systems"""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
        self.baseline_metrics = {}
        
    async def evaluate_system(
        self, 
        rag: RAGAnything, 
        query: str, 
        system_type: str
    ) -> EvaluationResult:
        """Evaluate a single system with comprehensive metrics"""
        
        result = EvaluationResult(query=query, system_type=system_type)
        
        # Start monitoring
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        start_time = time.time()
        
        try:
            # Execute query
            if system_type == "Standard RAG":
                response = await rag.aquery(query, mode="mix")
            else:
                response = await rag.aquery_enhanced(query, mode="mix", use_efr=True)
            
            # End monitoring
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = process.cpu_percent()
            
            # Performance metrics
            result.performance.response_time = end_time - start_time
            result.performance.memory_usage_mb = end_memory - start_memory
            result.performance.cpu_usage_percent = end_cpu - start_cpu
            
            # Content analysis
            result.raw_response = response
            result.success = True
            
            # Analyze content quality
            result.content_quality = await self._analyze_content_quality(query, response)
            
            # Analyze user experience
            result.user_experience = await self._analyze_user_experience(response)
            
            # System-specific metrics
            if hasattr(rag, 'efr_layer') and rag.efr_layer:
                result.performance.processing_steps = await self._count_processing_steps(rag.efr_layer)
                result.performance.cache_hit_rate = await self._calculate_cache_hit_rate(rag.efr_layer)
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.performance.response_time = time.time() - start_time
        
        return result
    
    async def _analyze_content_quality(self, query: str, response: str) -> ContentQualityMetrics:
        """Analyze content quality metrics"""
        metrics = ContentQualityMetrics()
        
        # Relevance score (keyword overlap with query)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        metrics.relevance_score = overlap / len(query_words) if query_words else 0.0
        
        # Completeness score (length and structure)
        metrics.completeness_score = min(1.0, len(response) / 2000)  # Normalize to 2000 chars
        
        # Accuracy score (citation presence and format)
        citation_pattern = r'\[[^\]]+\]'
        citations = re.findall(citation_pattern, response)
        metrics.citation_quality = min(1.0, len(citations) / 5)  # Normalize to 5 citations
        
        # Coherence score (sentence structure and flow)
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        metrics.coherence_score = min(1.0, avg_sentence_length / 20)  # Normalize to 20 words per sentence
        
        # Factual consistency (repetition and contradiction detection)
        metrics.factual_consistency = self._check_factual_consistency(response)
        
        return metrics
    
    async def _analyze_user_experience(self, response: str) -> UserExperienceMetrics:
        """Analyze user experience metrics"""
        metrics = UserExperienceMetrics()
        
        # Readability score (Flesch-like score)
        sentences = response.split('.')
        words = response.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        if sentences and words:
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            metrics.readability_score = max(0, min(1.0, 1 - (avg_sentence_length * 0.1 + avg_syllables_per_word * 0.2)))
        
        # Structure score (headings, lists, paragraphs)
        headings = len(re.findall(r'^#+\s', response, re.MULTILINE))
        lists = len(re.findall(r'^\s*[-*+]\s', response, re.MULTILINE))
        metrics.structure_score = min(1.0, (headings + lists) / 10)
        
        # Information density (unique information per word)
        unique_words = len(set(response.lower().split()))
        total_words = len(response.split())
        metrics.information_density = unique_words / total_words if total_words else 0.0
        
        # Answer directness (how directly the response answers the question)
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        direct_answers = sum(1 for word in question_words if word in response.lower())
        metrics.answer_directness = min(1.0, direct_answers / 3)
        
        # Technical depth (technical terms and concepts)
        technical_terms = ['algorithm', 'framework', 'model', 'system', 'method', 'approach', 'technique']
        technical_count = sum(1 for term in technical_terms if term in response.lower())
        metrics.technical_depth = min(1.0, technical_count / 5)
        
        return metrics
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _check_factual_consistency(self, response: str) -> float:
        """Check for factual consistency in response"""
        # Simple heuristic: check for contradictory statements
        contradictions = [
            ('yes', 'no'), ('true', 'false'), ('correct', 'incorrect'),
            ('increase', 'decrease'), ('positive', 'negative')
        ]
        
        response_lower = response.lower()
        contradiction_count = 0
        
        for pos, neg in contradictions:
            if pos in response_lower and neg in response_lower:
                contradiction_count += 1
        
        # Return consistency score (higher is better)
        return max(0.0, 1.0 - (contradiction_count * 0.2))
    
    async def _count_processing_steps(self, efr_layer) -> int:
        """Count processing steps in EFR layer"""
        # This would need to be implemented based on the actual EFR layer
        # For now, return a default value
        return 8 if hasattr(efr_layer, 'enhanced_config') else 4
    
    async def _calculate_cache_hit_rate(self, efr_layer) -> float:
        """Calculate cache hit rate for EFR layer"""
        # This would need to be implemented based on the actual EFR layer
        # For now, return a default value
        return 0.3 if hasattr(efr_layer, 'smart_cache') else 0.0
    
    def calculate_overall_score(self, result: EvaluationResult) -> float:
        """Calculate overall score for a result"""
        if not result.success:
            return 0.0
        
        # Weighted combination of metrics
        performance_score = (
            (1.0 / (1.0 + result.performance.response_time / 20.0)) * 0.3 +  # Time (inverse)
            (1.0 / (1.0 + result.performance.memory_usage_mb / 100.0)) * 0.2 +  # Memory (inverse)
            (1.0 - result.performance.cpu_usage_percent / 100.0) * 0.1  # CPU (inverse)
        )
        
        content_score = (
            result.content_quality.relevance_score * 0.25 +
            result.content_quality.completeness_score * 0.2 +
            result.content_quality.citation_quality * 0.2 +
            result.content_quality.coherence_score * 0.15 +
            result.content_quality.factual_consistency * 0.2
        )
        
        ux_score = (
            result.user_experience.readability_score * 0.3 +
            result.user_experience.structure_score * 0.25 +
            result.user_experience.information_density * 0.2 +
            result.user_experience.answer_directness * 0.15 +
            result.user_experience.technical_depth * 0.1
        )
        
        return (performance_score * 0.3 + content_score * 0.5 + ux_score * 0.2)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Group results by system type
        system_results = {}
        for result in self.results:
            if result.system_type not in system_results:
                system_results[result.system_type] = []
            system_results[result.system_type].append(result)
        
        # Calculate statistics for each system
        system_stats = {}
        for system_type, results in system_results.items():
            successful_results = [r for r in results if r.success]
            
            if not successful_results:
                system_stats[system_type] = {"error": "No successful results"}
                continue
            
            # Performance statistics
            response_times = [r.performance.response_time for r in successful_results]
            memory_usage = [r.performance.memory_usage_mb for r in successful_results]
            
            # Content quality statistics
            relevance_scores = [r.content_quality.relevance_score for r in successful_results]
            completeness_scores = [r.content_quality.completeness_score for r in successful_results]
            citation_quality = [r.content_quality.citation_quality for r in successful_results]
            
            # User experience statistics
            readability_scores = [r.user_experience.readability_score for r in successful_results]
            structure_scores = [r.user_experience.structure_score for r in successful_results]
            
            # Overall scores
            overall_scores = [self.calculate_overall_score(r) for r in successful_results]
            
            system_stats[system_type] = {
                "success_rate": len(successful_results) / len(results),
                "performance": {
                    "avg_response_time": statistics.mean(response_times),
                    "std_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                    "avg_memory_usage": statistics.mean(memory_usage),
                    "avg_cpu_usage": statistics.mean([r.performance.cpu_usage_percent for r in successful_results])
                },
                "content_quality": {
                    "avg_relevance": statistics.mean(relevance_scores),
                    "avg_completeness": statistics.mean(completeness_scores),
                    "avg_citation_quality": statistics.mean(citation_quality),
                    "avg_coherence": statistics.mean([r.content_quality.coherence_score for r in successful_results])
                },
                "user_experience": {
                    "avg_readability": statistics.mean(readability_scores),
                    "avg_structure": statistics.mean(structure_scores),
                    "avg_information_density": statistics.mean([r.user_experience.information_density for r in successful_results])
                },
                "overall_score": {
                    "mean": statistics.mean(overall_scores),
                    "std": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                    "min": min(overall_scores),
                    "max": max(overall_scores)
                }
            }
        
        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_queries": len(set(r.query for r in self.results)),
            "total_results": len(self.results),
            "system_comparison": system_stats,
            "detailed_results": [
                {
                    "query": r.query,
                    "system_type": r.system_type,
                    "success": r.success,
                    "overall_score": self.calculate_overall_score(r),
                    "performance": {
                        "response_time": r.performance.response_time,
                        "memory_usage_mb": r.performance.memory_usage_mb,
                        "cpu_usage_percent": r.performance.cpu_usage_percent
                    },
                    "content_quality": {
                        "relevance": r.content_quality.relevance_score,
                        "completeness": r.content_quality.completeness_score,
                        "citation_quality": r.content_quality.citation_quality,
                        "coherence": r.content_quality.coherence_score
                    },
                    "user_experience": {
                        "readability": r.user_experience.readability_score,
                        "structure": r.user_experience.structure_score,
                        "information_density": r.user_experience.information_density
                    }
                }
                for r in self.results
            ]
        }


async def create_llm_model_func():
    """Create LLM model function"""
    import openai
    
    async def llm_model_func(messages, **kwargs):
        try:
            response = await openai.AsyncOpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.1),
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {e}")
            return "Error: Unable to generate response"
    
    return llm_model_func


async def create_vision_model_func():
    """Create vision model function"""
    import openai
    
    async def vision_model_func(messages, **kwargs):
        try:
            response = await openai.AsyncOpenAI().chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.1),
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Vision API error: {e}")
            return "Error: Unable to process image"
    
    return vision_model_func


async def create_embedding_func():
    """Create embedding function"""
    import openai
    
    async def embedding_func(texts, **kwargs):
        try:
            response = await openai.AsyncOpenAI().embeddings.create(
                model="text-embedding-3-large",
                input=texts if isinstance(texts, list) else [texts],
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Embedding API error: {e}")
            return [[0.0] * 3072] * (len(texts) if isinstance(texts, list) else 1)
    
    # Add embedding_dim attribute
    embedding_func.embedding_dim = 3072
    return embedding_func


async def run_comprehensive_evaluation():
    """Run comprehensive evaluation of EFR systems"""
    print("🔬 Comprehensive EFR Layer Evaluation Framework")
    print("=" * 80)
    
    # Check if we have a document to process
    doc_path = "example_doc/2502.00989v1.pdf"
    if not os.path.exists(doc_path):
        print(f"❌ Document not found: {doc_path}")
        return
    
    # Create model functions
    llm_model_func = await create_llm_model_func()
    vision_model_func = await create_vision_model_func()
    embedding_func = await create_embedding_func()
    
    # Test queries with different complexity levels
    test_queries = [
        {
            "query": "What is the main contribution of this paper?",
            "complexity": "simple",
            "expected_length": "medium"
        },
        {
            "query": "What are the key technologies used in ChartCitor?",
            "complexity": "medium",
            "expected_length": "long"
        },
        {
            "query": "How does the multi-agent framework work?",
            "complexity": "complex",
            "expected_length": "long"
        },
        {
            "query": "What are the experimental results and performance metrics?",
            "complexity": "medium",
            "expected_length": "medium"
        },
        {
            "query": "How does ChartCitor compare with other baseline methods?",
            "complexity": "complex",
            "expected_length": "medium"
        }
    ]
    
    # System configurations
    systems = [
        {
            "name": "Standard RAG",
            "config": None,
            "use_efr": False,
            "use_enhanced_efr": False,
            "use_optimized_efr": False
        },
        {
            "name": "Enhanced EFR",
            "config": EnhancedEFRConfig(),
            "use_efr": True,
            "use_enhanced_efr": True,
            "use_optimized_efr": False
        },
        {
            "name": "Optimized EFR",
            "config": OptimizedEFRConfig(),
            "use_efr": True,
            "use_enhanced_efr": False,
            "use_optimized_efr": True
        }
    ]
    
    evaluator = ComprehensiveEvaluator()
    
    for system in systems:
        print(f"\n🔧 Initializing {system['name']}...")
        
        # Create RAGAnything instance
        rag_config = RAGAnythingConfig(
            working_dir=f"comprehensive_eval_{system['name'].lower().replace(' ', '_')}",
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            max_concurrent_files=1
        )
        
        rag = RAGAnything(
            config=rag_config,
            efr_config=system["config"],
            enable_efr=system["use_efr"],
            use_enhanced_efr=system["use_enhanced_efr"],
            use_optimized_efr=system["use_optimized_efr"],
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func
        )
        
        # Process document
        print(f"📄 Processing document: {doc_path}")
        process_result = await rag.process_document_complete(doc_path)
        
        if not process_result.get("success", False):
            print(f"❌ Failed to process document: {process_result.get('error', 'Unknown error')}")
            continue
        
        print(f"✅ Document processed successfully")
        
        # Evaluate each query
        for i, query_info in enumerate(test_queries, 1):
            query = query_info["query"]
            print(f"\n📝 Evaluating Query {i}/{len(test_queries)}: {query[:50]}...")
            
            result = await evaluator.evaluate_system(rag, query, system["name"])
            evaluator.results.append(result)
            
            if result.success:
                overall_score = evaluator.calculate_overall_score(result)
                print(f"✅ Success - Overall Score: {overall_score:.3f}")
                print(f"   Response Time: {result.performance.response_time:.3f}s")
                print(f"   Content Quality: {result.content_quality.relevance_score:.3f}")
                print(f"   User Experience: {result.user_experience.readability_score:.3f}")
            else:
                print(f"❌ Failed: {result.error_message}")
        
        # Cleanup
        try:
            await rag.finalize_storages()
        except:
            pass
    
    # Generate comprehensive report
    print(f"\n📊 Generating Comprehensive Report...")
    report = evaluator.generate_comprehensive_report()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"comprehensive_evaluation_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    if "system_comparison" in report:
        for system_name, stats in report["system_comparison"].items():
            if "error" in stats:
                print(f"\n❌ {system_name}: {stats['error']}")
                continue
            
            print(f"\n🔍 {system_name}:")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            print(f"  Overall Score: {stats['overall_score']['mean']:.3f} ± {stats['overall_score']['std']:.3f}")
            print(f"  Avg Response Time: {stats['performance']['avg_response_time']:.3f}s")
            print(f"  Avg Memory Usage: {stats['performance']['avg_memory_usage']:.1f}MB")
            print(f"  Content Quality: {stats['content_quality']['avg_relevance']:.3f}")
            print(f"  User Experience: {stats['user_experience']['avg_readability']:.3f}")
    
    print(f"\n📁 Detailed report saved to: {report_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_evaluation())
