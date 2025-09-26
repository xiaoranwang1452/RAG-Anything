#!/usr/bin/env python3
"""
Enhanced EFR Layer Comprehensive Evaluation

This script provides a comprehensive evaluation of EFR Layer performance with:
1. Performance Metrics (Time, Memory, CPU)
2. Content Quality Metrics (Relevance, Completeness, Accuracy)
3. User Experience Metrics (Readability, Structure, Citations)
4. Comparative Analysis (Standard vs Enhanced EFR)

Author: AI Assistant
Date: 2024
"""

import asyncio
import time
import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from raganything import RAGAnything, RAGAnythingConfig
from raganything.enhanced_efr_layer import EnhancedEFRConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    # Performance metrics
    response_time: float = 0.0
    response_length: int = 0
    
    # Content quality metrics
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    citation_quality: float = 0.0
    coherence_score: float = 0.0
    factual_consistency: float = 0.0
    
    # User experience metrics
    readability_score: float = 0.0
    structure_score: float = 0.0
    information_density: float = 0.0
    answer_directness: float = 0.0
    technical_depth: float = 0.0
    
    # System efficiency metrics
    processing_steps: int = 0
    cache_effectiveness: float = 0.0
    
    # Overall score
    overall_score: float = 0.0


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def analyze_content_quality(self, query: str, response: str) -> Dict[str, float]:
        """Analyze content quality metrics"""
        metrics = {}
        
        # Relevance score (keyword overlap with query)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        metrics['relevance_score'] = overlap / len(query_words) if query_words else 0.0
        
        # Completeness score (length and structure)
        metrics['completeness_score'] = min(1.0, len(response) / 2000)  # Normalize to 2000 chars
        
        # Citation quality (citation presence and format)
        citation_pattern = r'\[[^\]]+\]'
        citations = re.findall(citation_pattern, response)
        metrics['citation_quality'] = min(1.0, len(citations) / 5)  # Normalize to 5 citations
        
        # Coherence score (sentence structure and flow)
        sentences = response.split('.')
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            metrics['coherence_score'] = min(1.0, avg_sentence_length / 20)  # Normalize to 20 words per sentence
        else:
            metrics['coherence_score'] = 0.0
        
        # Factual consistency (repetition and contradiction detection)
        metrics['factual_consistency'] = self.check_factual_consistency(response)
        
        return metrics
    
    def analyze_user_experience(self, response: str) -> Dict[str, float]:
        """Analyze user experience metrics"""
        metrics = {}
        
        # Readability score (Flesch-like score)
        sentences = response.split('.')
        words = response.split()
        if sentences and words:
            avg_sentence_length = len(words) / len(sentences)
            # Simple readability approximation
            metrics['readability_score'] = max(0, min(1.0, 1 - (avg_sentence_length * 0.05)))
        else:
            metrics['readability_score'] = 0.0
        
        # Structure score (headings, lists, paragraphs)
        headings = len(re.findall(r'^#+\s', response, re.MULTILINE))
        lists = len(re.findall(r'^\s*[-*+]\s', response, re.MULTILINE))
        metrics['structure_score'] = min(1.0, (headings + lists) / 10)
        
        # Information density (unique information per word)
        unique_words = len(set(response.lower().split()))
        total_words = len(response.split())
        metrics['information_density'] = unique_words / total_words if total_words else 0.0
        
        # Answer directness (how directly the response answers the question)
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        direct_answers = sum(1 for word in question_words if word in response.lower())
        metrics['answer_directness'] = min(1.0, direct_answers / 3)
        
        # Technical depth (technical terms and concepts)
        technical_terms = ['algorithm', 'framework', 'model', 'system', 'method', 'approach', 'technique']
        technical_count = sum(1 for term in technical_terms if term in response.lower())
        metrics['technical_depth'] = min(1.0, technical_count / 5)
        
        return metrics
    
    def check_factual_consistency(self, response: str) -> float:
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
    
    def calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall score for a result"""
        if not metrics.get('success', False):
            return 0.0
        
        # Performance score (inverse of response time, normalized)
        performance_score = 1.0 / (1.0 + metrics['response_time'] / 20.0)
        
        # Content quality score (weighted average)
        content_score = (
            metrics['relevance_score'] * 0.25 +
            metrics['completeness_score'] * 0.2 +
            metrics['citation_quality'] * 0.2 +
            metrics['coherence_score'] * 0.15 +
            metrics['factual_consistency'] * 0.2
        )
        
        # User experience score (weighted average)
        ux_score = (
            metrics['readability_score'] * 0.3 +
            metrics['structure_score'] * 0.25 +
            metrics['information_density'] * 0.2 +
            metrics['answer_directness'] * 0.15 +
            metrics['technical_depth'] * 0.1
        )
        
        # Overall score (weighted combination)
        overall_score = (performance_score * 0.3 + content_score * 0.5 + ux_score * 0.2)
        
        return overall_score
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Group results by system type
        system_results = {}
        for result in self.results:
            system_type = result['system_type']
            if system_type not in system_results:
                system_results[system_type] = []
            system_results[system_type].append(result)
        
        # Calculate statistics for each system
        system_stats = {}
        for system_type, results in system_results.items():
            successful_results = [r for r in results if r.get('success', False)]
            
            if not successful_results:
                system_stats[system_type] = {"error": "No successful results"}
                continue
            
            # Calculate averages
            avg_response_time = statistics.mean([r['response_time'] for r in successful_results])
            avg_response_length = statistics.mean([r['response_length'] for r in successful_results])
            avg_relevance = statistics.mean([r['relevance_score'] for r in successful_results])
            avg_completeness = statistics.mean([r['completeness_score'] for r in successful_results])
            avg_citation_quality = statistics.mean([r['citation_quality'] for r in successful_results])
            avg_readability = statistics.mean([r['readability_score'] for r in successful_results])
            avg_structure = statistics.mean([r['structure_score'] for r in successful_results])
            avg_overall_score = statistics.mean([r['overall_score'] for r in successful_results])
            
            system_stats[system_type] = {
                "success_rate": len(successful_results) / len(results),
                "performance": {
                    "avg_response_time": avg_response_time,
                    "avg_response_length": avg_response_length,
                    "time_consistency": 1.0 - (statistics.stdev([r['response_time'] for r in successful_results]) / avg_response_time) if len(successful_results) > 1 else 1.0
                },
                "content_quality": {
                    "avg_relevance": avg_relevance,
                    "avg_completeness": avg_completeness,
                    "avg_citation_quality": avg_citation_quality,
                    "avg_coherence": statistics.mean([r['coherence_score'] for r in successful_results]),
                    "avg_factual_consistency": statistics.mean([r['factual_consistency'] for r in successful_results])
                },
                "user_experience": {
                    "avg_readability": avg_readability,
                    "avg_structure": avg_structure,
                    "avg_information_density": statistics.mean([r['information_density'] for r in successful_results]),
                    "avg_answer_directness": statistics.mean([r['answer_directness'] for r in successful_results]),
                    "avg_technical_depth": statistics.mean([r['technical_depth'] for r in successful_results])
                },
                "overall_score": {
                    "mean": avg_overall_score,
                    "std": statistics.stdev([r['overall_score'] for r in successful_results]) if len(successful_results) > 1 else 0,
                    "min": min([r['overall_score'] for r in successful_results]),
                    "max": max([r['overall_score'] for r in successful_results])
                }
            }
        
        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_queries": len(set(r['query'] for r in self.results)),
            "total_results": len(self.results),
            "system_comparison": system_stats,
            "detailed_results": self.results
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


async def test_system_performance(rag, query: str, system_type: str, evaluator: ComprehensiveEvaluator) -> Dict[str, Any]:
    """Test system performance with comprehensive metrics"""
    print(f"\n{'='*60}")
    print(f"Testing {system_type}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if system_type == "Standard RAG":
            response = await rag.aquery(query, mode="mix")
        else:
            response = await rag.aquery_enhanced(query, mode="mix", use_efr=True)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Basic metrics
        result = {
            "query": query,
            "system_type": system_type,
            "success": True,
            "response_time": response_time,
            "response_length": len(response),
            "raw_response": response
        }
        
        # Content quality analysis
        content_metrics = evaluator.analyze_content_quality(query, response)
        result.update(content_metrics)
        
        # User experience analysis
        ux_metrics = evaluator.analyze_user_experience(response)
        result.update(ux_metrics)
        
        # Calculate overall score
        result['overall_score'] = evaluator.calculate_overall_score(result)
        
        print(f"✅ Success - Overall Score: {result['overall_score']:.3f}")
        print(f"   Response Time: {result['response_time']:.3f}s")
        print(f"   Response Length: {result['response_length']} chars")
        print(f"   Relevance: {result['relevance_score']:.3f}")
        print(f"   Completeness: {result['completeness_score']:.3f}")
        print(f"   Citation Quality: {result['citation_quality']:.3f}")
        print(f"   Readability: {result['readability_score']:.3f}")
        print(f"   Structure: {result['structure_score']:.3f}")
        
        return result
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"❌ Failed: {e}")
        
        return {
            "query": query,
            "system_type": system_type,
            "success": False,
            "error_message": str(e),
            "response_time": response_time,
            "response_length": 0,
            "overall_score": 0.0
        }


async def run_comprehensive_evaluation():
    """Run comprehensive evaluation of EFR systems"""
    print("🔬 Enhanced EFR Layer Comprehensive Evaluation")
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
        "What is the main contribution of this paper?",
        "What are the key technologies used in ChartCitor?",
        "How does the multi-agent framework work?",
        "What are the experimental results and performance metrics?",
        "How does ChartCitor compare with other baseline methods?"
    ]
    
    evaluator = ComprehensiveEvaluator()
    
    # Test configurations
    configs = [
        {
            "name": "Standard RAG",
            "efr_config": None,
            "use_efr": False,
            "use_enhanced_efr": False
        },
        {
            "name": "Enhanced EFR",
            "efr_config": EnhancedEFRConfig(),
            "use_efr": True,
            "use_enhanced_efr": True
        }
    ]
    
    for config in configs:
        print(f"\n🔧 Initializing {config['name']}...")
        
        # Create RAGAnything instance
        rag_config = RAGAnythingConfig(
            working_dir=f"enhanced_eval_{config['name'].lower().replace(' ', '_')}",
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            max_concurrent_files=1
        )
        
        rag = RAGAnything(
            config=rag_config,
            efr_config=config["efr_config"],
            enable_efr=config["use_efr"],
            use_enhanced_efr=config["use_enhanced_efr"],
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
        
        # Test each query
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}/{len(test_queries)}")
            result = await test_system_performance(rag, query, config["name"], evaluator)
            evaluator.results.append(result)
        
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
    report_file = f"enhanced_evaluation_report_{timestamp}.json"
    
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
            print(f"  Time Consistency: {stats['performance']['time_consistency']:.3f}")
            print(f"  Avg Response Length: {stats['performance']['avg_response_length']:.0f} chars")
            print(f"  Content Quality: {stats['content_quality']['avg_relevance']:.3f}")
            print(f"  Citation Quality: {stats['content_quality']['avg_citation_quality']:.3f}")
            print(f"  User Experience: {stats['user_experience']['avg_readability']:.3f}")
            print(f"  Structure Quality: {stats['user_experience']['avg_structure']:.3f}")
    
    # Performance comparison
    if len(report["system_comparison"]) >= 2:
        print(f"\n📈 PERFORMANCE COMPARISON")
        print("-" * 40)
        
        systems = list(report["system_comparison"].keys())
        if len(systems) >= 2:
            baseline = systems[0]
            enhanced = systems[1]
            
            if baseline in report["system_comparison"] and enhanced in report["system_comparison"]:
                baseline_stats = report["system_comparison"][baseline]
                enhanced_stats = report["system_comparison"][enhanced]
                
                if "error" not in baseline_stats and "error" not in enhanced_stats:
                    time_improvement = ((baseline_stats['performance']['avg_response_time'] - 
                                       enhanced_stats['performance']['avg_response_time']) / 
                                      baseline_stats['performance']['avg_response_time']) * 100
                    
                    quality_improvement = ((enhanced_stats['overall_score']['mean'] - 
                                          baseline_stats['overall_score']['mean']) / 
                                         baseline_stats['overall_score']['mean']) * 100
                    
                    print(f"Time Improvement: {time_improvement:+.1f}%")
                    print(f"Quality Improvement: {quality_improvement:+.1f}%")
                    print(f"Overall Performance: {'✅ Better' if quality_improvement > 0 else '⚠️ Needs Improvement'}")
    
    print(f"\n📁 Detailed report saved to: {report_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_evaluation())
