#!/usr/bin/env python3
"""
Comprehensive EFR Test - 全面EFR测试

这个测试会：
1. 使用英文查询（因为论文是英文的）
2. 配置rerank模型的API key
3. 展示Enhanced EFR层的完整功能
4. 进行全面的性能和质量评估
5. 保存详细的测试结果

Author: AI Assistant
Date: 2024
"""

import asyncio
import os
import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from raganything import RAGAnything, RAGAnythingConfig
from raganything.enhanced_efr_layer import EnhancedEFRConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for EFR systems"""
    
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
            avg_response_time = sum(r['response_time'] for r in successful_results) / len(successful_results)
            avg_response_length = sum(r['response_length'] for r in successful_results) / len(successful_results)
            avg_relevance = sum(r['relevance_score'] for r in successful_results) / len(successful_results)
            avg_completeness = sum(r['completeness_score'] for r in successful_results) / len(successful_results)
            avg_citation_quality = sum(r['citation_quality'] for r in successful_results) / len(successful_results)
            avg_readability = sum(r['readability_score'] for r in successful_results) / len(successful_results)
            avg_structure = sum(r['structure_score'] for r in successful_results) / len(successful_results)
            avg_overall_score = sum(r['overall_score'] for r in successful_results) / len(successful_results)
            
            system_stats[system_type] = {
                "success_rate": len(successful_results) / len(results),
                "performance": {
                    "avg_response_time": avg_response_time,
                    "avg_response_length": avg_response_length,
                    "time_consistency": 1.0 - (max(r['response_time'] for r in successful_results) - min(r['response_time'] for r in successful_results)) / avg_response_time if len(successful_results) > 1 else 1.0
                },
                "content_quality": {
                    "avg_relevance": avg_relevance,
                    "avg_completeness": avg_completeness,
                    "avg_citation_quality": avg_citation_quality,
                    "avg_coherence": sum(r['coherence_score'] for r in successful_results) / len(successful_results),
                    "avg_factual_consistency": sum(r['factual_consistency'] for r in successful_results) / len(successful_results)
                },
                "user_experience": {
                    "avg_readability": avg_readability,
                    "avg_structure": avg_structure,
                    "avg_information_density": sum(r['information_density'] for r in successful_results) / len(successful_results),
                    "avg_answer_directness": sum(r['answer_directness'] for r in successful_results) / len(successful_results),
                    "avg_technical_depth": sum(r['technical_depth'] for r in successful_results) / len(successful_results)
                },
                "overall_score": {
                    "mean": avg_overall_score,
                    "min": min(r['overall_score'] for r in successful_results),
                    "max": max(r['overall_score'] for r in successful_results)
                }
            }
        
        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_queries": len(set(r['query'] for r in self.results)),
            "total_results": len(self.results),
            "system_comparison": system_stats,
            "detailed_results": self.results
        }


async def setup_rag_systems():
    """Setup both standard and enhanced EFR RAG systems with rerank model"""
    
    print("🚀 设置RAG系统...")
    
    # Check API configuration
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST", "https://yinli.one/v1")
    
    if not api_key:
        print("❌ 未找到 API 密钥")
        return None, None
    
    print("✅ API 配置正确")
    
    # Common configuration
    config = RAGAnythingConfig(
        working_dir="./comprehensive_efr_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        max_concurrent_files=1
    )
    
    # LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        """LLM model function with image support"""
        if history_messages is None:
            history_messages = []
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    
    # Vision model function
    def vision_model_func(prompt, image_data=None, **kwargs):
        """Vision model function for image processing"""
        if image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                prompt,
                system_prompt="You are a helpful assistant that analyzes images and provides detailed descriptions.",
                history_messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    } if image_data else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
    
    # Rerank model function
    def rerank_model_func(query, documents, top_n=None, **kwargs):
        """Rerank function using the same API"""
        try:
            # Create rerank prompt
            rerank_prompt = f"""Please rerank the following documents based on their relevance to the query: "{query}"

Documents:
"""
            for i, doc in enumerate(documents):
                rerank_prompt += f"{i+1}. {doc[:200]}...\n"
            
            rerank_prompt += f"""
Please return the reranked results in JSON format with relevance scores (0-1):
[{{"index": 0, "relevance_score": 0.95}}, {{"index": 1, "relevance_score": 0.87}}, ...]
"""
            
            # Call LLM for reranking
            response = openai_complete_if_cache(
                "gpt-4o-mini",
                rerank_prompt,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
            
            # Parse response (simplified)
            try:
                import json
                results = json.loads(response)
                return results
            except:
                # Fallback: simple scoring
                results = []
                for i, doc in enumerate(documents):
                    # Simple keyword matching score
                    query_words = set(query.lower().split())
                    doc_words = set(doc.lower().split())
                    overlap = len(query_words.intersection(doc_words))
                    score = overlap / len(query_words) if query_words else 0.0
                    results.append({"index": i, "relevance_score": score})
                return results
                
        except Exception as e:
            print(f"Rerank error: {e}")
            # Fallback: return original order
            return [{"index": i, "relevance_score": 0.5} for i in range(len(documents))]
    
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )
    
    # 1. Standard RAG (no EFR)
    print("📝 设置标准RAG...")
    standard_rag = RAGAnything(
        config=config,
        enable_efr=False,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    
    # 2. Enhanced EFR RAG with rerank model
    print("🚀 设置Enhanced EFR RAG (with rerank model)...")
    enhanced_efr_config = EnhancedEFRConfig(
        # 启用核心功能
        enable_multilingual=True,
        enable_hybrid_reranking=True,  # 启用混合重排序
        enable_robustness_check=True,  # 启用鲁棒性检查
        enable_source_reliability=True,  # 启用来源可靠性
        enable_adaptive_mmr=True,  # 启用自适应MMR
        mmr_top_k=5,
        
        # 模态权重
        modality_weights={
            "text": 1.0,
            "table": 1.2,
            "equation": 1.1,
            "image": 0.8,
        },
        
        # 语言权重
        language_weights={
            "zh": 1.0,
            "en": 1.0,
        },
    )
    
    enhanced_rag = RAGAnything(
        config=config,
        efr_config=enhanced_efr_config,
        use_enhanced_efr=True,
        enable_efr=True,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    
    # Set rerank function for enhanced RAG
    if hasattr(enhanced_rag, 'lightrag') and enhanced_rag.lightrag:
        enhanced_rag.lightrag.rerank_model_func = rerank_model_func
        print("✅ Rerank model configured")
    
    print("✅ 两个RAG系统初始化完成")
    return standard_rag, enhanced_rag


async def process_document(rag, doc_path):
    """处理文档"""
    
    print(f"\n📄 处理文档: {doc_path.name}")
    
    try:
        result = await rag.process_document_complete(
            file_path=str(doc_path),
            output_dir="./comprehensive_efr_output",
            parse_method="auto"
        )
        print("✅ 文档处理完成")
        # Handle case where result might be None (already processed)
        if result is None:
            print("📄 文档已经处理过，使用现有数据")
            return True
        return result.get("success", False)
    except Exception as e:
        print(f"❌ 文档处理失败: {e}")
        return False


async def test_query_comprehensive(rag, query: str, system_name: str, evaluator: ComprehensiveEvaluator) -> Dict[str, Any]:
    """Test a single query with comprehensive metrics"""
    print(f"\n{'='*60}")
    print(f"Testing {system_name}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if system_name == "Standard RAG":
            response = await rag.aquery(query, mode="hybrid")
        else:
            response = await rag.aquery_enhanced(query, mode="hybrid", use_efr=True)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Basic metrics
        result = {
            "query": query,
            "system_type": system_name,
            "success": True,
            "response_time": response_time,
            "response_length": len(response),
            "response": response
        }
        
        # Content quality analysis
        content_metrics = evaluator.analyze_content_quality(query, response)
        result.update(content_metrics)
        
        # User experience analysis
        ux_metrics = evaluator.analyze_user_experience(response)
        result.update(ux_metrics)
        
        # Calculate overall score
        result['overall_score'] = evaluator.calculate_overall_score(result)
        
        print(f"✅ Success!")
        print(f"Response time: {response_time:.3f}s")
        print(f"Response length: {len(response)} characters")
        print(f"Overall Score: {result['overall_score']:.3f}")
        print(f"Relevance: {result['relevance_score']:.3f}")
        print(f"Completeness: {result['completeness_score']:.3f}")
        print(f"Citation Quality: {result['citation_quality']:.3f}")
        print(f"Readability: {result['readability_score']:.3f}")
        print(f"Structure: {result['structure_score']:.3f}")
        print(f"Response preview: {response[:200]}...")
        
        return result
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"❌ Failed: {e}")
        
        return {
            "query": query,
            "system_type": system_name,
            "success": False,
            "error_message": str(e),
            "response_time": response_time,
            "response_length": 0,
            "overall_score": 0.0
        }


async def run_comprehensive_test():
    """Run comprehensive EFR test"""
    
    print("🔬 Comprehensive EFR Layer Test")
    print("=" * 80)
    
    # Check document
    doc_path = Path("example_doc/2502.00989v1.pdf")
    if not doc_path.exists():
        print(f"❌ Document not found: {doc_path}")
        return
    
    # Setup systems
    standard_rag, enhanced_rag = await setup_rag_systems()
    if not standard_rag or not enhanced_rag:
        print("❌ Failed to setup RAG systems")
        return
    
    # Process document
    print(f"\n📄 Processing document: {doc_path}")
    success = await process_document(standard_rag, doc_path)
    if not success:
        print("❌ Failed to process document")
        return
    
    # Both systems use the same storage directory, so no need to copy
    print("✅ Both systems will use the same processed data")
    
    # Test queries
    test_queries = [
        "What is the main contribution of this paper?",
        "What are the key technologies used in ChartCitor?",
        "How does the multi-agent framework work?",
        "What are the experimental results and performance metrics?",
        "How does ChartCitor compare with other baseline methods?"
    ]
    
    evaluator = ComprehensiveEvaluator()
    
    # Test both systems
    for system_name, rag in [("Standard RAG", standard_rag), ("Enhanced EFR", enhanced_rag)]:
        print(f"\n🔍 Testing {system_name}...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}/{len(test_queries)}")
            result = await test_query_comprehensive(rag, query, system_name, evaluator)
            evaluator.results.append(result)
    
    # Generate comprehensive report
    print(f"\n📊 Generating Comprehensive Report...")
    report = evaluator.generate_comprehensive_report()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"comprehensive_efr_report_{timestamp}.json"
    
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
            print(f"  Overall Score: {stats['overall_score']['mean']:.3f}")
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
    asyncio.run(run_comprehensive_test())
