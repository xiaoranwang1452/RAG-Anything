#!/usr/bin/env python3
"""
Simple EFR Layer Demo

This demo shows the core functionality of the Enhanced EFR Layer
without requiring external API keys. It demonstrates:

1. Evidence Fusion & Re-Ranking
2. Multimodal processing
3. Source reliability estimation
4. Robustness evaluation

Author: AI Assistant
Date: 2024
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from raganything.enhanced_efr_layer import EnhancedEFRConfig, EnhancedEFRLayer, RetrievalResult


class MockReranker:
    """Mock reranker for demonstration"""
    
    async def __call__(self, query: str, documents: List[str], **kwargs):
        """Mock reranking function"""
        results = []
        for i, doc in enumerate(documents):
            # Simple scoring based on query-document similarity
            query_words = set(query.lower().split())
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


async def create_sample_retrieval_data():
    """Create sample retrieval data for demonstration"""
    
    # Sample documents from different sources
    sample_docs = {
        "vector": [
            {
                "content": "深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的复杂模式。深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
                "score": 0.85,
                "chunk_id": "vec_001",
                "doc_id": "doc_001",
                "created_at": time.time() - 86400,  # 1 day ago
                "source_type": "academic"
            },
            {
                "content": "卷积神经网络(CNN)是深度学习中用于图像处理的重要架构。CNN通过卷积层、池化层和全连接层来提取图像特征，在计算机视觉任务中表现出色。",
                "score": 0.78,
                "chunk_id": "vec_002", 
                "doc_id": "doc_001",
                "created_at": time.time() - 172800,  # 2 days ago
                "source_type": "academic"
            },
            {
                "content": "循环神经网络(RNN)是处理序列数据的神经网络架构。RNN通过隐藏状态来记忆之前的信息，在自然语言处理和时序预测中广泛应用。",
                "score": 0.72,
                "chunk_id": "vec_003",
                "doc_id": "doc_002", 
                "created_at": time.time() - 259200,  # 3 days ago
                "source_type": "academic"
            }
        ],
        "entity": [
            {
                "content": "深度学习(Deep Learning)是一种基于人工神经网络的机器学习方法，由Geoffrey Hinton等人在2006年提出。",
                "score": 0.90,
                "entity_name": "深度学习",
                "chunk_id": "ent_001",
                "doc_id": "doc_001",
                "created_at": time.time() - 43200,  # 12 hours ago
                "source_type": "official"
            },
            {
                "content": "神经网络(Neural Network)是模拟生物神经系统的计算模型，由多个相互连接的节点(神经元)组成。",
                "score": 0.88,
                "entity_name": "神经网络",
                "chunk_id": "ent_002",
                "doc_id": "doc_002",
                "created_at": time.time() - 86400,  # 1 day ago
                "source_type": "academic"
            }
        ],
        "relation": [
            {
                "content": "深度学习是机器学习的一个子集，机器学习是人工智能的一个分支。深度学习通过多层神经网络实现端到端的学习。",
                "score": 0.82,
                "relation_type": "is_a",
                "chunk_id": "rel_001",
                "doc_id": "doc_001",
                "created_at": time.time() - 129600,  # 1.5 days ago
                "source_type": "academic"
            }
        ],
        "chunk": [
            {
                "content": "Transformer架构是2017年由Vaswani等人提出的革命性模型，它完全基于注意力机制，摒弃了传统的循环和卷积结构。",
                "score": 0.75,
                "chunk_id": "chk_001",
                "doc_id": "doc_003",
                "created_at": time.time() - 216000,  # 2.5 days ago
                "source_type": "news"
            },
            {
                "content": "注意力机制(Attention Mechanism)允许模型在处理序列时动态地关注不同位置的信息，这是Transformer的核心创新。",
                "score": 0.80,
                "chunk_id": "chk_002",
                "doc_id": "doc_003",
                "created_at": time.time() - 172800,  # 2 days ago
                "source_type": "academic"
            }
        ]
    }
    
    return sample_docs


async def demonstrate_efr_processing():
    """Demonstrate EFR layer processing"""
    
    print("🚀 Enhanced EFR Layer 演示")
    print("=" * 60)
    
    # Create enhanced EFR configuration
    efr_config = EnhancedEFRConfig(
        # Basic RRF Configuration
        rrf_k=60,
        rrf_weights={
            "vector": 1.0,
            "entity": 0.8,
            "relation": 0.7,
            "chunk": 0.9,
        },
        
        # MMMORRF-inspired multimodal weights
        modality_weights={
            "text": 1.0,
            "image": 0.8,
            "table": 0.9,
            "equation": 0.8,
        },
        
        # Multilingual support
        enable_multilingual=True,
        language_weights={
            "zh": 1.0,
            "en": 1.0,
        },
        
        # Hybrid reranking configuration
        enable_hybrid_reranking=True,
        lightweight_rerank_threshold=20,
        llm_rerank_threshold=10,
        
        # Robustness evaluation
        enable_robustness_check=True,
        robustness_threshold=0.5,
        enable_conflict_resolution=True,
        
        # Source reliability estimation
        enable_source_reliability=True,
        reliability_weights={
            "consistency": 0.4,
            "accuracy": 0.3,
            "authority": 0.2,
            "freshness": 0.1,
        },
        
        # Advanced MMR configuration
        enable_adaptive_mmr=True,
        mmr_lambda=0.7,
        mmr_adaptation_factor=0.1,
        mmr_top_k=5,
        
        # Recency & Source Trust Configuration
        enable_recency=True,
        recency_weight=0.2,
        recency_decay_factor=0.1,
        
        enable_source_trust=True,
        source_trust_weights={
            "academic": 1.0,
            "official": 0.9,
            "news": 0.7,
            "blog": 0.5,
            "forum": 0.3,
            "unknown": 0.6,
        },
    )
    
    # Create mock reranker
    mock_reranker = MockReranker()
    
    # Initialize Enhanced EFR Layer
    efr_layer = EnhancedEFRLayer(config=efr_config, rerank_func=mock_reranker)
    
    print("✅ Enhanced EFR Layer 初始化完成")
    print(f"配置信息:")
    print(f"  - 多模态支持: {efr_config.enable_multilingual}")
    print(f"  - 混合重排序: {efr_config.enable_hybrid_reranking}")
    print(f"  - 鲁棒性检查: {efr_config.enable_robustness_check}")
    print(f"  - 来源可靠性: {efr_config.enable_source_reliability}")
    print(f"  - 自适应MMR: {efr_config.enable_adaptive_mmr}")
    
    # Create sample retrieval data
    retrieval_lists = await create_sample_retrieval_data()
    
    print(f"\n📊 原始检索数据:")
    for source_type, items in retrieval_lists.items():
        print(f"  {source_type}: {len(items)} 个结果")
        for i, item in enumerate(items[:2]):  # Show first 2 items
            print(f"    {i+1}. 分数: {item['score']:.3f}, 内容: {item['content'][:50]}...")
    
    # Test queries
    test_queries = [
        "什么是深度学习？",
        "比较CNN和RNN的区别",
        "Explain the relationship between deep learning and neural networks"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"🔍 测试查询 {i}: {query}")
        print(f"{'='*60}")
        
        # Process with Enhanced EFR
        start_time = time.time()
        efr_results = await efr_layer.process_retrieval_results(
            query=query,
            retrieval_lists=retrieval_lists
        )
        processing_time = time.time() - start_time
        
        print(f"⏱️  处理时间: {processing_time:.3f}秒")
        print(f"📈 EFR处理结果: {len(efr_results)} 个最终结果")
        
        # Show detailed results
        print(f"\n🏆 前3个Enhanced EFR结果:")
        for j, result in enumerate(efr_results[:3], 1):
            print(f"\n  {j}. 来源: {result.source}")
            print(f"     最终分数: {result.final_score:.3f}")
            print(f"     RRF分数: {result.rrf_score:.3f}")
            print(f"     重排序分数: {result.rerank_score:.3f}")
            print(f"     新鲜度分数: {result.recency_score:.3f}")
            print(f"     来源可信度: {result.source_trust_score:.3f}")
            
            # Show metadata
            metadata = result.metadata
            print(f"     模态: {metadata.get('modality', 'text')}")
            print(f"     语言: {metadata.get('language', 'zh')}")
            print(f"     可靠性: {metadata.get('reliability_score', 0.5):.3f}")
            
            print(f"     内容: {result.content[:100]}...")
        
        # Show processing statistics
        if hasattr(efr_layer, 'get_efr_statistics'):
            stats = efr_layer.get_efr_statistics()
            print(f"\n📊 处理统计:")
            print(f"  - 配置: {stats.get('config', {}).get('enable_robustness_check', False)}")
            print(f"  - 处理统计: {stats.get('processing_stats', {})}")


async def demonstrate_comparison():
    """Demonstrate comparison between different approaches"""
    
    print(f"\n{'='*60}")
    print("📊 EFR层效果对比演示")
    print(f"{'='*60}")
    
    # Create sample data
    retrieval_lists = await create_sample_retrieval_data()
    query = "什么是深度学习？"
    
    # 1. Simple RRF (without EFR)
    print(f"\n1️⃣ 简单RRF处理 (无EFR层):")
    simple_results = []
    for source_type, items in retrieval_lists.items():
        for item in items:
            simple_results.append({
                'content': item['content'],
                'score': item['score'],
                'source': source_type
            })
    
    # Sort by score
    simple_results.sort(key=lambda x: x['score'], reverse=True)
    print(f"   结果数量: {len(simple_results)}")
    print(f"   前3个结果:")
    for i, result in enumerate(simple_results[:3], 1):
        print(f"     {i}. 来源: {result['source']}, 分数: {result['score']:.3f}")
        print(f"        内容: {result['content'][:60]}...")
    
    # 2. Enhanced EFR
    print(f"\n2️⃣ Enhanced EFR处理:")
    efr_config = EnhancedEFRConfig(
        enable_robustness_check=True,
        enable_source_reliability=True,
        enable_adaptive_mmr=True,
        mmr_top_k=5
    )
    
    mock_reranker = MockReranker()
    efr_layer = EnhancedEFRLayer(config=efr_config, rerank_func=mock_reranker)
    
    efr_results = await efr_layer.process_retrieval_results(
        query=query,
        retrieval_lists=retrieval_lists
    )
    
    print(f"   结果数量: {len(efr_results)}")
    print(f"   前3个结果:")
    for i, result in enumerate(efr_results[:3], 1):
        print(f"     {i}. 来源: {result.source}, 最终分数: {result.final_score:.3f}")
        print(f"        RRF: {result.rrf_score:.3f}, 重排序: {result.rerank_score:.3f}")
        print(f"        新鲜度: {result.recency_score:.3f}, 可信度: {result.source_trust_score:.3f}")
        print(f"        内容: {result.content[:60]}...")
    
    # 3. Comparison summary
    print(f"\n📈 对比总结:")
    print(f"   - 简单RRF: 基于原始分数排序")
    print(f"   - Enhanced EFR: 多维度综合评分")
    print(f"   - 改进点:")
    print(f"     • 多源融合: 加权RRF融合不同来源")
    print(f"     • 重排序优化: 基于查询-文档相关性重排")
    print(f"     • 新鲜度考虑: 时间衰减因子")
    print(f"     • 来源可信度: 基于来源类型的权重")
    print(f"     • 多样性优化: MMR去冗余")
    print(f"     • 鲁棒性检查: 查询复杂度评估")


async def main():
    """Main demonstration function"""
    
    print("🌟 Enhanced EFR Layer 核心功能演示")
    print("基于最新研究论文的增强技术:")
    print("• MMMORRF: 多模态多语言模块化互惠排名融合")
    print("• LLM重排器分析: 混合重排序策略")
    print("• RARE: 检索感知鲁棒性评估")
    print("• RA-RAG: 来源可靠性估计")
    print("=" * 60)
    
    # Demonstrate EFR processing
    await demonstrate_efr_processing()
    
    # Demonstrate comparison
    await demonstrate_comparison()
    
    print(f"\n{'='*60}")
    print("🎉 Enhanced EFR Layer 演示完成！")
    print("\n🚀 主要增强特性:")
    print("✅ 多模态多语言支持 - 模态感知权重和语言权重")
    print("✅ 混合重排序策略 - 结合轻量级和LLM重排序")
    print("✅ 鲁棒性评估 - 查询复杂度和冲突检测")
    print("✅ 来源可靠性估计 - 多源交叉验证和加权融合")
    print("✅ 自适应MMR - 基于查询复杂度的动态多样性优化")
    print("✅ 增强评分系统 - 多维度综合评分")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

