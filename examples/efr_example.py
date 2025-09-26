#!/usr/bin/env python3
"""
EFR Layer Example for RAGAnything

This example demonstrates how to use the Evidence Fusion & Re-Ranking (EFR) layer
with RAGAnything for enhanced retrieval capabilities.

Features demonstrated:
1. Basic EFR layer setup and configuration
2. Enhanced querying with EFR
3. EFR analysis and statistics
4. Comparison between standard and EFR-enhanced queries

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
from raganything.efr_layer import EFRConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


async def setup_rag_with_efr():
    """Setup RAGAnything with EFR layer"""
    
    print("🚀 Setting up RAGAnything with EFR Layer...")
    
    # Check API configuration
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST", "https://yinli.one/v1")
    
    if not api_key:
        print("❌ 未找到 API 密钥，请检查环境变量 LLM_BINDING_API_KEY")
        return None
    
    print("✅ API 配置正确")
    
    # Create EFR configuration
    efr_config = EFRConfig(
        # RRF Configuration
        rrf_k=60,
        rrf_weights={
            "vector": 1.0,      # Vector similarity weight
            "entity": 0.8,      # Entity-based retrieval weight  
            "relation": 0.7,    # Relationship-based retrieval weight
            "chunk": 0.9,       # Direct chunk retrieval weight
        },
        
        # Re-ranking Configuration
        enable_rerank=True,
        rerank_top_k=20,
        min_rerank_score=0.3,
        
        # MMR Configuration
        enable_mmr=True,
        mmr_lambda=0.7,  # Balance between relevance and diversity
        mmr_top_k=10,
        
        # Recency & Source Trust Configuration
        enable_recency=True,
        recency_weight=0.2,
        recency_decay_factor=0.1,
        
        enable_source_trust=True,
        source_trust_weights={
            "academic": 1.0,     # Academic papers
            "official": 0.9,     # Official documents
            "news": 0.7,         # News articles
            "blog": 0.5,         # Blog posts
            "forum": 0.3,        # Forum discussions
            "unknown": 0.6,      # Unknown sources
        },
        
        # Performance Configuration
        max_parallel_rerank=5,
        cache_rerank_results=True,
    )
    
    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./efr_example_storage",
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
    
    # Initialize RAGAnything with EFR
    rag = RAGAnything(
        config=config,
        efr_config=efr_config,
        enable_efr=True,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    
    print("✅ RAGAnything with EFR layer initialized")
    return rag


async def demo_efr_queries(rag):
    """Demonstrate EFR-enhanced queries"""
    
    print("\n🔍 Demonstrating EFR-Enhanced Queries...")
    
    # Example queries
    queries = [
        "什么是机器学习的主要应用领域？",
        "深度学习与传统机器学习有什么区别？",
        "人工智能在医疗领域有哪些应用？",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- 查询 {i}: {query} ---")
        
        # Standard query (without EFR)
        print("📝 标准查询结果:")
        try:
            standard_result = await rag.aquery(query, mode="hybrid", use_efr=False)
            print(f"结果: {standard_result[:200]}...")
        except Exception as e:
            print(f"标准查询失败: {e}")
        
        # Enhanced query with EFR
        print("\n🚀 EFR增强查询结果:")
        try:
            enhanced_result = await rag.aquery_enhanced(query, mode="hybrid", use_efr=True)
            print(f"结果: {enhanced_result[:200]}...")
        except Exception as e:
            print(f"EFR增强查询失败: {e}")
        
        # EFR analysis
        print("\n📊 EFR分析:")
        try:
            analysis = await rag.aquery_with_efr_analysis(query, mode="hybrid")
            print(f"检索结果数量: {analysis['retrieval_summary']['total_results']}")
            print(f"平均最终分数: {analysis['retrieval_summary']['avg_final_score']:.3f}")
            print(f"按来源分布: {analysis['retrieval_summary']['by_source']}")
            
            # Show top EFR results
            print("\n🏆 前3个EFR结果:")
            for j, result in enumerate(analysis['efr_results'][:3], 1):
                print(f"  {j}. 来源: {result['source']}, 分数: {result['final_score']:.3f}")
                print(f"     内容: {result['content'][:100]}...")
                
        except Exception as e:
            print(f"EFR分析失败: {e}")


async def demo_efr_configuration(rag):
    """Demonstrate EFR configuration and statistics"""
    
    print("\n⚙️ EFR配置和统计信息...")
    
    if rag.efr_layer:
        stats = rag.efr_layer.get_efr_statistics()
        print("EFR层统计信息:")
        print(f"  RRF常数: {stats['config']['rrf_k']}")
        print(f"  RRF权重: {stats['config']['rrf_weights']}")
        print(f"  重排序启用: {stats['config']['enable_rerank']}")
        print(f"  MMR启用: {stats['config']['enable_mmr']}")
        print(f"  新鲜度启用: {stats['config']['enable_recency']}")
        print(f"  来源可信度启用: {stats['config']['enable_source_trust']}")
        print(f"  缓存大小: {stats['cache_size']}")
    else:
        print("EFR层未启用")


async def demo_multimodal_efr(rag):
    """Demonstrate EFR with multimodal content"""
    
    print("\n🎨 演示多模态EFR查询...")
    
    # Example multimodal query
    multimodal_content = [
        {
            "type": "table",
            "table_data": """算法,准确率,速度
                        随机森林,85.2%,快速
                        神经网络,92.1%,中等
                        支持向量机,88.7%,快速""",
            "table_caption": "机器学习算法性能对比"
        }
    ]
    
    query = "根据这个表格，哪种算法最适合实时应用？"
    
    print(f"查询: {query}")
    print("多模态内容: 性能对比表格")
    
    try:
        # Enhanced multimodal query with EFR
        result = await rag.aquery_with_multimodal(
            query=query,
            multimodal_content=multimodal_content,
            mode="hybrid"
        )
        print(f"\n结果: {result[:300]}...")
        
    except Exception as e:
        print(f"多模态EFR查询失败: {e}")


async def main():
    """Main demonstration function"""
    
    print("🌟 RAGAnything EFR Layer 演示")
    print("=" * 50)
    
    # Setup RAG with EFR
    rag = await setup_rag_with_efr()
    if not rag:
        print("❌ 设置失败，退出演示")
        return
    
    # Process a sample document if available
    sample_doc = project_root / "example_doc" / "2005.11401v4.pdf"
    if sample_doc.exists():
        print(f"\n📄 处理示例文档: {sample_doc.name}")
        try:
            await rag.process_document_complete(
                file_path=str(sample_doc),
                output_dir="./efr_example_output",
                parse_method="auto"
            )
            print("✅ 文档处理完成")
        except Exception as e:
            print(f"⚠️ 文档处理失败: {e}")
    else:
        print("⚠️ 未找到示例文档，将使用现有知识库")
    
    # Demonstrate EFR features
    await demo_efr_configuration(rag)
    await demo_efr_queries(rag)
    await demo_multimodal_efr(rag)
    
    print("\n🎉 EFR Layer 演示完成！")
    print("\n主要特性:")
    print("✅ Weighted RRF - 多源证据融合")
    print("✅ Listwise Re-ranking - 按需列表式重排")
    print("✅ MMR Diversification - 多样性优化")
    print("✅ Recency & Source Trust - 新鲜度和可信度评分")
    print("✅ 多模态支持 - 图像、表格、公式处理")


if __name__ == "__main__":
    asyncio.run(main())

