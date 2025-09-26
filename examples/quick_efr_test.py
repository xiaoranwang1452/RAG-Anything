#!/usr/bin/env python3
"""
快速EFR效果对比测试

这个脚本提供了一个简单快速的方式来对比EFR前后的效果，
适合快速验证EFR层的改进效果。

使用方法:
python examples/quick_efr_test.py

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


async def quick_efr_test():
    """快速EFR效果测试"""
    
    print("🚀 快速EFR效果对比测试")
    print("=" * 50)
    
    # 检查API配置
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST", "https://yinli.one/v1")
    
    if not api_key:
        print("❌ 未找到 API 密钥，请设置环境变量 LLM_BINDING_API_KEY")
        return
    
    print("✅ API 配置正确")
    
    # 创建EFR配置
    efr_config = EFRConfig(
        enable_rerank=True,
        enable_mmr=True,
        enable_recency=True,
        enable_source_trust=True,
    )
    
    # 创建RAGAnything配置
    config = RAGAnythingConfig(
        working_dir="./quick_efr_test_storage",
        parser="mineru",
        parse_method="auto",
    )
    
    # 定义模型函数
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
    
    # 初始化RAGAnything
    rag = RAGAnything(
        config=config,
        efr_config=efr_config,
        enable_efr=True,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )
    
    print("✅ RAGAnything 初始化完成")
    
    # 处理示例文档
    sample_doc = project_root / "example_doc" / "2005.11401v4.pdf"
    if sample_doc.exists():
        print(f"\n📄 处理示例文档: {sample_doc.name}")
        try:
            await rag.process_document_complete(
                file_path=str(sample_doc),
                output_dir="./quick_efr_test_output",
                parse_method="auto"
            )
            print("✅ 文档处理完成")
        except Exception as e:
            print(f"⚠️ 文档处理失败: {e}")
    else:
        print("⚠️ 未找到示例文档，将使用现有知识库")
    
    # 测试查询
    test_query = "什么是机器学习的主要应用领域？"
    
    print(f"\n🔍 测试查询: {test_query}")
    print("=" * 60)
    
    # 1. 标准查询（不使用EFR）
    print("\n📝 标准查询结果:")
    print("-" * 40)
    try:
        standard_result = await rag.aquery(test_query, mode="hybrid", use_efr=False)
        print(f"响应: {standard_result}")
        print(f"响应长度: {len(standard_result)} 字符")
    except Exception as e:
        print(f"❌ 标准查询失败: {e}")
        standard_result = ""
    
    # 2. EFR增强查询
    print("\n🚀 EFR增强查询结果:")
    print("-" * 40)
    try:
        enhanced_result = await rag.aquery_enhanced(test_query, mode="hybrid", use_efr=True)
        print(f"响应: {enhanced_result}")
        print(f"响应长度: {len(enhanced_result)} 字符")
    except Exception as e:
        print(f"❌ EFR增强查询失败: {e}")
        enhanced_result = ""
    
    # 3. EFR详细分析
    print("\n📊 EFR详细分析:")
    print("-" * 40)
    try:
        analysis = await rag.aquery_with_efr_analysis(test_query, mode="hybrid")
        
        print(f"检索结果数量: {analysis['retrieval_summary']['total_results']}")
        print(f"平均最终分数: {analysis['retrieval_summary']['avg_final_score']:.3f}")
        print(f"按来源分布: {analysis['retrieval_summary']['by_source']}")
        
        print(f"\n🏆 前3个EFR结果:")
        for i, result in enumerate(analysis['efr_results'][:3], 1):
            print(f"  {i}. 来源: {result['source']}, 分数: {result['final_score']:.3f}")
            print(f"     内容: {result['content'][:100]}...")
            
    except Exception as e:
        print(f"❌ EFR分析失败: {e}")
    
    # 4. 简单对比
    print(f"\n📈 简单对比:")
    print("-" * 40)
    
    if standard_result and enhanced_result:
        length_diff = len(enhanced_result) - len(standard_result)
        length_improvement = (length_diff / len(standard_result) * 100) if len(standard_result) > 0 else 0
        
        print(f"响应长度变化: {length_diff:+d} 字符 ({length_improvement:+.1f}%)")
        
        if len(enhanced_result) > len(standard_result):
            print("✅ EFR增强了响应质量")
        elif len(enhanced_result) < len(standard_result):
            print("⚠️ EFR可能过度优化了响应")
        else:
            print("➖ 响应长度无明显变化")
    
    # 5. 交互式测试
    print(f"\n🎯 交互式测试:")
    print("-" * 40)
    print("你可以输入自己的查询来测试EFR效果")
    print("输入 'quit' 或 'exit' 退出")
    
    while True:
        try:
            user_query = input("\n请输入查询 (或 'quit' 退出): ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_query:
                continue
            
            print(f"\n🔍 查询: {user_query}")
            
            # 标准查询
            print("\n📝 标准查询:")
            try:
                standard = await rag.aquery(user_query, mode="hybrid", use_efr=False)
                print(f"{standard[:200]}...")
            except Exception as e:
                print(f"❌ 失败: {e}")
            
            # EFR查询
            print("\n🚀 EFR增强查询:")
            try:
                enhanced = await rag.aquery_enhanced(user_query, mode="hybrid", use_efr=True)
                print(f"{enhanced[:200]}...")
            except Exception as e:
                print(f"❌ 失败: {e}")
            
            # 快速分析
            print("\n📊 EFR分析:")
            try:
                analysis = await rag.aquery_with_efr_analysis(user_query, mode="hybrid")
                print(f"检索数量: {analysis['retrieval_summary']['total_results']}")
                print(f"平均分数: {analysis['retrieval_summary']['avg_final_score']:.3f}")
            except Exception as e:
                print(f"❌ 分析失败: {e}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 处理查询时出错: {e}")
    
    print("\n🎉 快速EFR测试完成！")
    print("\n💡 提示:")
    print("- 查看响应长度和质量的变化")
    print("- 注意EFR分析中的检索数量和分数")
    print("- 尝试不同类型的查询来测试效果")


if __name__ == "__main__":
    asyncio.run(quick_efr_test())

