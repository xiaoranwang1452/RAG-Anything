#!/usr/bin/env python3
"""
RAG-Anything 简单使用示例
演示如何配置和使用 RAG-Anything 处理文档
"""

import asyncio
import os
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def main():
    """主函数 - 演示 RAG-Anything 的基本用法"""
    
    print("🚀 RAG-Anything 简单示例开始...")
    
    # 检查 API 密钥
    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST", "https://yinli.one/v1")
    
    if not api_key:
        print("⚠️  未找到 API 密钥")
        print("请设置您的 API 密钥：")
        print("export LLM_BINDING_API_KEY='your-api-key-here'")
        print("\n或者编辑 .env 文件添加：")
        print("LLM_BINDING_API_KEY=your-api-key-here")
        return
    
    print("✅ 找到 API 密钥")
    
    # 创建配置
    config = RAGAnythingConfig(
        working_dir="./example_rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # 定义 LLM 模型函数
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
    
    # 定义视觉模型函数
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
    
    # 定义嵌入函数
    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-ada-002",
            api_key=api_key,
            base_url=base_url,
        ),
    )
    
    # 初始化 RAGAnything
    print("🔧 初始化 RAGAnything...")
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    
    print("✅ RAGAnything 初始化完成")
    
    # 检查是否有示例文档
    example_docs = [
        "example_doc/2005.11401v4.pdf",
        "example_doc/CS55-1_Week6_TutorialPresentation.pdf",
        "example_doc/CS55-Agentic Multimodal RAG_An Intelligent Framework for Scientific Concept Discovery from Text and Visuals.pdf"
    ]
    
    available_docs = [doc for doc in example_docs if os.path.exists(doc)]
    
    if available_docs:
        print(f"\n📄 找到 {len(available_docs)} 个示例文档")
        
        # 处理第一个文档
        doc_path = available_docs[0]
        print(f"🔄 处理文档: {doc_path}")
        
        try:
            # 处理文档
            await rag.process_document_complete(
                file_path=doc_path,
                output_dir="./example_output",
                parse_method="auto"
            )
            
            print("✅ 文档处理完成")
            
            # 进行查询
            print("\n🔍 进行查询测试...")
            result = await rag.aquery(
                "这个文档的主要内容是什么？",
                mode="hybrid"
            )
            
            print("📝 查询结果:")
            print(result)
            
        except Exception as e:
            print(f"❌ 处理文档时出错: {e}")
            print("这可能是由于 API 限制或网络问题导致的")
    
    else:
        print("\n📄 未找到示例文档")
        print("您可以：")
        print("1. 将您的 PDF 文档放在项目目录中")
        print("2. 修改脚本中的 file_path 变量")
        print("3. 运行脚本处理您的文档")
    
    print("\n🎉 示例完成！")

if __name__ == "__main__":
    asyncio.run(main())
