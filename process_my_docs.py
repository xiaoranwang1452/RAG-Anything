#!/usr/bin/env python3
"""
处理您的文档脚本
将您的文档放在 my_documents/ 文件夹中，然后运行此脚本
"""

import os
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def process_documents():
    """处理文档的主函数"""
    
    print("🚀 开始处理您的文档...")
    
    # 检查 API 配置
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST", "https://yinli.one/v1")
    
    if not api_key:
        print("❌ 未找到 API 密钥，请检查 .env 文件")
        return
    
    print("✅ API 配置正确")
    
    # 创建配置
    config = RAGAnythingConfig(
        working_dir="./my_rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # 定义模型函数
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-3.5-turbo",
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
    
    # 检查文档文件夹
    docs_folder = "./my_documents"
    if not os.path.exists(docs_folder):
        print(f"❌ 文档文件夹不存在: {docs_folder}")
        print("请创建文件夹并放入您的文档")
        return
    
    # 查找文档
    supported_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.txt', '.md']
    documents = []
    
    for file in os.listdir(docs_folder):
        if any(file.lower().endswith(ext) for ext in supported_extensions):
            documents.append(os.path.join(docs_folder, file))
    
    if not documents:
        print(f"❌ 在 {docs_folder} 文件夹中没有找到支持的文档")
        print("支持的格式:", ", ".join(supported_extensions))
        return
    
    print(f"📄 找到 {len(documents)} 个文档:")
    for doc in documents:
        print(f"  - {os.path.basename(doc)}")
    
    # 处理每个文档
    for i, doc_path in enumerate(documents, 1):
        print(f"\n🔄 处理文档 {i}/{len(documents)}: {os.path.basename(doc_path)}")
        
        try:
            await rag.process_document_complete(
                file_path=doc_path,
                output_dir="./my_output",
                parse_method="auto"
            )
            print(f"✅ 文档处理完成: {os.path.basename(doc_path)}")
            
        except Exception as e:
            print(f"❌ 处理文档失败: {os.path.basename(doc_path)}")
            print(f"错误: {e}")
            continue
    
    print("\n🎉 所有文档处理完成！")
    
    # 进行测试查询
    print("\n🔍 进行测试查询...")
    try:
        result = await rag.aquery(
            "这些文档的主要内容是什么？请总结一下。",
            mode="hybrid"
        )
        
        print("📝 查询结果:")
        print(result)
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")

async def interactive_query():
    """交互式查询模式"""
    print("\n🤖 进入交互式查询模式...")
    print("输入 'quit' 或 'exit' 退出")
    
    # 这里需要重新初始化 RAGAnything（简化版本）
    # 实际使用中，您可以将 rag 对象保存为全局变量
    
    while True:
        try:
            question = input("\n❓ 请输入您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            
            if not question:
                continue
            
            # 这里需要重新初始化 RAGAnything 进行查询
            # 为了简化，我们跳过交互式查询
            print("💡 提示：交互式查询功能需要重新初始化 RAGAnything")
            print("   您可以修改脚本添加此功能")
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break

async def main():
    """主函数"""
    print("📚 RAG-Anything 文档处理工具")
    print("=" * 50)
    
    # 处理文档
    await process_documents()
    
    # 询问是否进入交互式查询
    print("\n" + "=" * 50)
    choice = input("是否进入交互式查询模式？(y/n): ").strip().lower()
    if choice in ['y', 'yes', '是']:
        await interactive_query()

if __name__ == "__main__":
    asyncio.run(main())


