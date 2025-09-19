#!/usr/bin/env python
"""
Example script demonstrating MinerU parser + RAGAnything
using Google Gemini (via google-generativeai SDK).
"""

import os
import argparse
import asyncio
import logging
from pathlib import Path
import sys

# Project root
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from lightrag.utils import logger, set_verbose_debug, EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

import google.generativeai as genai

# 加载 .env
load_dotenv(dotenv_path=".env", override=False)

# 配置 Google Gemini
genai.configure(api_key=os.getenv("LLM_BINDING_API_KEY"))


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


# === Gemini LLM wrapper ===
def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name)

    # 把历史消息拼起来
    history_text = ""
    if history_messages:
        for m in history_messages:
            history_text += f"{m['role']}: {m['content']}\n"

    final_prompt = ""
    if system_prompt:
        final_prompt += f"System: {system_prompt}\n"
    final_prompt += history_text
    final_prompt += f"User: {prompt}\nAssistant:"

    response = model.generate_content(final_prompt)
    return response.text if response else ""


# === Gemini Embedding wrapper ===
def gemini_embed(texts, model="text-embedding-004"):
    emb_model = genai.embed_content(model=model, content=texts)
    if isinstance(texts, str):
        return [emb_model["embedding"]]
    else:
        return [item["embedding"] for item in emb_model["embedding"]]


embedding_func = EmbeddingFunc(
    embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
    max_token_size=8192,
    func=lambda texts: gemini_embed(texts, model=os.getenv("EMBEDDING_MODEL", "text-embedding-004")),
)


async def process_with_rag(file_path: str, output_dir: str, working_dir: str = None, parser: str = None):
    try:
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser or "mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
        )

        await rag.process_document_complete(file_path=file_path, output_dir=output_dir, parse_method="auto")
        logger.info("\n✅ Document processed. Running queries...")

        # 示例查询
        q1 = "What is the main content of the document?"
        result1 = await rag.aquery(q1, mode="hybrid")
        print(f"\nQ: {q1}\nA: {result1}\n")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Gemini RAG Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument("--working_dir", "-w", default="./rag_storage", help="Working directory path")
    parser.add_argument("--output", "-o", default="./output", help="Output directory path")
    parser.add_argument("--parser", default=os.getenv("PARSER", "mineru"), help="Parser (default: mineru)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    asyncio.run(process_with_rag(
        file_path=args.file_path,
        output_dir=args.output,
        working_dir=args.working_dir,
        parser=args.parser,
    ))


if __name__ == "__main__":
    configure_logging()
    print("🚀 RAGAnything Example (Gemini, google-generativeai)")
    main()
