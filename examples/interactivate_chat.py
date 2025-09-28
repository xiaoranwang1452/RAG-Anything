#!/usr/bin/env python
"""
Interactive RAGAnything Chatbot with OpenAI API

A completely new interactive example that allows you to:
1. Process documents with RAGAnything using MinerU parser
2. Chat interactively with your documents
3. Ask questions in natural language
4. Get intelligent answers based on document content

Usage:
    python interactive_chat.py --input your_document.pdf
    python interactive_chat.py --input docs_folder/ --working_dir ./chat_storage
"""

import os
import argparse
import asyncio
import json
import logging
import logging.config
from pathlib import Path
import numpy as np
from datetime import datetime

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def _env_flag(name: str, default: str = "false") -> bool:
    """Read boolean feature flags from environment."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, separator: str = ",") -> list[str]:
    value = os.getenv(name, "")
    return [item.strip().lower() for item in value.split(separator) if item.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _http_post_json(url: str, payload: dict, timeout: int = 240) -> dict:
    """Minimal helper to POST JSON without external deps."""
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} error from {url}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e}")


def _build_ollama_embedding_func() -> EmbeddingFunc:
    """Create an embedding function that talks to Ollama via HTTP."""

    host = os.getenv("EMBEDDING_BINDING_HOST", os.getenv("LLM_BINDING_HOST", "http://localhost:11434")).rstrip("/")
    model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest").strip()
    url = f"{host}/api/embeddings"
    timeout = _env_int("TIMEOUT", 240)
    max_concurrency = max(1, _env_int("EMBEDDING_MAX_CONCURRENCY", 4))
    embed_semaphore = asyncio.Semaphore(max_concurrency)

    async def embed_batch(texts):
        async def one(text):
            payload = {"model": model, "prompt": text}
            async with embed_semaphore:
                resp = await asyncio.to_thread(_http_post_json, url, payload, timeout)
            embedding = resp.get("embedding")
            if not embedding:
                raise RuntimeError("No embedding returned from Ollama")
            return embedding

        if not texts:
            return []
        return list(await asyncio.gather(*(one(t) for t in texts)))

    dim = os.getenv("EMBEDDING_DIM")
    embedding_dim = int(dim) if dim and dim.isdigit() else None

    return EmbeddingFunc(
        embedding_dim=embedding_dim or 1024,
        max_token_size=_env_int("MAX_EMBED_TOKENS", 8192),
        func=embed_batch,
    )


# OpenAI API Configuration
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 3072)
DEFAULT_COMPLETION_MAX_TOKENS = _env_int("LLM_COMPLETION_MAX_TOKENS", 512)


def configure_logging():
    """Configure logging for the interactive chat application"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "interactive_chat.log"))

    print(f"\nInteractive Chat log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


def print_welcome_banner():
    """Print a welcome banner for the interactive chat"""
    print("\n" + "="*80)
    print("🤖 RAGAnything Interactive Chatbot")
    print("="*80)
    print("Welcome! I'm your AI assistant powered by RAGAnything.")
    print("I can help you understand and explore your documents.")
    print("\nCommands:")
    print("  • Type your questions naturally")
    print("  • Type 'help' for more information")
    print("  • Type 'quit', 'exit', or 'bye' to end the conversation")
    print("="*80)


def print_help():
    """Print help information"""
    print("\n" + "-"*60)
    print("📚 HELP - How to use this chatbot:")
    print("-"*60)
    print("• Ask questions about your documents in natural language")
    print("• Examples of good questions:")
    print("  - 'What is this document about?'")
    print("  - 'Summarize the main points'")
    print("  - 'What are the key findings?'")
    print("  - 'Are there any tables or figures?'")
    print("  - 'Explain the methodology used'")
    print("• I can understand both English and Chinese questions")
    print("• Type 'quit', 'exit', or 'bye' to end the conversation")
    print("-"*60)


async def setup_rag_system(
    input_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    Set up the RAG system and process documents
    
    Returns:
        RAGAnything instance ready for querying
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or os.getenv("WORKING_DIR", "./chat_storage"),
            parser=parser,
            parse_method=os.getenv("PARSE_METHOD", "auto"),
            enable_image_processing=_env_flag("ENABLE_IMAGE_PROCESSING", "true"),
            enable_table_processing=_env_flag("ENABLE_TABLE_PROCESSING", "true"),
            enable_equation_processing=_env_flag("ENABLE_EQUATION_PROCESSING", "true"),
            enable_micro_planner=_env_flag("ENABLE_MICRO_PLANNER", "true"),
        )

        # Define LLM model function using OpenAI API
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            kwargs = dict(kwargs)
            kwargs.setdefault("max_tokens", DEFAULT_COMPLETION_MAX_TOKENS)
            kwargs.setdefault("temperature", _env_float("TEMPERATURE", 0.2))
            return await openai_complete_if_cache(
                DEFAULT_LLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Define vision model function for image processing using OpenAI API
        async def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return await openai_complete_if_cache(
                    DEFAULT_VISION_MODEL,
                    None,
                    system_prompt=None,
                    history_messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            elif image_data:
                structured_messages = [
                    (
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None
                    ),
                    (
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
                        }
                        if image_data
                        else {"role": "user", "content": prompt}
                    ),
                ]
                structured_messages = [m for m in structured_messages if m]
                return await openai_complete_if_cache(
                    DEFAULT_VISION_MODEL,
                    None,
                    system_prompt=None,
                    history_messages=structured_messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return await llm_model_func(
                    prompt, system_prompt, history_messages, **kwargs
                )

        # Define embedding function using OpenAI API
        embedding_binding = os.getenv("EMBEDDING_BINDING", "").strip().lower()
        if embedding_binding == "ollama":
            embedding_func = _build_ollama_embedding_func()
        else:
            embedding_func = EmbeddingFunc(
                embedding_dim=DEFAULT_EMBEDDING_DIM,
                max_token_size=_env_int("MAX_EMBED_TOKENS", 8192),
                func=lambda texts: openai_embed(
                    texts,
                    model=DEFAULT_EMBEDDING_MODEL,
                    api_key=api_key,
                    base_url=base_url,
                ),
            )

        # Define rerank model function using embedding similarity
        async def rerank_model_func(
            query: str, documents: list[str], top_n: int | None = None, **kwargs
        ) -> list[dict[str, float]]:
            texts = [query] + documents
            embeddings = await embedding_func(texts)
            query_vec = np.array(embeddings[0])
            doc_vecs = [np.array(e) for e in embeddings[1:]]
            scores = [float(np.dot(doc_vec, query_vec)) for doc_vec in doc_vecs]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            if top_n is not None:
                ranked = ranked[:top_n]
            return [
                {"index": i, "relevance_score": scores[i]} for i in ranked
            ]

        # Initialize RAGAnything
        rag = RAGAnything(
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func, 
            config=config,
            lightrag_kwargs={"rerank_model_func": rerank_model_func},
        )

        # Micro planner uses lexical fallback to avoid awaiting async evaluator
        if rag.micro_planner:
            rag.micro_planner.evaluator_func = None

        # Process input files
        input_root = Path(input_path)
        if not input_root.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        recursive = _env_flag("RECURSIVE_FOLDER_PROCESSING", "false")
        supported_exts = set(_env_list("SUPPORTED_FILE_EXTENSIONS"))

        if input_root.is_file():
            files = [input_root]
        elif recursive:
            files = [p for p in input_root.rglob("*") if p.is_file()]
        else:
            files = [p for p in input_root.iterdir() if p.is_file()]

        if supported_exts:
            files = [f for f in files if f.suffix.lower() in supported_exts]

        if not files:
            raise FileNotFoundError(
                f"No ingestible files found under {input_path}."
            )

        max_files = _env_int("MAX_CONCURRENT_FILES", 0)
        if max_files > 0:
            files = files[:max_files]

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_path)

        # Process documents
        print(f"\n📄 Processing {len(files)} document(s)...")
        for i, file in enumerate(files, 1):
            print(f"   [{i}/{len(files)}] Processing: {file.name}")
            logger.info(f"Processing document: {file}")
            await rag.process_document_complete(
                file_path=str(file),
                output_dir=output_dir,
                parse_method=config.parse_method,
                device="cpu",
            )

        print(f"\n✅ Document processing completed!")
        print("🤖 I'm now ready to answer your questions about the documents.")
        return rag

    except Exception as e:
        logger.error(f"Error setting up RAG system: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


async def start_chat_session(rag):
    """
    Start the interactive chat session
    
    Args:
        rag: RAGAnything instance ready for querying
    """
    print_welcome_banner()
    
    question_count = 0
    session_start = datetime.now()
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n💬 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                session_duration = datetime.now() - session_start
                print(f"\n👋 Goodbye! Thanks for chatting with me.")
                print(f"📊 Session summary: {question_count} questions answered in {session_duration}")
                break
            
            # Check for help command
            if user_input.lower() in ['help', '?', 'h']:
                print_help()
                continue
            
            # Skip empty inputs
            if not user_input:
                print("🤔 Please ask me a question about your documents.")
                continue
            
            question_count += 1
            print(f"\n🤖 Processing your question...")
            
            # Process query
            result = await rag.aquery(user_input, mode="hybrid")
            
            print(f"\n🤖 Assistant:")
            print(f"{result}")
            print(f"\n{'─' * 60}")
            
        except KeyboardInterrupt:
            print(f"\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Sorry, I encountered an error: {str(e)}")
            logger.error(f"Error processing query: {str(e)}")
            continue


async def main_async(
    input_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    Main async function to set up RAG and start chat
    """
    try:
        # Set up RAG system
        rag = await setup_rag_system(
            input_path, output_dir, api_key, base_url, working_dir, parser
        )
        
        # Start chat session
        await start_chat_session(rag)
        
    except Exception as e:
        print(f"\n❌ Failed to start the chat system: {str(e)}")
        logger.error(f"Error in main: {str(e)}")


def main():
    """Main function to run the interactive chat"""
    parser = argparse.ArgumentParser(
        description="Interactive RAGAnything Chatbot with OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python interactive_chat.py --input document.pdf
    python interactive_chat.py --input docs_folder/ --working_dir ./chat_storage
    python interactive_chat.py --input example_doc/ --parser docling
        """
    )
    
    parser.add_argument(
        "--input",
        "-i",
        default=os.getenv("INPUT_DIR", "example_doc"),
        help="Path to a document file or directory to process (default: example_doc)",
    )
    parser.add_argument(
        "--working_dir",
        "-w",
        default=os.getenv("WORKING_DIR", "./chat_storage"),
        help="Working directory path for RAG storage (default: ./chat_storage)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.getenv("OUTPUT_DIR", "./output"),
        help="Output directory path for results (default: ./output)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        choices=["mineru", "docling"],
        help="Parser to use: mineru or docling (default: mineru)",
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        print("❌ Error: OpenAI API key is required")
        print("Please set the LLM_BINDING_API_KEY environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Run the async main function
    asyncio.run(
        main_async(
            args.input,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("🚀 Starting RAGAnything Interactive Chatbot...")
    print("=" * 50)

    main()