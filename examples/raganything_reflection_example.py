#!/usr/bin/env python3
"""
RAGAnything Reflection Example

This example demonstrates the Reflection Layer functionality for answer verification
and attribution. The reflection layer performs:
1. Sentence-level support analysis
2. Coverage analysis
3. Contradiction detection
4. Attributable rewriting with citations

Usage:
    python examples/raganything_reflection_example.py <document_path> --api-key <your_api_key>

Example:
    python examples/raganything_reflection_example.py data/sample.pdf --api-key sk-xxx
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from raganything.config import ReflectionConfig


def setup_openai_functions(api_key: str, base_url: str = None):
    """Setup OpenAI model functions"""

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

    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # Support multimodal messages format for VLM enhanced queries
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
        # Traditional single image format
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
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
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
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

    return llm_model_func, vision_model_func, embedding_func


async def demonstrate_reflection_layer():
    """Demonstrate the reflection layer functionality"""

    parser = argparse.ArgumentParser(description="RAGAnything Reflection Example")
    parser.add_argument("document_path", help="Path to document to process")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", help="OpenAI API base URL (optional)")
    parser.add_argument("--working-dir", default="./rag_storage_reflection",
                       help="Working directory for RAG storage")
    parser.add_argument("--output-dir", default="./output_reflection",
                       help="Output directory for parsed content")
    parser.add_argument("--question",
                       default="What are the main findings and conclusions presented in this document?",
                       help="Question to ask about the document")
    parser.add_argument("--enable-reflection", action="store_true", default=True,
                       help="Enable reflection layer (default: True)")
    parser.add_argument("--reflection-top-k", type=int, default=3,
                       help="Number of top results for reflection verification")

    args = parser.parse_args()

    # Validate document exists
    document_path = Path(args.document_path)
    if not document_path.exists():
        print(f" Document not found: {document_path}")
        sys.exit(1)

    print(" RAGAnything Reflection Layer Example")
    print("=" * 50)
    print(f" Document: {document_path}")
    print(f" Question: {args.question}")
    print(f" Reflection enabled: {args.enable_reflection}")
    print()

    try:
        # Setup model functions
        print(" Setting up model functions...")
        llm_model_func, vision_model_func, embedding_func = setup_openai_functions(
            args.api_key, args.base_url
        )

        # Create reflection configuration
        reflection_config = ReflectionConfig(
            enable_reflection=args.enable_reflection,
            reflection_top_k=args.reflection_top_k,
            reflection_query_mode="hybrid",
            reflection_temperature=0.1,
            reflection_max_sentences=15,
            reflection_support_threshold=0.7,
        )

        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=args.working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            reflection=reflection_config,
        )

        # Initialize RAGAnything
        print(" Initializing RAGAnything...")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process document if needed
        print(" Processing document...")
        await rag.process_document_complete(
            file_path=str(document_path),
            output_dir=args.output_dir,
            parse_method="auto"
        )

        print("\n" + "=" * 50)
        print(" GENERATING ANSWER WITH REFLECTION")
        print("=" * 50)

        # Generate answer with reflection
        result = await rag.answer_with_reflection(
            question=args.question,
            mode="hybrid"
        )

        # Display results
        print("\n ORIGINAL QUESTION:")
        print("-" * 30)
        print(result["question"])

        print("\n DRAFT ANSWER:")
        print("-" * 30)
        print(result["draft"])

        print("\n FINAL ANSWER (WITH CITATIONS):")
        print("-" * 30)
        print(result["final"])

        print("\n REFLECTION ANALYSIS:")
        print("-" * 30)
        reflection_report = result["reflection_report"]

        # Support analysis
        if "support" in reflection_report:
            support = reflection_report["support"]
            if "overall_support_score" in support:
                print(f" Overall Support Score: {support['overall_support_score']:.2f}")

            if "sentences" in support:
                print(f" Sentences Analyzed: {len(support['sentences'])}")
                for i, sentence_analysis in enumerate(support["sentences"][:3]):  # Show first 3
                    if "support_score" in sentence_analysis:
                        print(f"   Sentence {i+1}: {sentence_analysis['support_score']:.2f}")

        # Coverage analysis
        if "coverage" in reflection_report:
            coverage = reflection_report["coverage"]
            if "coverage_score" in coverage:
                print(f" Coverage Score: {coverage['coverage_score']:.2f}")

        # Contradiction analysis
        if "contradiction" in reflection_report:
            contradiction = reflection_report["contradiction"]
            if "has_contradictions" in contradiction:
                print(f"  Has Contradictions: {contradiction['has_contradictions']}")
            if "overall_consistency" in contradiction:
                print(f" Consistency Score: {contradiction['overall_consistency']:.2f}")

        # Attribution analysis
        if "attributable" in reflection_report:
            attribution = reflection_report["attributable"]
            if "citations_added" in attribution:
                print(f" Citations Added: {attribution['citations_added']}")
            if "rewrite_successful" in attribution:
                print(f"  Rewrite Successful: {attribution['rewrite_successful']}")

        # Targeted queries
        if "targeted_queries" in reflection_report:
            queries = reflection_report["targeted_queries"]
            print(f" Targeted Queries Used: {len(queries)}")
            for i, query in enumerate(queries[:3]):  # Show first 3
                print(f"   Query {i+1}: {query[:80]}...")

        print("\n" + "=" * 50)
        print(" REFLECTION EXAMPLE COMPLETED")
        print("=" * 50)

        # Save detailed results to file
        output_file = Path(args.working_dir) / "reflection_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f" Detailed results saved to: {output_file}")

        # Demonstration of different reflection configurations
        print("\n TRYING DIFFERENT REFLECTION SETTINGS...")

        # Test with different support threshold
        strict_reflection_config = ReflectionConfig(
            enable_reflection=True,
            reflection_top_k=5,
            reflection_support_threshold=0.9,  # Stricter threshold
            reflection_temperature=0.05,       # More deterministic
        )

        rag.config.reflection = strict_reflection_config

        strict_result = await rag.answer_with_reflection(
            question="What specific data or statistics are mentioned?",
            mode="hybrid"
        )

        print(f" Strict Reflection Final Answer: {strict_result['final'][:200]}...")

        if "attributable" in strict_result["reflection_report"]:
            strict_citations = strict_result["reflection_report"]["attributable"].get("citations_added", 0)
            print(f" Citations with strict settings: {strict_citations}")

    except Exception as e:
        print(f" Error during reflection example: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    asyncio.run(demonstrate_reflection_layer())


if __name__ == "__main__":
    main()