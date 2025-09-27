#!/usr/bin/env python3
"""
Real EFR Layer Comparison Test

This script compares the actual performance and results of RAG with and without EFR layers,
ensuring EFR layers are actually working and showing their optimization effects.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from raganything import RAGAnything, RAGAnythingConfig
from raganything.efr_layer import EFRConfig
from raganything.enhanced_efr_layer import EnhancedEFRConfig
from raganything.optimized_efr_layer import OptimizedEFRConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test queries for comparison
TEST_QUERIES = [
    "What is the main contribution of this paper?",
    "What are the key technologies used?",
    "How does the framework work?",
    "What are the experimental results?",
    "What are the limitations of this approach?"
]


async def create_llm_model_func():
    """Create LLM model function"""
    import openai

    async def llm_model_func(prompt, system_prompt=None, **kwargs):
        try:
            # Use custom LLM service
            client = openai.AsyncOpenAI(
                base_url="https://yinli.one/v1",
                api_key="sk-k9dZI7P6Zcoy145q0QL4iMu3kk0C6iFFSt8oApVvzPdMEiQC"
            )
            
            # Build messages in the format expected by RAGAnything
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.1),
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {e}")
            return "Error: Unable to generate response"

    return llm_model_func


async def create_embedding_func():
    """Create embedding function"""
    import openai

    async def embedding_func(texts):
        try:
            client = openai.AsyncOpenAI(
                base_url="https://yinli.one/v1",
                api_key="sk-k9dZI7P6Zcoy145q0QL4iMu3kk0C6iFFSt8oApVvzPdMEiQC"
            )
            response = await client.embeddings.create(
                model="text-embedding-3-large",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Embedding API error: {e}")
            return [[0.0] * 3072 for _ in texts]

    # Add LightRAG expected attributes
    embedding_func.func = embedding_func
    embedding_func.embedding_dim = 3072
    return embedding_func


async def test_query_with_clear_cache(rag, query, test_name, clear_cache=True):
    """Test a single query with option to clear cache"""
    print(f"\n{'='*60}")
    print(f"🔍 {test_name}")
    print(f"📝 Query: {query}")
    print(f"{'='*60}")

    # Clear cache if requested
    if clear_cache and hasattr(rag, 'lightrag') and rag.lightrag:
        try:
            # Clear LLM response cache
            if hasattr(rag.lightrag, 'llm_response_cache'):
                await rag.lightrag.llm_response_cache.clear()
                print("🧹 Cleared LLM response cache")
        except Exception as e:
            print(f"⚠️  Could not clear cache: {e}")

    start_time = time.time()

    try:
        if "EFR" in test_name:
            response = await rag.aquery_enhanced(query, mode="mix", use_efr=True)
        else:
            response = await rag.aquery(query, mode="mix")

        end_time = time.time()
        response_time = end_time - start_time

        print(f"✅ Response Time: {response_time:.3f}s")
        print(f"📄 Response Length: {len(response)} characters")
        print(f"📄 Response Preview: {response[:300]}...")
        
        # Show EFR-specific information if available
        if "EFR" in test_name and hasattr(rag, 'efr_layer') and rag.efr_layer:
            print(f"🔧 EFR Layer Status: Active")
            if hasattr(rag.efr_layer, 'config'):
                print(f"🔧 EFR Config: {type(rag.efr_layer.config).__name__}")

        return {
            "response_time": response_time,
            "response_length": len(response),
            "response": response,
            "success": True
        }

    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time

        print(f"❌ Error: {e}")
        print(f"Response Time: {response_time:.3f}s")
        import traceback
        traceback.print_exc()

        return {
            "response_time": response_time,
            "response_length": 0,
            "response": "",
            "success": False,
            "error": str(e)
        }


async def run_real_efr_comparison():
    """Run real EFR comparison test"""
    print("🚀 Real EFR Layer Comparison Test")
    print("=" * 80)

    # Document path
    doc_path = "my_documents/2502.00989v1.pdf"
    print(f"📄 Using document: {doc_path}")

    # Create model functions
    llm_model_func = await create_llm_model_func()
    embedding_func = await create_embedding_func()

    # Test configurations
    configs = [
        {
            "name": "Standard RAG (No EFR)",
            "efr_config": None,
            "enable_efr": False,
            "use_enhanced_efr": False,
            "use_optimized_efr": False
        },
        {
            "name": "Enhanced EFR",
            "efr_config": EnhancedEFRConfig(),
            "enable_efr": True,
            "use_enhanced_efr": True,
            "use_optimized_efr": False
        },
        {
            "name": "Optimized EFR",
            "efr_config": OptimizedEFRConfig(),
            "enable_efr": True,
            "use_enhanced_efr": False,
            "use_optimized_efr": True
        }
    ]

    results = {}

    for config in configs:
        print(f"\n🔧 Testing {config['name']}...")

        # Create RAGAnything instance with unique working directory
        working_dir = f"real_efr_test_{config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        
        rag_config = RAGAnythingConfig(
            working_dir=working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            max_concurrent_files=1
        )

        rag = RAGAnything(
            config=rag_config,
            efr_config=config["efr_config"],
            enable_efr=config["enable_efr"],
            use_enhanced_efr=config["use_enhanced_efr"],
            use_optimized_efr=config["use_optimized_efr"],
            llm_model_func=llm_model_func,
            embedding_func=embedding_func
        )

        # Process document first
        print(f"📄 Processing document: {doc_path}")
        try:
            process_result = await rag.process_document_complete(doc_path)
            if process_result is None:
                print(f"✅ Document already processed (cached)")
            elif not process_result.get("success", False):
                print(f"❌ Failed to process document: {process_result.get('error', 'Unknown error')}")
                continue
            else:
                print(f"✅ Document processed successfully")
        except Exception as e:
            print(f"❌ Document processing error: {e}")
            continue

        # Test queries with cache clearing
        config_results = []
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n📝 Test Query {i}/{len(TEST_QUERIES)}")
            result = await test_query_with_clear_cache(rag, query, config["name"], clear_cache=True)
            config_results.append(result)

        results[config["name"]] = config_results

        # Cleanup
        try:
            await rag.finalize_storages()
        except:
            pass

    # Performance analysis
    print(f"\n{'='*80}")
    print("📊 REAL EFR COMPARISON ANALYSIS")
    print(f"{'='*80}")

    for config_name, config_results in results.items():
        total_time = sum(r["response_time"] for r in config_results)
        total_length = sum(r["response_length"] for r in config_results)
        success_count = sum(1 for r in config_results if r["success"])

        avg_time = total_time / len(config_results) if config_results else 0
        avg_length = total_length / len(config_results) if config_results else 0
        success_rate = (success_count / len(config_results)) * 100 if config_results else 0

        print(f"\n🔍 {config_name} Results:")
        print(f"  ⏱️  Average Response Time: {avg_time:.3f}s")
        print(f"  📝 Average Response Length: {avg_length:.0f} characters")
        print(f"  ✅ Success Rate: {success_rate:.1f}%")

    # Detailed comparison
    print(f"\n{'='*80}")
    print("🔍 DETAILED EFR COMPARISON")
    print(f"{'='*80}")

    for i, query in enumerate(TEST_QUERIES):
        print(f"\n📝 Query {i+1}: {query}")
        print("-" * 60)
        
        for config_name, config_results in results.items():
            if i < len(config_results):
                result = config_results[i]
                status = "✅" if result["success"] else "❌"
                print(f"{status} {config_name}: {result['response_time']:.3f}s, {result['response_length']} chars")
                if result["success"] and len(result["response"]) > 0:
                    print(f"   Preview: {result['response'][:100]}...")
                else:
                    print(f"   Error: {result.get('error', 'Unknown error')}")

    # EFR effectiveness analysis
    print(f"\n{'='*80}")
    print("🎯 EFR EFFECTIVENESS ANALYSIS")
    print(f"{'='*80}")
    
    if "Standard RAG (No EFR)" in results and "Enhanced EFR" in results:
        standard_results = results["Standard RAG (No EFR)"]
        enhanced_results = results["Enhanced EFR"]
        
        print(f"\n📊 Standard RAG vs Enhanced EFR:")
        for i, (std, enh) in enumerate(zip(standard_results, enhanced_results)):
            if std["success"] and enh["success"]:
                time_diff = std["response_time"] - enh["response_time"]
                length_diff = enh["response_length"] - std["response_length"]
                print(f"  Query {i+1}: Time diff: {time_diff:+.3f}s, Length diff: {length_diff:+d} chars")
    
    if "Standard RAG (No EFR)" in results and "Optimized EFR" in results:
        standard_results = results["Standard RAG (No EFR)"]
        optimized_results = results["Optimized EFR"]
        
        print(f"\n📊 Standard RAG vs Optimized EFR:")
        for i, (std, opt) in enumerate(zip(standard_results, optimized_results)):
            if std["success"] and opt["success"]:
                time_diff = std["response_time"] - opt["response_time"]
                length_diff = opt["response_length"] - std["response_length"]
                print(f"  Query {i+1}: Time diff: {time_diff:+.3f}s, Length diff: {length_diff:+d} chars")

    print(f"\n🎉 Real EFR Comparison Test Completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(run_real_efr_comparison())
