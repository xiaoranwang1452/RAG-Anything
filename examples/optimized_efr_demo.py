#!/usr/bin/env python3
"""
Optimized EFR Layer Performance Demo

This script demonstrates the performance improvements of the Optimized EFR Layer
compared to the standard and enhanced versions.

Author: AI Assistant
Date: 2024
"""

import asyncio
import time
import os
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

# Test queries for performance comparison
TEST_QUERIES = [
    "What is the main contribution of this paper?",
    "What are the key technologies used in ChartCitor?",
    "How does the multi-agent framework work?",
    "What are the experimental results and performance metrics?",
    "What are the limitations and future work mentioned?"
]


async def create_llm_model_func():
    """Create LLM model function"""
    import openai
    
    async def llm_model_func(messages, **kwargs):
        try:
            response = await openai.AsyncOpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.1),
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {e}")
            return "Error: Unable to generate response"
    
    return llm_model_func


async def create_vision_model_func():
    """Create vision model function"""
    import openai
    
    async def vision_model_func(messages, **kwargs):
        try:
            response = await openai.AsyncOpenAI().chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.1),
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Vision API error: {e}")
            return "Error: Unable to process image"
    
    return vision_model_func


async def create_embedding_func():
    """Create embedding function"""
    import openai
    
    async def embedding_func(texts, **kwargs):
        try:
            response = await openai.AsyncOpenAI().embeddings.create(
                model="text-embedding-3-large",
                input=texts if isinstance(texts, list) else [texts],
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Embedding API error: {e}")
            return [[0.0] * 3072] * (len(texts) if isinstance(texts, list) else 1)
    
    # Add embedding_dim attribute
    embedding_func.embedding_dim = 3072
    return embedding_func


async def test_efr_performance(rag, query, efr_type="Standard"):
    """Test EFR performance for a single query"""
    print(f"\n{'='*60}")
    print(f"Testing {efr_type} EFR Layer")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if efr_type == "Standard":
            # Disable EFR for standard test
            response = await rag.aquery(query, mode="mix")
        else:
            # Use EFR for enhanced/optimized test
            response = await rag.aquery_enhanced(query, mode="mix", use_efr=True)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"Response Time: {response_time:.3f}s")
        print(f"Response Length: {len(response)} characters")
        print(f"Response Preview: {response[:200]}...")
        
        return {
            "response_time": response_time,
            "response_length": len(response),
            "response": response,
            "success": True
        }
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"Error: {e}")
        print(f"Response Time: {response_time:.3f}s")
        
        return {
            "response_time": response_time,
            "response_length": 0,
            "response": "",
            "success": False,
            "error": str(e)
        }


async def run_performance_comparison():
    """Run comprehensive performance comparison"""
    print("🚀 Optimized EFR Layer Performance Demo")
    print("=" * 60)
    
    # Check if we have a document to process
    doc_path = "examples/example_doc/2005.11401v4.pdf"
    if not os.path.exists(doc_path):
        print(f"❌ Document not found: {doc_path}")
        print("Please place a PDF document in the example_doc/ directory")
        return
    
    # Create model functions
    llm_model_func = await create_llm_model_func()
    vision_model_func = await create_vision_model_func()
    embedding_func = await create_embedding_func()
    
    # Test configurations
    configs = [
        {
            "name": "Standard RAG",
            "efr_config": None,
            "use_efr": False,
            "use_enhanced_efr": False,
            "use_optimized_efr": False
        },
        {
            "name": "Enhanced EFR",
            "efr_config": EnhancedEFRConfig(),
            "use_efr": True,
            "use_enhanced_efr": True,
            "use_optimized_efr": False
        },
        {
            "name": "Optimized EFR",
            "efr_config": OptimizedEFRConfig(),
            "use_efr": True,
            "use_enhanced_efr": False,
            "use_optimized_efr": True
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🔧 Initializing {config['name']}...")
        
        # Create RAGAnything instance
        rag_config = RAGAnythingConfig(
            working_dir=f"optimized_efr_output_{config['name'].lower().replace(' ', '_')}",
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
            enable_efr=config["use_efr"],
            use_enhanced_efr=config["use_enhanced_efr"],
            use_optimized_efr=config["use_optimized_efr"],
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func
        )
        
        # Process document
        print(f"📄 Processing document: {doc_path}")
        process_result = await rag.process_document_complete(doc_path)
        
        if not process_result.get("success", False):
            print(f"❌ Failed to process document: {process_result.get('error', 'Unknown error')}")
            continue
        
        print(f"✅ Document processed successfully")
        
        # Test queries
        config_results = []
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n📝 Test Query {i}/{len(TEST_QUERIES)}")
            result = await test_efr_performance(rag, query, config["name"])
            config_results.append(result)
        
        results[config["name"]] = config_results
        
        # Cleanup
        try:
            await rag.finalize_storages()
        except:
            pass
    
    # Performance analysis
    print(f"\n{'='*80}")
    print("📊 PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    for config_name, config_results in results.items():
        print(f"\n🔍 {config_name} Results:")
        print("-" * 40)
        
        successful_results = [r for r in config_results if r["success"]]
        if not successful_results:
            print("❌ No successful queries")
            continue
        
        avg_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
        avg_length = sum(r["response_length"] for r in successful_results) / len(successful_results)
        success_rate = len(successful_results) / len(config_results) * 100
        
        print(f"Average Response Time: {avg_time:.3f}s")
        print(f"Average Response Length: {avg_length:.0f} characters")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Individual query results
        for i, result in enumerate(config_results, 1):
            status = "✅" if result["success"] else "❌"
            print(f"  Query {i}: {result['response_time']:.3f}s, {result['response_length']} chars {status}")
    
    # Comparison analysis
    if len(results) >= 2:
        print(f"\n📈 COMPARISON ANALYSIS")
        print("-" * 40)
        
        baseline_name = "Standard RAG"
        if baseline_name in results:
            baseline_results = [r for r in results[baseline_name] if r["success"]]
            if baseline_results:
                baseline_avg_time = sum(r["response_time"] for r in baseline_results) / len(baseline_results)
                baseline_avg_length = sum(r["response_length"] for r in baseline_results) / len(baseline_results)
                
                print(f"Baseline ({baseline_name}):")
                print(f"  Average Time: {baseline_avg_time:.3f}s")
                print(f"  Average Length: {baseline_avg_length:.0f} chars")
                
                for config_name, config_results in results.items():
                    if config_name == baseline_name:
                        continue
                    
                    successful_results = [r for r in config_results if r["success"]]
                    if successful_results:
                        avg_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
                        avg_length = sum(r["response_length"] for r in successful_results) / len(successful_results)
                        
                        time_improvement = ((baseline_avg_time - avg_time) / baseline_avg_time) * 100
                        length_change = ((avg_length - baseline_avg_length) / baseline_avg_length) * 100
                        
                        print(f"\n{config_name}:")
                        print(f"  Average Time: {avg_time:.3f}s ({time_improvement:+.1f}%)")
                        print(f"  Average Length: {avg_length:.0f} chars ({length_change:+.1f}%)")
                        
                        if time_improvement > 0:
                            print(f"  🚀 Speed Improvement: {time_improvement:.1f}%")
                        else:
                            print(f"  ⏱️  Speed Decrease: {abs(time_improvement):.1f}%")
    
    print(f"\n{'='*80}")
    print("✅ Performance comparison completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(run_performance_comparison())
