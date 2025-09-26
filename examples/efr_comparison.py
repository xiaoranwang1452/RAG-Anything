#!/usr/bin/env python3
"""
EFR Layer 效果对比测试脚本

这个脚本提供了多种方式来对比EFR层启用前后的效果：
1. 并排对比标准查询和EFR增强查询
2. 详细的检索结果分析对比
3. 性能指标对比
4. 可视化对比报告

Author: AI Assistant
Date: 2024
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from raganything import RAGAnything, RAGAnythingConfig
from raganything.efr_layer import EFRConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


@dataclass
class QueryResult:
    """查询结果数据结构"""
    query: str
    response: str
    retrieval_count: int
    avg_score: float
    processing_time: float
    source_distribution: Dict[str, int]
    top_results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None


@dataclass
class ComparisonReport:
    """对比报告数据结构"""
    query: str
    standard_result: QueryResult
    efr_result: QueryResult
    improvement_metrics: Dict[str, float]
    detailed_analysis: Dict[str, Any]


class EFRComparison:
    """EFR效果对比测试类"""
    
    def __init__(self, rag: RAGAnything):
        self.rag = rag
        self.comparison_results = []
    
    async def compare_single_query(
        self, 
        query: str, 
        mode: str = "hybrid",
        save_detailed: bool = True
    ) -> ComparisonReport:
        """
        对比单个查询的EFR前后效果
        
        Args:
            query: 查询文本
            mode: 查询模式
            save_detailed: 是否保存详细分析
            
        Returns:
            ComparisonReport: 对比报告
        """
        print(f"\n🔍 对比查询: {query}")
        print("=" * 80)
        
        # 1. 标准查询（不使用EFR）
        print("📝 执行标准查询...")
        start_time = time.time()
        
        try:
            standard_response = await self.rag.aquery(query, mode=mode, use_efr=False)
            standard_time = time.time() - start_time
            
            # 获取标准查询的检索信息（通过上下文分析）
            standard_analysis = await self._analyze_standard_query(query, mode)
            
            standard_result = QueryResult(
                query=query,
                response=standard_response,
                retrieval_count=standard_analysis.get('retrieval_count', 0),
                avg_score=standard_analysis.get('avg_score', 0.0),
                processing_time=standard_time,
                source_distribution=standard_analysis.get('source_distribution', {}),
                top_results=standard_analysis.get('top_results', []),
                metadata=standard_analysis
            )
            
        except Exception as e:
            print(f"❌ 标准查询失败: {e}")
            standard_result = QueryResult(
                query=query,
                response=f"查询失败: {e}",
                retrieval_count=0,
                avg_score=0.0,
                processing_time=0.0,
                source_distribution={},
                top_results=[],
                metadata={"error": str(e)}
            )
        
        # 2. EFR增强查询
        print("🚀 执行EFR增强查询...")
        start_time = time.time()
        
        try:
            efr_analysis = await self.rag.aquery_with_efr_analysis(query, mode=mode)
            efr_time = time.time() - start_time
            
            efr_result = QueryResult(
                query=query,
                response=efr_analysis['response'],
                retrieval_count=efr_analysis['retrieval_summary']['total_results'],
                avg_score=efr_analysis['retrieval_summary']['avg_final_score'],
                processing_time=efr_time,
                source_distribution=efr_analysis['retrieval_summary']['by_source'],
                top_results=efr_analysis['efr_results'][:5],  # 前5个结果
                metadata=efr_analysis
            )
            
        except Exception as e:
            print(f"❌ EFR查询失败: {e}")
            efr_result = QueryResult(
                query=query,
                response=f"EFR查询失败: {e}",
                retrieval_count=0,
                avg_score=0.0,
                processing_time=0.0,
                source_distribution={},
                top_results=[],
                metadata={"error": str(e)}
            )
        
        # 3. 计算改进指标
        improvement_metrics = self._calculate_improvement_metrics(standard_result, efr_result)
        
        # 4. 生成详细分析
        detailed_analysis = self._generate_detailed_analysis(standard_result, efr_result)
        
        # 5. 创建对比报告
        report = ComparisonReport(
            query=query,
            standard_result=standard_result,
            efr_result=efr_result,
            improvement_metrics=improvement_metrics,
            detailed_analysis=detailed_analysis
        )
        
        # 6. 显示对比结果
        self._display_comparison(report)
        
        # 7. 保存详细结果
        if save_detailed:
            await self._save_detailed_comparison(report)
        
        self.comparison_results.append(report)
        return report
    
    async def _analyze_standard_query(self, query: str, mode: str) -> Dict[str, Any]:
        """分析标准查询的检索信息"""
        try:
            # 获取上下文信息
            context = await self.rag.lightrag.aquery(
                query, 
                param=self.rag.lightrag.QueryParam(mode=mode, only_need_context=True)
            )
            
            # 简单分析上下文
            lines = context.split('\n')
            retrieval_count = len([line for line in lines if line.strip() and len(line.strip()) > 20])
            
            # 估算分数（基于上下文长度和质量）
            avg_score = min(1.0, len(context) / 1000)  # 简单的启发式评分
            
            return {
                'retrieval_count': retrieval_count,
                'avg_score': avg_score,
                'source_distribution': {'context': retrieval_count},
                'top_results': [{'content': line[:100], 'score': avg_score} for line in lines[:3] if line.strip()],
                'context_length': len(context)
            }
            
        except Exception as e:
            return {
                'retrieval_count': 0,
                'avg_score': 0.0,
                'source_distribution': {},
                'top_results': [],
                'error': str(e)
            }
    
    def _calculate_improvement_metrics(
        self, 
        standard: QueryResult, 
        efr: QueryResult
    ) -> Dict[str, float]:
        """计算改进指标"""
        metrics = {}
        
        # 检索数量改进
        if standard.retrieval_count > 0:
            metrics['retrieval_count_improvement'] = (
                (efr.retrieval_count - standard.retrieval_count) / standard.retrieval_count * 100
            )
        else:
            metrics['retrieval_count_improvement'] = 100.0 if efr.retrieval_count > 0 else 0.0
        
        # 平均分数改进
        if standard.avg_score > 0:
            metrics['score_improvement'] = (
                (efr.avg_score - standard.avg_score) / standard.avg_score * 100
            )
        else:
            metrics['score_improvement'] = 100.0 if efr.avg_score > 0 else 0.0
        
        # 处理时间变化
        if standard.processing_time > 0:
            metrics['time_change'] = (
                (efr.processing_time - standard.processing_time) / standard.processing_time * 100
            )
        else:
            metrics['time_change'] = 0.0
        
        # 响应质量改进（基于长度和结构）
        standard_quality = len(standard.response) / 1000  # 简单启发式
        efr_quality = len(efr.response) / 1000
        if standard_quality > 0:
            metrics['response_quality_improvement'] = (
                (efr_quality - standard_quality) / standard_quality * 100
            )
        else:
            metrics['response_quality_improvement'] = 100.0 if efr_quality > 0 else 0.0
        
        return metrics
    
    def _generate_detailed_analysis(
        self, 
        standard: QueryResult, 
        efr: QueryResult
    ) -> Dict[str, Any]:
        """生成详细分析"""
        analysis = {
            'response_comparison': {
                'standard_length': len(standard.response),
                'efr_length': len(efr.response),
                'length_difference': len(efr.response) - len(standard.response),
                'standard_preview': standard.response[:200] + "..." if len(standard.response) > 200 else standard.response,
                'efr_preview': efr.response[:200] + "..." if len(efr.response) > 200 else efr.response,
            },
            'retrieval_analysis': {
                'standard_sources': standard.source_distribution,
                'efr_sources': efr.source_distribution,
                'source_diversity': len(efr.source_distribution) - len(standard.source_distribution),
            },
            'performance_analysis': {
                'standard_time': standard.processing_time,
                'efr_time': efr.processing_time,
                'time_overhead': efr.processing_time - standard.processing_time,
                'time_overhead_percent': ((efr.processing_time - standard.processing_time) / standard.processing_time * 100) if standard.processing_time > 0 else 0,
            },
            'quality_indicators': {
                'standard_avg_score': standard.avg_score,
                'efr_avg_score': efr.avg_score,
                'score_improvement': efr.avg_score - standard.avg_score,
                'standard_top_results': len(standard.top_results),
                'efr_top_results': len(efr.top_results),
            }
        }
        
        return analysis
    
    def _display_comparison(self, report: ComparisonReport):
        """显示对比结果"""
        print(f"\n📊 对比结果 - 查询: {report.query}")
        print("=" * 80)
        
        # 基本指标对比
        print("\n📈 基本指标对比:")
        print(f"{'指标':<25} {'标准查询':<15} {'EFR查询':<15} {'改进':<15}")
        print("-" * 70)
        print(f"{'检索结果数量':<25} {report.standard_result.retrieval_count:<15} {report.efr_result.retrieval_count:<15} {report.improvement_metrics.get('retrieval_count_improvement', 0):+.1f}%")
        print(f"{'平均分数':<25} {report.standard_result.avg_score:<15.3f} {report.efr_result.avg_score:<15.3f} {report.improvement_metrics.get('score_improvement', 0):+.1f}%")
        print(f"{'处理时间(秒)':<25} {report.standard_result.processing_time:<15.3f} {report.efr_result.processing_time:<15.3f} {report.improvement_metrics.get('time_change', 0):+.1f}%")
        print(f"{'响应长度':<25} {len(report.standard_result.response):<15} {len(report.efr_result.response):<15} {report.improvement_metrics.get('response_quality_improvement', 0):+.1f}%")
        
        # 来源分布对比
        print(f"\n🔍 检索来源分布对比:")
        print(f"{'来源类型':<20} {'标准查询':<15} {'EFR查询':<15}")
        print("-" * 50)
        
        all_sources = set(report.standard_result.source_distribution.keys()) | set(report.efr_result.source_distribution.keys())
        for source in all_sources:
            standard_count = report.standard_result.source_distribution.get(source, 0)
            efr_count = report.efr_result.source_distribution.get(source, 0)
            print(f"{source:<20} {standard_count:<15} {efr_count:<15}")
        
        # 响应质量对比
        print(f"\n💬 响应质量对比:")
        print(f"\n📝 标准查询响应:")
        print(f"{report.standard_result.response[:300]}...")
        print(f"\n🚀 EFR增强响应:")
        print(f"{report.efr_result.response[:300]}...")
        
        # 改进总结
        print(f"\n🎯 改进总结:")
        improvements = []
        if report.improvement_metrics.get('retrieval_count_improvement', 0) > 0:
            improvements.append(f"检索数量提升 {report.improvement_metrics['retrieval_count_improvement']:.1f}%")
        if report.improvement_metrics.get('score_improvement', 0) > 0:
            improvements.append(f"平均分数提升 {report.improvement_metrics['score_improvement']:.1f}%")
        if report.improvement_metrics.get('response_quality_improvement', 0) > 0:
            improvements.append(f"响应质量提升 {report.improvement_metrics['response_quality_improvement']:.1f}%")
        
        if improvements:
            print("✅ " + ", ".join(improvements))
        else:
            print("⚠️ 未发现显著改进")
        
        if report.improvement_metrics.get('time_change', 0) > 0:
            print(f"⏱️ 处理时间增加 {report.improvement_metrics['time_change']:.1f}%")
    
    async def _save_detailed_comparison(self, report: ComparisonReport):
        """保存详细对比结果到文件"""
        try:
            # 创建输出目录
            output_dir = Path("./efr_comparison_results")
            output_dir.mkdir(exist_ok=True)
            
            # 生成文件名
            query_safe = "".join(c for c in report.query if c.isalnum() or c in (' ', '-', '_')).rstrip()
            query_safe = query_safe.replace(' ', '_')[:50]
            timestamp = int(time.time())
            filename = f"comparison_{query_safe}_{timestamp}.json"
            
            # 保存详细结果
            detailed_data = {
                'timestamp': timestamp,
                'query': report.query,
                'standard_result': asdict(report.standard_result),
                'efr_result': asdict(report.efr_result),
                'improvement_metrics': report.improvement_metrics,
                'detailed_analysis': report.detailed_analysis
            }
            
            filepath = output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 详细对比结果已保存到: {filepath}")
            
        except Exception as e:
            print(f"⚠️ 保存详细结果失败: {e}")
    
    async def batch_comparison(
        self, 
        queries: List[str], 
        mode: str = "hybrid"
    ) -> List[ComparisonReport]:
        """批量对比多个查询"""
        print(f"\n🔄 开始批量对比 {len(queries)} 个查询...")
        
        reports = []
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] 处理查询: {query[:50]}...")
            try:
                report = await self.compare_single_query(query, mode, save_detailed=True)
                reports.append(report)
            except Exception as e:
                print(f"❌ 查询 {i} 失败: {e}")
                continue
        
        # 生成批量对比总结
        self._generate_batch_summary(reports)
        
        return reports
    
    def _generate_batch_summary(self, reports: List[ComparisonReport]):
        """生成批量对比总结"""
        if not reports:
            print("❌ 没有成功的对比结果")
            return
        
        print(f"\n📊 批量对比总结 ({len(reports)} 个查询)")
        print("=" * 80)
        
        # 计算平均改进指标
        avg_metrics = {}
        for metric in ['retrieval_count_improvement', 'score_improvement', 'time_change', 'response_quality_improvement']:
            values = [r.improvement_metrics.get(metric, 0) for r in reports]
            avg_metrics[metric] = sum(values) / len(values)
        
        print(f"\n📈 平均改进指标:")
        print(f"检索数量改进: {avg_metrics['retrieval_count_improvement']:+.1f}%")
        print(f"平均分数改进: {avg_metrics['score_improvement']:+.1f}%")
        print(f"处理时间变化: {avg_metrics['time_change']:+.1f}%")
        print(f"响应质量改进: {avg_metrics['response_quality_improvement']:+.1f}%")
        
        # 成功率统计
        successful_improvements = sum(1 for r in reports if any(
            r.improvement_metrics.get(metric, 0) > 0 
            for metric in ['retrieval_count_improvement', 'score_improvement', 'response_quality_improvement']
        ))
        
        print(f"\n🎯 改进成功率: {successful_improvements}/{len(reports)} ({successful_improvements/len(reports)*100:.1f}%)")
        
        # 保存批量总结
        try:
            output_dir = Path("./efr_comparison_results")
            output_dir.mkdir(exist_ok=True)
            
            summary_data = {
                'total_queries': len(reports),
                'successful_queries': len(reports),
                'average_metrics': avg_metrics,
                'improvement_success_rate': successful_improvements / len(reports),
                'individual_reports': [asdict(r) for r in reports]
            }
            
            timestamp = int(time.time())
            summary_file = output_dir / f"batch_summary_{timestamp}.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 批量对比总结已保存到: {summary_file}")
            
        except Exception as e:
            print(f"⚠️ 保存批量总结失败: {e}")


async def setup_rag_for_comparison():
    """设置用于对比的RAG实例"""
    print("🚀 设置RAGAnything用于EFR对比测试...")
    
    # Check API configuration
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST", "https://yinli.one/v1")
    
    if not api_key:
        print("❌ 未找到 API 密钥，请检查环境变量 LLM_BINDING_API_KEY")
        return None
    
    # Create EFR configuration
    efr_config = EFRConfig(
        rrf_k=60,
        rrf_weights={
            "vector": 1.0,
            "entity": 0.8,
            "relation": 0.7,
            "chunk": 0.9,
        },
        enable_rerank=True,
        rerank_top_k=20,
        min_rerank_score=0.3,
        enable_mmr=True,
        mmr_lambda=0.7,
        mmr_top_k=10,
        enable_recency=True,
        recency_weight=0.2,
        enable_source_trust=True,
        source_trust_weights={
            "academic": 1.0,
            "official": 0.9,
            "news": 0.7,
            "blog": 0.5,
            "unknown": 0.6,
        },
    )
    
    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./efr_comparison_storage",
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
    
    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        efr_config=efr_config,
        enable_efr=True,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )
    
    return rag


async def main():
    """主函数 - 演示EFR对比功能"""
    print("🌟 RAGAnything EFR Layer 效果对比测试")
    print("=" * 60)
    
    # 设置RAG
    rag = await setup_rag_for_comparison()
    if not rag:
        return
    
    # 处理示例文档
    sample_doc = project_root / "example_doc" / "2005.11401v4.pdf"
    if sample_doc.exists():
        print(f"\n📄 处理示例文档: {sample_doc.name}")
        try:
            await rag.process_document_complete(
                file_path=str(sample_doc),
                output_dir="./efr_comparison_output",
                parse_method="auto"
            )
            print("✅ 文档处理完成")
        except Exception as e:
            print(f"⚠️ 文档处理失败: {e}")
    
    # 创建对比测试器
    comparator = EFRComparison(rag)
    
    # 测试查询列表
    test_queries = [
        "什么是机器学习的主要应用领域？",
        "深度学习与传统机器学习有什么区别？",
        "人工智能在医疗领域有哪些应用？",
        "神经网络的基本原理是什么？",
        "数据挖掘的主要技术有哪些？",
    ]
    
    print(f"\n🔍 开始对比测试...")
    
    # 选择测试模式
    print("\n请选择测试模式:")
    print("1. 单个查询详细对比")
    print("2. 批量查询对比")
    print("3. 自定义查询对比")
    
    try:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            # 单个查询详细对比
            query = test_queries[0]
            await comparator.compare_single_query(query, mode="hybrid", save_detailed=True)
            
        elif choice == "2":
            # 批量查询对比
            await comparator.batch_comparison(test_queries, mode="hybrid")
            
        elif choice == "3":
            # 自定义查询对比
            custom_query = input("\n请输入你的查询: ").strip()
            if custom_query:
                await comparator.compare_single_query(custom_query, mode="hybrid", save_detailed=True)
            else:
                print("❌ 未输入查询")
                
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n\n👋 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
    
    print("\n🎉 EFR对比测试完成！")
    print("\n📁 结果文件保存在: ./efr_comparison_results/")
    print("📊 查看详细对比数据和统计信息")


if __name__ == "__main__":
    asyncio.run(main())

