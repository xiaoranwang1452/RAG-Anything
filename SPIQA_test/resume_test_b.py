#!/usr/bin/env python3
"""
SPIQA Test-B 断点续传脚本
从上次保存的进度继续测试
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# 导入测试脚本的组件
sys.path.append(str(Path(__file__).parent))
from test_spiqa_comprehensive import ComprehensiveSPIQATester, main

def load_progress():
    """加载之前的测试进度"""
    backup_files = []
    for file in os.listdir('.'):
        if file.startswith('spiqa_comprehensive_results_backup_') and file.endswith('.json'):
            backup_files.append(file)
    
    if not backup_files:
        print('❌ 未找到备份文件')
        return None, set()
    
    # 选择最新的备份文件
    latest_backup = sorted(backup_files)[-1]
    print(f'📁 加载备份文件: {latest_backup}')
    
    with open(latest_backup, 'r') as f:
        data = json.load(f)
    
    # 获取已处理的论文ID
    processed_papers = set()
    for key in data.keys():
        paper_id = key.split('_')[0]
        processed_papers.add(paper_id)
    
    print(f'📊 已处理论文数: {len(processed_papers)}')
    print(f'📊 已处理问题数: {len(data)}')
    
    return data, processed_papers

async def resume_test():
    """从断点继续测试"""
    print('🚀 开始断点续传测试...')
    
    # 加载进度
    existing_data, processed_papers = load_progress()
    if existing_data is None:
        return
    
    # 加载完整数据集
    with open('dataset/test-B/SPIQA_testB.json', 'r') as f:
        full_data = json.load(f)
    
    # 找出未处理的论文
    all_papers = set(full_data.keys())
    remaining_papers = all_papers - processed_papers
    
    print(f'📄 总论文数: {len(all_papers)}')
    print(f'📄 已处理: {len(processed_papers)}')
    print(f'📄 剩余: {len(remaining_papers)}')
    
    if not remaining_papers:
        print('🎉 所有论文已处理完成！')
        return
    
    print('\n📋 剩余论文:')
    for i, paper_id in enumerate(sorted(remaining_papers), 1):
        print(f'  {i:2d}. {paper_id}')
    
    # 创建测试器
    image_root = "dataset/test-B/SPIQA_testB_Images" if os.path.exists("dataset/test-B/SPIQA_testB_Images") else ""
    tester = ComprehensiveSPIQATester(image_root)
    
    # 继续处理剩余论文
    all_results = existing_data.copy()
    processed_count = len(existing_data)
    failed_count = 0
    correct_count = sum(1 for item in existing_data.values() if 'evaluation' in item and item['evaluation']['is_correct'])
    
    # 统计信息
    similarity_scores = [item['evaluation']['similarity_score'] for item in existing_data.values() if 'evaluation' in item]
    phrase_overlaps = [item['evaluation']['phrase_overlap'] for item in existing_data.values() if 'evaluation' in item]
    question_types = {}
    paper_stats = {}
    
    # 初始化统计
    for item in existing_data.values():
        if 'evaluation' in item:
            q_type = item.get('question_type', 'Unknown')
            if q_type not in question_types:
                question_types[q_type] = {'total': 0, 'correct': 0}
            question_types[q_type]['total'] += 1
            if item['evaluation']['is_correct']:
                question_types[q_type]['correct'] += 1
    
    print(f'\n🔄 继续处理剩余论文...')
    
    for paper_id in sorted(remaining_papers):
        print(f'\n🔬 处理论文: {paper_id}')
        paper_content = full_data[paper_id]
        
        try:
            paper_results = await tester.process_paper(paper_id, paper_content)
            
            paper_correct = 0
            paper_total = 0
            
            for res in paper_results:
                qid = f"{res['paper_id']}_q{res['question_index']}"
                all_results[qid] = res
                
                if "error" in res:
                    failed_count += 1
                else:
                    processed_count += 1
                    paper_total += 1
                    
                    # 收集统计信息
                    if "evaluation" in res:
                        evaluation = res["evaluation"]
                        similarity_scores.append(evaluation["similarity_score"])
                        phrase_overlaps.append(evaluation["phrase_overlap"])
                        
                        if evaluation["is_correct"]:
                            correct_count += 1
                            paper_correct += 1
                        
                        # 问题类型统计
                        q_type = res.get("question_type", "Unknown")
                        if q_type not in question_types:
                            question_types[q_type] = {"total": 0, "correct": 0}
                        question_types[q_type]["total"] += 1
                        if evaluation["is_correct"]:
                            question_types[q_type]["correct"] += 1
            
            paper_stats[paper_id] = {
                "total_questions": paper_total,
                "correct_questions": paper_correct,
                "accuracy": paper_correct / paper_total if paper_total > 0 else 0
            }
            
            # 保存中间结果
            with open("spiqa_comprehensive_results.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f'  ✅ 完成: {paper_correct}/{paper_total} 正确')
            
        except Exception as e:
            print(f'  ❌ 处理论文 {paper_id} 时出错: {e}')
            failed_count += 1
    
    # 计算最终统计
    overall_accuracy = correct_count / processed_count if processed_count > 0 else 0
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    avg_phrase_overlap = sum(phrase_overlaps) / len(phrase_overlaps) if phrase_overlaps else 0
    
    print("\n" + "="*60)
    print("📈 最终结果摘要")
    print("="*60)
    print(f"📊 总问题数: {processed_count + failed_count}")
    print(f"✅ 成功处理: {processed_count}")
    print(f"❌ 失败: {failed_count}")
    print(f"🎯 正确回答: {correct_count}")
    print(f"📈 总体准确率: {overall_accuracy:.3f} ({correct_count}/{processed_count})")
    print(f"📊 平均相似度: {avg_similarity:.3f}")
    print(f"📊 平均短语重叠: {avg_phrase_overlap:.3f}")
    
    print(f"\n📋 问题类型表现:")
    for q_type, stats in question_types.items():
        type_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {q_type}: {type_accuracy:.3f} ({stats['correct']}/{stats['total']})")
    
    # 保存最终结果
    output_file = "spiqa_comprehensive_results_final.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 最终结果已保存到: {output_file}")

if __name__ == '__main__':
    asyncio.run(resume_test())
