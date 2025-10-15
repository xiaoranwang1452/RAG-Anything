#!/usr/bin/env python3
"""
SPIQA Test-B 断点续传脚本
从上次保存的进度继续测试
"""

import json
import os
import sys
from pathlib import Path

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

def continue_test():
    """继续测试"""
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
    
    print('\n💡 要继续测试，请运行:')
    print(f'   python test_spiqa_comprehensive.py --test_number 0')

if __name__ == '__main__':
    continue_test()