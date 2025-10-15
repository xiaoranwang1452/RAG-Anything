#!/usr/bin/env python3
"""
保存当前测试进度
"""

import json
import os
from datetime import datetime

def save_progress():
    """保存当前测试进度"""
    if os.path.exists('spiqa_comprehensive_results.json'):
        # 创建带时间戳的备份
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f'spiqa_comprehensive_results_backup_{timestamp}.json'
        
        # 复制当前结果文件
        with open('spiqa_comprehensive_results.json', 'r') as src:
            data = json.load(src)
        
        with open(backup_file, 'w') as dst:
            json.dump(data, dst, indent=2, ensure_ascii=False)
        
        print(f'✅ 进度已保存到: {backup_file}')
        
        # 统计当前进度
        papers = set()
        for key in data.keys():
            paper_id = key.split('_')[0]
            papers.add(paper_id)
        
        print(f'📊 已处理论文: {len(papers)}')
        print(f'📊 已处理问题: {len(data)}')
        
        # 计算准确率
        correct_count = 0
        for item in data.values():
            if 'evaluation' in item and item['evaluation']['is_correct']:
                correct_count += 1
        
        accuracy = correct_count / len(data) * 100 if len(data) > 0 else 0
        print(f'📈 当前准确率: {accuracy:.1f}%')
        
        return backup_file
    else:
        print('❌ 未找到结果文件')
        return None

if __name__ == '__main__':
    save_progress()
