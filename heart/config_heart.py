# heart/config_heart.py
"""心脏数据集配置 - 所有脚本共用"""
from pathlib import Path

class HeartConfig:
    # 路径配置
    RAW_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/raw")
    GT_DIR = Path("/ihome/jbwang/liy121/ifimage/heart/ground_truth_masks")
    OUTPUT_BASE = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results")
    
    # 数据集结构
    REGIONS = ['LA', 'RA', 'LV', 'RV', 'SEP']
    CELL_TYPES = ['Epi', 'Immune', 'Mural']
    
    # 文件路径
    MAPPING_FILE = GT_DIR / 'file_mapping.csv'
    
    @staticmethod
    def get_algo_dir(algo_name):
        """获取算法输出目录"""
        output_base = Path("/ihome/jbwang/liy121/ifimage/heart/benchmark_results")
        return output_base / f"{algo_name}_predictions"