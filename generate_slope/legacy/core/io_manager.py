import os
import laspy
import json
from datetime import datetime

class IOManager:
    @staticmethod
    def load_las(path):
        """读取指定路径的 LAS 文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        return laspy.read(path)

    @staticmethod
    def save_result(las_obj, args):
        """
        保存处理后的点云，并生成包含操作参数的文件夹
        """
        # 1. 构建文件夹名称和路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 文件夹命名规则：状态_方向_平滑方式_时间戳
        folder_name = f"{args.slope_status}_{args.z_direction}_{args.smooth_type}_{timestamp}"
        target_dir = os.path.join(args.out_dir_base, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        # 2. 保存 LAS 文件
        # 使用原始文件名作为基础，保存在新创建的文件夹内
        out_las_path = os.path.join(target_dir, "modified_data.las")
        try:
            las_obj.write(out_las_path)
            print(f"LAS file saved to: {out_las_path}")
        except Exception as e:
            print(f"Failed to write LAS: {e}")

        # 3. 生成操作记录文件 (JSON格式)
        log_path = os.path.join(target_dir, "operation_log.json")
        log_content = {
            "description": "Point cloud editing record",
            "timestamp": timestamp,
            "parameters": args.to_dict()  # 自动包含所有配置参数
        }

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_content, f, indent=4)
        
        print(f"Log file saved to: {log_path}")
        return target_dir
    
    # 1.14 新增：直接保存 LAS（给 batch 脚本用）
    @staticmethod
    def save_las_direct(las_data, out_path):
        """
        直接保存 las 文件（不依赖 Args）
        用于批量增强、自动生成数据等场景
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        las_data.write(out_path)
