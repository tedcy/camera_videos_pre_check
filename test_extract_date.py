#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import datetime

def extract_date_from_path(file_path, source_dir):
    """从文件路径中提取日期信息，返回YYYYMMDD格式的字符串和额外信息"""
    print(f"正在提取日期，文件路径: {file_path}")
    print(f"源目录: {source_dir}")
    
    # 提取子目录名
    # 首先规范化路径，确保使用正确的路径分隔符
    norm_path = os.path.normpath(file_path)
    norm_source = os.path.normpath(source_dir)
    
    print(f"规范化后的文件路径: {norm_path}")
    print(f"规范化后的源目录: {norm_source}")
    
    # 检查文件路径是否包含源目录
    if norm_source in norm_path:
        print(f"文件路径包含源目录")
        # 获取源目录之后的路径部分
        rel_path = os.path.relpath(norm_path, norm_source)
        print(f"相对路径: {rel_path}")
        parts = rel_path.split(os.sep)
        print(f"路径部分: {parts}")
        
        # 提取日期目录之前的所有部分作为子目录名
        date_dir_index = -1
        for i, part in enumerate(parts):
            if re.match(r'^\d{10}$', part):  # 查找形如2025051300的日期目录
                date_dir_index = i
                break
        
        if date_dir_index > 0:
            # 如果找到日期目录，取其之前的所有部分作为子目录名
            subdir_parts = parts[:date_dir_index]
            subdir_name = os.path.join(*subdir_parts)
            print(f"找到日期目录，之前的部分作为子目录名: {subdir_name}")
        else:
            # 如果没有找到日期目录，取除了最后一个部分（文件名）之外的所有部分作为子目录名
            if len(parts) > 1:
                subdir_parts = parts[:-1]  # 排除最后一个部分（文件名）
                subdir_name = os.path.join(*subdir_parts)
                print(f"未找到日期目录，使用除文件名外的所有部分作为子目录名: {subdir_name}")
            else:
                subdir_name = ""
                print(f"路径部分不足，无法提取子目录名")
    else:
        print(f"文件路径不包含源目录，尝试使用其他方法")
        # 如果文件路径不包含源目录，尝试使用其他方法提取子目录名
        parts = norm_path.split(os.sep)
        print(f"路径部分: {parts}")
        
        # 尝试找到日期目录，并取其之前的部分作为子目录名
        date_dir_index = -1
        for i, part in enumerate(parts):
            if re.match(r'^\d{10}$', part):  # 查找形如2025051300的日期目录
                date_dir_index = i
                break
        
        if date_dir_index > 0:
            # 如果找到日期目录，取其之前的最后一个部分作为子目录名
            subdir_name = parts[date_dir_index-1]
            print(f"找到日期目录，之前的部分作为子目录名: {subdir_name}")
        else:
            # 如果没有找到日期目录，使用启发式方法
            subdir_name = ""
            for part in parts:
                if not re.match(r'^\d+$', part):  # 不是纯数字的部分可能是子目录名
                    print(f"检查部分: {part}")
                    if part and not part.startswith('.'):
                        subdir_name = part
                        print(f"找到可能的子目录名: {subdir_name}")
            print(f"使用启发式方法提取的子目录名: {subdir_name}")
    
    # 尝试从路径中提取日期
    # 格式1: .../2025051122/45M49S_1746855949.mp4
    pattern1 = r'/(\d{10})/'
    match = re.search(pattern1, file_path)
    if match:
        date_str = match.group(1)
        date_part = date_str[:8]  # YYYYMMDD部分
        hour_part = date_str[8:10]  # HH部分
        print(f"匹配格式1，提取日期: {date_part}，小时: {hour_part}，子目录: {subdir_name}")
        return date_part, hour_part, "format1", subdir_name
    
    # 格式2: .../00_20250516014118_20250516014118.mp4
    pattern2 = r'_(\d{14})_'
    match = re.search(pattern2, file_path)
    if match:
        date_str = match.group(1)
        date_part = date_str[:8]  # YYYYMMDD部分
        print(f"匹配格式2，提取日期: {date_part}，子目录: {subdir_name}")
        return date_part, "", "format2", subdir_name
    
    # 如果无法提取日期，使用当前日期
    print(f"无法提取日期，使用当前日期，子目录: {subdir_name}")
    return datetime.datetime.now().strftime('%Y%m%d'), "", "unknown", subdir_name

# 测试函数
def test_extract_date():
    # 测试路径
    test_cases = [
        {
            "file_path": "/volume1/camera_save/xiaomi_camera_videos/78DF72BBDC20/2025051223/52M00S_1747065120.mp4",
            "source_dir": "/volume1/camera_save/xiaomi_camera_videos",
            "expected_subdir": "78DF72BBDC20",
            "target_dir": "/volume1/baidu/xiaomi_camera_videos"
        },
        {
            "file_path": "/volume1/camera_save/xiaomi_camera_videos/78DF72BBDC20/2025051300/48M05S_1747068485.mp4",
            "source_dir": "/volume1/camera_save",
            "expected_subdir": "xiaomi_camera_videos/78DF72BBDC20",
            "target_dir": "/volume1/baidu/xiaomi_camera_videos"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        file_path = test_case["file_path"]
        source_dir = test_case["source_dir"]
        expected_subdir = test_case["expected_subdir"]
        target_dir = test_case["target_dir"]
        
        # 调用函数
        date_part, hour_part, format_type, subdir_name = extract_date_from_path(file_path, source_dir)
        
        # 打印结果
        print("\n测试结果:")
        print(f"日期部分: {date_part}")
        print(f"小时部分: {hour_part}")
        print(f"格式类型: {format_type}")
        print(f"子目录名: {subdir_name}")
        print(f"期望的子目录名: {expected_subdir}")
        print(f"子目录名是否匹配: {'是' if subdir_name == expected_subdir else '否'}")
        
        # 构建目标路径
        if subdir_name:
            # 检查子目录名是否包含目标目录的一部分，避免重复
            target_base = os.path.basename(target_dir)
            print(f"目标目录基名: {target_base}")
            if target_base in subdir_name.split(os.sep):
                print(f"子目录名包含目标目录的一部分，需要去除重复")
                # 如果子目录名包含目标目录的一部分，则去除重复部分
                subdir_parts = subdir_name.split(os.sep)
                print(f"子目录部分: {subdir_parts}")
                if target_base in subdir_parts:
                    subdir_parts.remove(target_base)
                    print(f"移除重复部分后的子目录部分: {subdir_parts}")
                subdir_name = os.path.join(*subdir_parts) if subdir_parts else ""
                print(f"处理后的子目录名: {subdir_name}")
            
            dst_dir = os.path.join(target_dir, subdir_name, date_part) if subdir_name else os.path.join(target_dir, date_part)
        else:
            # 如果没有子目录名，则只使用日期
            dst_dir = os.path.join(target_dir, date_part)
        
        file_name = os.path.basename(file_path)
        if format_type == "format1" and hour_part:
            new_file_name = f"{hour_part}_{file_name}"
        else:
            new_file_name = file_name
        
        dst_path = os.path.join(dst_dir, new_file_name)
        print(f"目标路径: {dst_path}")

if __name__ == "__main__":
    test_extract_date()
