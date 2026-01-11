#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import cv2
import time
import datetime
import configparser
import numpy as np
import concurrent.futures
import re
from skimage.metrics import structural_similarity as ssim

def load_config(config_path='config.ini'):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise IOError(f"配置文件不存在: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    return {
        'source_dir': config.get('Directories', 'source_dir'),
        'target_dir': config.get('Directories', 'target_dir'),
        'video_extensions': [ext.strip() for ext in config.get('Directories', 'video_extensions').split(',')],
        'empty_folder_age_days': config.getint('Directories', 'empty_folder_age_days', fallback=3),
        'detection_algorithm': config.get('VideoProcessing', 'detection_algorithm', fallback='ssim'),
        'ssim_threshold': config.getfloat('VideoProcessing', 'ssim_threshold'),
        'ssim_scale': config.getfloat('VideoProcessing', 'ssim_scale', fallback=0.5),
        'histogram_threshold': config.getfloat('VideoProcessing', 'histogram_threshold', fallback=0.15),
        'pixel_diff_threshold': config.getfloat('VideoProcessing', 'pixel_diff_threshold', fallback=30),
        'sample_interval_sec': config.getfloat('VideoProcessing', 'sample_interval_sec'),
        'max_workers': config.getint('VideoProcessing', 'max_workers', fallback=4),
        'allowed_sizes_mb': [int(size.strip()) for size in config.get('VideoProcessing', 'allowed_sizes_mb').split(',') if size.strip()]
    }

def compare_histogram(prev_frame, curr_frame):
    """使用直方图比较两帧的差异，返回差异值 (0-1.0)"""
    # 转换为HSV颜色空间，对光照变化更鲁棒
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    
    # 计算H通道的直方图
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    channels = [0, 1]  # H和S通道
    
    prev_hist = cv2.calcHist([prev_hsv], channels, None, histSize, ranges)
    curr_hist = cv2.calcHist([curr_hsv], channels, None, histSize, ranges)
    
    # 归一化直方图
    cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
    
    # 比较直方图，返回差异值
    # 使用HISTCMP_BHATTACHARYYA方法，返回值范围0-1，值越大差异越大
    diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
    return diff

def compare_pixel_diff(prev_frame, curr_frame):
    """计算两帧之间的平均像素差异"""
    # 转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # 计算绝对差异
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # 返回平均差异值
    return np.mean(diff)

def _detect_scene_change_part(video_path, detection_algorithm, ssim_threshold, ssim_scale, histogram_threshold, pixel_diff_threshold, sample_interval_sec, start_frame, end_frame):
    """ 片段检测，有变化即返回 True """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频片段: {video_path}", flush=True)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame >= total_frames:
        cap.release()
        return False

    interval = max(int(fps * sample_interval_sec), 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return False

    max_diff_value = 0
    if detection_algorithm == 'ssim':
        prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (3,3), 0)
        if ssim_scale and ssim_scale < 1.0:
            prev_gray = cv2.resize(prev_gray, (0,0), fx=ssim_scale, fy=ssim_scale, interpolation=cv2.INTER_AREA)
        max_diff_value = 1.0  # For SSIM, we look for the minimum value

    frame_index = start_frame
    while frame_index <= end_frame:
        frame_index += interval
        if frame_index >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        if detection_algorithm == 'ssim':
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3,3), 0)
            if ssim_scale and ssim_scale < 1.0:
                gray = cv2.resize(gray, (0,0), fx=ssim_scale, fy=ssim_scale, interpolation=cv2.INTER_AREA)
            similarity = ssim(prev_gray, gray)
            max_diff_value = min(max_diff_value, similarity)
            if similarity < ssim_threshold:
                print(f"SSIM change detected: {similarity:.4f} < {ssim_threshold}", flush=True)
                cap.release()
                return True
            prev_gray = gray
        elif detection_algorithm == 'histogram':
            diff = compare_histogram(prev_frame, frame)
            max_diff_value = max(max_diff_value, diff)
            if diff > histogram_threshold:
                print(f"Histogram change detected: {diff:.4f} > {histogram_threshold}", flush=True)
                cap.release()
                return True
        elif detection_algorithm == 'pixel_diff':
            diff = compare_pixel_diff(prev_frame, frame)
            max_diff_value = max(max_diff_value, diff)
            if diff > pixel_diff_threshold:
                print(f"Pixel diff change detected: {diff:.2f} > {pixel_diff_threshold}", flush=True)
                cap.release()
                return True
        prev_frame = frame
    
    if detection_algorithm == 'ssim':
        print(f"片段 {start_frame}-{end_frame}: 最小SSIM值: {max_diff_value:.4f} (阈值: < {ssim_threshold})", flush=True)
    else:
        print(f"片段 {start_frame}-{end_frame}: 最大 {detection_algorithm} 差异: {max_diff_value:.4f}", flush=True)

    cap.release()
    return False

def has_scene_change(video_path, detection_algorithm='ssim', ssim_threshold=0.95, ssim_scale=0.5,
                    histogram_threshold=0.15, pixel_diff_threshold=30, sample_interval_sec=1, max_workers=4):
    """ 调用并行进程处理 """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}", flush=True)
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= max_workers or max_workers <= 1:
        # 帧数少或者单核，直接顺序处理
        return _detect_scene_change_part(video_path, detection_algorithm, ssim_threshold, ssim_scale,
                                         histogram_threshold, pixel_diff_threshold, sample_interval_sec, 0, total_frames)

    frames_per_worker = total_frames // max_workers
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(max_workers):
            start = i * frames_per_worker
            end = (i+1) * frames_per_worker - 1 if i < max_workers - 1 else total_frames - 1
            tasks.append(executor.submit(_detect_scene_change_part, video_path, detection_algorithm,
                                         ssim_threshold, ssim_scale, histogram_threshold, pixel_diff_threshold,
                                         sample_interval_sec, start, end))
        for future in concurrent.futures.as_completed(tasks):
            if future.result():  # 任何片段发现变化，即可提前退出
                return True
    return False

def is_folder_older_than_days(folder_path, days):
    """检查文件夹是否创建超过指定天数"""
    if not os.path.exists(folder_path):
        return False
    
    creation_time = os.path.getctime(folder_path)
    creation_date = datetime.datetime.fromtimestamp(creation_time)
    current_date = datetime.datetime.now()
    age = current_date - creation_date
    
    return age.days > days

def ensure_dir(path):
    """确保目录存在，如不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)

def process_video(src_path, dst_path, config):
    """处理单个视频文件"""
    start_time = time.time()
    try:
        print(f"开始处理视频: {src_path}", flush=True)
        if has_scene_change(
            src_path,
            detection_algorithm=config['detection_algorithm'],
            ssim_threshold=config['ssim_threshold'],
            ssim_scale=config['ssim_scale'],
            histogram_threshold=config['histogram_threshold'],
            pixel_diff_threshold=config['pixel_diff_threshold'],
            sample_interval_sec=config['sample_interval_sec'],
            max_workers=config['max_workers']  # 添加并行处理参数
        ):
            # 确保目标目录存在
            dst_dir = os.path.dirname(dst_path)
            ensure_dir(dst_dir)
            
            # 有画面变化，移动到目标目录
            shutil.move(src_path, dst_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"有画面变化，已移动到: {dst_path}，耗时: {elapsed_time:.2f}秒", flush=True)
            return True
        else:
            # 无画面变化，删除
            os.remove(src_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"无画面变化，已删除: {src_path}，耗时: {elapsed_time:.2f}秒", flush=True)
            return False
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"处理出错: {src_path}, 错误: {e}，耗时: {elapsed_time:.2f}秒", flush=True)
        return False

def is_empty_directory(path):
    """检查目录是否为空"""
    return len(os.listdir(path)) == 0

def extract_date_from_path(file_path, source_dir):
    """从文件路径中提取日期信息，返回YYYYMMDD格式的字符串和额外信息"""
    # 打印当前处理的文件路径，用于调试
    print(f"正在提取日期，文件路径: {file_path}", flush=True)
    print(f"源目录: {source_dir}", flush=True)
    
    # 提取子目录名
    # 首先规范化路径，确保使用正确的路径分隔符
    norm_path = os.path.normpath(file_path)
    norm_source = os.path.normpath(source_dir)
    
    print(f"规范化后的文件路径: {norm_path}", flush=True)
    print(f"规范化后的源目录: {norm_source}", flush=True)
    
    # 检查文件路径是否包含源目录
    if norm_source in norm_path:
        print(f"文件路径包含源目录", flush=True)
        # 获取源目录之后的路径部分
        rel_path = os.path.relpath(norm_path, norm_source)
        print(f"相对路径: {rel_path}", flush=True)
        parts = rel_path.split(os.sep)
        print(f"路径部分: {parts}", flush=True)
        
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
            print(f"找到日期目录，之前的部分作为子目录名: {subdir_name}", flush=True)
        else:
            # 如果没有找到日期目录，取除了最后一个部分（文件名）之外的所有部分作为子目录名
            if len(parts) > 1:
                subdir_parts = parts[:-1]  # 排除最后一个部分（文件名）
                subdir_name = os.path.join(*subdir_parts)
                print(f"未找到日期目录，使用除文件名外的所有部分作为子目录名: {subdir_name}", flush=True)
            else:
                subdir_name = ""
                print(f"路径部分不足，无法提取子目录名", flush=True)
    else:
        print(f"文件路径不包含源目录，尝试使用其他方法", flush=True)
        # 如果文件路径不包含源目录，尝试使用其他方法提取子目录名
        parts = norm_path.split(os.sep)
        print(f"路径部分: {parts}", flush=True)
        
        # 尝试找到日期目录，并取其之前的部分作为子目录名
        date_dir_index = -1
        for i, part in enumerate(parts):
            if re.match(r'^\d{10}$', part):  # 查找形如2025051300的日期目录
                date_dir_index = i
                break
        
        if date_dir_index > 0:
            # 如果找到日期目录，取其之前的最后一个部分作为子目录名
            subdir_name = parts[date_dir_index-1]
            print(f"找到日期目录，之前的部分作为子目录名: {subdir_name}", flush=True)
        else:
            # 如果没有找到日期目录，使用启发式方法
            subdir_name = ""
            for part in parts:
                if not re.match(r'^\d+$', part):  # 不是纯数字的部分可能是子目录名
                    print(f"检查部分: {part}", flush=True)
                    if part and not part.startswith('.'):
                        subdir_name = part
                        print(f"找到可能的子目录名: {subdir_name}", flush=True)
            print(f"使用启发式方法提取的子目录名: {subdir_name}", flush=True)
    
    # 尝试从路径中提取日期
    # 格式1: .../2025051122/45M49S_1746855949.mp4
    pattern1 = r'/(\d{10})/'
    match = re.search(pattern1, file_path)
    if match:
        date_str = match.group(1)
        date_part = date_str[:8]  # YYYYMMDD部分
        hour_part = date_str[8:10]  # HH部分
        print(f"匹配格式1，提取日期: {date_part}，小时: {hour_part}，子目录: {subdir_name}", flush=True)
        return date_part, hour_part, "format1", subdir_name
    
    # 格式2: .../00_20250516014118_20250516014118.mp4
    pattern2 = r'_(\d{14})_'
    match = re.search(pattern2, file_path)
    if match:
        date_str = match.group(1)
        date_part = date_str[:8]  # YYYYMMDD部分
        print(f"匹配格式2，提取日期: {date_part}，子目录: {subdir_name}", flush=True)
        return date_part, "", "format2", subdir_name
    
    # 如果无法提取日期，使用当前日期
    print(f"无法提取日期，使用当前日期，子目录: {subdir_name}", flush=True)
    return datetime.datetime.now().strftime('%Y%m%d'), "", "unknown", subdir_name

def process_directory(config):
    """处理源目录"""
    source_dir = config['source_dir']
    target_dir = config['target_dir']
    video_extensions = config['video_extensions']
    empty_folder_age_days = config['empty_folder_age_days']
    
    print(f"视频扩展名: {video_extensions}", flush=True)
    
    # 确保目标根目录存在
    ensure_dir(target_dir)
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"源目录不存在: {source_dir}", flush=True)
        return
    
    # 获取源目录下的所有一级子目录
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"发现子目录: {subdirs}", flush=True)
    
    for subdir in subdirs:
        subdir_path = os.path.join(source_dir, subdir)
        print(f"处理子目录: {subdir_path}", flush=True)
        
        # 遍历子目录下的所有文件夹
        for root, dirs, files in os.walk(subdir_path, topdown=False):  # 自底向上遍历，先处理最深层目录
            dirs.sort()          # 可选：保证遍历子目录顺序稳定
            files.sort()         # 关键：保证文件顺序稳定
            print(f"当前目录: {root}", flush=True)
            print(f"包含文件: {files}", flush=True)
            
            # 检查是否为空目录
            if is_empty_directory(root):
                print(f"空目录: {root}", flush=True)
                if is_folder_older_than_days(root, empty_folder_age_days):
                    print(f"空目录创建超过{empty_folder_age_days}天，删除: {root}", flush=True)
                    shutil.rmtree(root)
                continue
            
            # 处理当前文件夹中的视频文件
            for file in files:
                print(f"检查文件: {file}", flush=True)
                is_video = any(file.lower().endswith(ext) for ext in video_extensions)
                print(f"是否为视频文件: {is_video}", flush=True)
                
                if is_video:
                    src_path = os.path.join(root, file)
                    print(f"找到视频文件: {src_path}", flush=True)

                    # 检查文件大小
                    if config['allowed_sizes_mb']:
                        file_size_mb = os.path.getsize(src_path) / (1024 * 1024)
                        if int(file_size_mb) not in config['allowed_sizes_mb']:
                            print(f"文件大小 {file_size_mb:.2f}MB 不在允许的列表中，跳过: {src_path}", flush=True)
                            continue
                    
                    # 从文件路径中提取日期和额外信息
                    date_part, hour_part, format_type, subdir_name = extract_date_from_path(src_path, source_dir)
                    
                    # 构建目标路径，保留子目录结构
                    if subdir_name:
                        # 如果有子目录名，则在目标路径中包含子目录名和日期
                        # 检查子目录名是否包含目标目录的一部分，避免重复
                        target_base = os.path.basename(target_dir)
                        if target_base in subdir_name.split(os.sep):
                            # 如果子目录名包含目标目录的一部分，则去除重复部分
                            subdir_parts = subdir_name.split(os.sep)
                            if target_base in subdir_parts:
                                subdir_parts.remove(target_base)
                            subdir_name = os.path.join(*subdir_parts) if subdir_parts else ""
                        
                        dst_dir = os.path.join(target_dir, subdir_name, date_part) if subdir_name else os.path.join(target_dir, date_part)
                    else:
                        # 如果没有子目录名，则只使用日期
                        dst_dir = os.path.join(target_dir, date_part)
                    
                    # 根据格式类型处理文件名
                    file_name, file_ext = os.path.splitext(file)
                    if format_type == "format1" and hour_part:
                        # 格式1：将小时信息添加到文件名前面
                        new_file_name = f"{hour_part}_{file_name}{file_ext}"
                    else:
                        new_file_name = file
                    
                    dst_path = os.path.join(dst_dir, new_file_name)
                    print(f"目标路径: {dst_path}", flush=True)
                    
                    # 如果目标文件已存在，跳过处理
                    if os.path.exists(dst_path):
                        print(f"目标文件已存在，源文件直接删除: {dst_path}", flush=True)
                        os.remove(src_path)
                        continue
                    
                    process_video(src_path, dst_path, config)

def main():
    """主函数"""
    try:
        # 加载配置
        config = load_config()
        
        print(f"开始处理视频文件...", flush=True)
        print(f"源目录: {config['source_dir']}", flush=True)
        print(f"目标目录: {config['target_dir']}", flush=True)
        print(f"检测算法: {config['detection_algorithm']}", flush=True)
        print(f"并行处理核心数: {config['max_workers']}", flush=True)
        print(f"空目录最大保留天数: {config['empty_folder_age_days']}", flush=True)
        
        # 处理源目录
        process_directory(config)
        
        print("处理完成!", flush=True)
        
    except Exception as e:
        print(f"程序出错: {e}", flush=True)

if __name__ == "__main__":
    main()
