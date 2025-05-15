import os
import shutil
import cv2
import time
import datetime
import configparser
import numpy as np
from skimage.metrics import structural_similarity as ssim

def load_config(config_path='config.ini'):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    return {
        'source_dir': config.get('Directories', 'source_dir'),
        'target_dir': config.get('Directories', 'target_dir'),
        'video_extensions': [ext.strip() for ext in config.get('Directories', 'video_extensions').split(',')],
        'max_folder_age_days': config.getint('Directories', 'max_folder_age_days'),
        'detection_algorithm': config.get('VideoProcessing', 'detection_algorithm', fallback='ssim'),
        'ssim_threshold': config.getfloat('VideoProcessing', 'ssim_threshold'),
        'histogram_threshold': config.getfloat('VideoProcessing', 'histogram_threshold', fallback=0.15),
        'pixel_diff_threshold': config.getfloat('VideoProcessing', 'pixel_diff_threshold', fallback=30),
        'sample_interval_sec': config.getfloat('VideoProcessing', 'sample_interval_sec')
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

def has_scene_change(video_path, detection_algorithm='ssim', ssim_threshold=0.95, 
                    histogram_threshold=0.15, pixel_diff_threshold=30, sample_interval_sec=1):
    """检测视频是否有画面变化，支持多种算法"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or total_frames <= 0:
        cap.release()
        print(f"视频无效: {video_path}")
        return False

    interval = int(fps * sample_interval_sec)
    if interval <= 0:
        interval = 1
        
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        print(f"视频无法读取: {video_path}")
        return False

    # 根据算法预处理第一帧
    if detection_algorithm == 'ssim':
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (3,3), 0)

    frame_index = 0
    change_detected = False

    while True:
        frame_index += interval
        if frame_index >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        # 根据选择的算法检测画面变化
        if detection_algorithm == 'ssim':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            similarity, _ = ssim(prev_gray, gray, full=True)
            if similarity < ssim_threshold:
                change_detected = True
                break
            prev_gray = gray
            
        elif detection_algorithm == 'histogram':
            diff = compare_histogram(prev_frame, frame)
            if diff > histogram_threshold:
                change_detected = True
                break
                
        elif detection_algorithm == 'pixel_diff':
            diff = compare_pixel_diff(prev_frame, frame)
            if diff > pixel_diff_threshold:
                change_detected = True
                break
        
        prev_frame = frame

    cap.release()
    return change_detected

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
        print(f"开始处理视频: {src_path}")
        if has_scene_change(
            src_path, 
            detection_algorithm=config['detection_algorithm'],
            ssim_threshold=config['ssim_threshold'],
            histogram_threshold=config['histogram_threshold'],
            pixel_diff_threshold=config['pixel_diff_threshold'],
            sample_interval_sec=config['sample_interval_sec']
        ):
            # 确保目标目录存在
            dst_dir = os.path.dirname(dst_path)
            ensure_dir(dst_dir)
            
            # 有画面变化，移动到目标目录
            shutil.move(src_path, dst_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"有画面变化，已移动到: {dst_path}，耗时: {elapsed_time:.2f}秒")
            return True
        else:
            # 无画面变化，删除
            os.remove(src_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"无画面变化，已删除: {src_path}，耗时: {elapsed_time:.2f}秒")
            return False
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"处理出错: {src_path}, 错误: {e}，耗时: {elapsed_time:.2f}秒")
        return False

def process_directory(config):
    """处理源目录下的所有子文件夹"""
    source_dir = config['source_dir']
    target_dir = config['target_dir']
    video_extensions = config['video_extensions']
    max_folder_age_days = config['max_folder_age_days']
    
    # 遍历源目录下的所有子文件夹
    for root, dirs, files in os.walk(source_dir):
        # 检查子文件夹是否超过最大保留天数
        rel_path = os.path.relpath(root, source_dir)
        if rel_path != '.' and is_folder_older_than_days(root, max_folder_age_days):
            print(f"文件夹创建超过{max_folder_age_days}天，删除: {root}")
            shutil.rmtree(root)
            continue
        
        # 处理当前文件夹中的视频文件
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                src_path = os.path.join(root, file)
                
                # 计算目标路径，保持相同的目录结构
                rel_path = os.path.relpath(root, source_dir)
                if rel_path == '.':
                    dst_dir = target_dir
                else:
                    dst_dir = os.path.join(target_dir, rel_path)
                
                dst_path = os.path.join(dst_dir, file)
                process_video(src_path, dst_path, config)

def main():
    """主函数"""
    try:
        # 加载配置
        config = load_config()
        
        print(f"开始处理视频文件...")
        print(f"源目录: {config['source_dir']}")
        print(f"目标目录: {config['target_dir']}")
        print(f"检测算法: {config['detection_algorithm']}")
        
        # 确保目标根目录存在
        ensure_dir(config['target_dir'])
        
        # 处理所有子目录
        process_directory(config)
        
        print("处理完成!")
        
    except Exception as e:
        print(f"程序出错: {e}")

if __name__ == "__main__":
    main()
