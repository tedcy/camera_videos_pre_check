#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import datetime
import configparser
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.ini'):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise IOError("配置文件不存在: {}".format(config_path))
    
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    return {
        'target_dir': config.get('Directories', 'target_dir'),
        'empty_folder_age_days': config.getint('Directories', 'empty_folder_age_days', fallback=3),
        'file_age_days': config.getint('Directories', 'file_age_days', fallback=7),
    }

def is_empty_directory(path):
    """检查目录是否为空"""
    return len(os.listdir(path)) == 0

def is_older_than_days(path, days):
    """检查文件或目录是否修改超过指定天数"""
    if not os.path.exists(path):
        return False
    
    # 使用修改时间而不是创建时间
    modification_time = os.path.getmtime(path)
    modification_date = datetime.datetime.fromtimestamp(modification_time)
    current_date = datetime.datetime.now()
    age = current_date - modification_date
    
    return age.days > days

def clean_empty_directories(target_dir, folder_max_age_days, file_max_age_days):
    """清理超过指定天数的空目录和文件"""
    if not os.path.exists(target_dir):
        logger.warning(f"目标目录不存在: {target_dir}")
        return
    
    logger.info("开始清理目录: {}".format(target_dir))
    logger.info(f"空目录最大保留天数: {folder_max_age_days}")
    logger.info(f"文件最大保留天数: {file_max_age_days}")
    
    # 统计信息
    deleted_dirs = 0
    deleted_files = 0
    
    # 自底向上遍历目录（先处理最深层的目录）
    for root, dirs, files in os.walk(target_dir, topdown=False):
        # 检查文件
        for file in files:
            file_path = os.path.join(root, file)
            if is_older_than_days(file_path, file_max_age_days):
                try:
                    os.remove(file_path)
                    logger.info(f"已删除过期文件: {file_path}")
                    deleted_files += 1
                except Exception as e:
                    logger.error(f"删除文件失败: {file_path}, 错误: {e}")
        
        # 检查当前目录是否为空
        if is_empty_directory(root) and root != target_dir:  # 不删除根目录
            if is_older_than_days(root, folder_max_age_days):
                try:
                    shutil.rmtree(root)
                    logger.info(f"已删除过期空目录: {root}")
                    deleted_dirs += 1
                except Exception as e:
                    logger.error(f"删除目录失败: {root}, 错误: {e}")
    
    logger.info(f"清理完成! 共删除 {deleted_dirs} 个空目录和 {deleted_files} 个文件")
    return deleted_dirs, deleted_files

def main():
    """主函数"""
    start_time = time.time()
    try:
        # 加载配置
        config = load_config()
        target_dir = config['target_dir']
        folder_max_age_days = config['empty_folder_age_days']
        file_max_age_days = config['file_age_days']
        
        # 清理空目录和过期文件
        deleted_dirs, deleted_files = clean_empty_directories(target_dir, folder_max_age_days, file_max_age_days)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"程序执行完成，耗时: {elapsed_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"程序出错: {e}")

if __name__ == "__main__":
    main()
