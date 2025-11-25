#!/usr/bin/env python3
"""
影片預處理腳本 - RGB Only 版本
將 data/raw/ 中的影片轉換為訓練用的 frame 序列
"""

import os
import cv2
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def extract_frames(video_path, output_dir, target_size=(224, 224), max_frames=None):
    """
    從影片中提取 frames
    
    Args:
        video_path: 影片檔案路徑
        output_dir: 輸出目錄
        target_size: 目標影像大小 (width, height)
        max_frames: 最大提取幀數，None 表示全部提取
    
    Returns:
        int: 成功提取的幀數
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Cannot open video {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame_resized = cv2.resize(frame, target_size)
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame_resized)
        
        saved_count += 1
        frame_count += 1
        
        if max_frames and saved_count >= max_frames:
            break
    
    cap.release()
    return saved_count

def process_dataset(raw_dir, output_dir, split_ratio=(0.7, 0.15, 0.15), target_size=(224, 224)):
    """
    處理整個資料集
    
    Args:
        raw_dir: 原始資料目錄，如 data/raw/tennis
        output_dir: 輸出目錄，如 data/processed/tennis
        split_ratio: (train, val, test) 分割比例
        target_size: 目標影像大小
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    # 檢查資料夾是否存在
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    # 支援的影片格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    # 掃描所有類別資料夾
    categories = [d for d in raw_path.iterdir() if d.is_dir()]
    
    if len(categories) == 0:
        raise ValueError(f"No category folders found in {raw_dir}")
    
    print(f"Found {len(categories)} categories: {[c.name for c in categories]}")
    
    # 為每個類別收集影片並分割
    for category in categories:
        category_name = category.name
        print(f"\n Processing category: {category_name}")
        
        # 收集該類別的所有影片
        videos = []
        for ext in video_extensions:
            videos.extend(list(category.glob(f"*{ext}")))
        
        if len(videos) == 0:
            print(f"  Warning: No videos found in {category}")
            continue
        
        print(f"  Found {len(videos)} videos")
        
        # 隨機打亂並分割
        random.shuffle(videos)
        n_total = len(videos)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        
        train_videos = videos[:n_train]
        val_videos = videos[n_train:n_train + n_val]
        test_videos = videos[n_train + n_val:]
        
        print(f"  Split: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
        
        # 處理每個分割
        for split_name, video_list in [('train', train_videos), 
                                        ('val', val_videos), 
                                        ('test', test_videos)]:
            if len(video_list) == 0:
                continue
            
            for video_path in tqdm(video_list, desc=f"  {split_name}", ncols=80):
                video_name = video_path.stem
                
                # 輸出目錄: data/processed/tennis/train/flat_service/video_001/
                video_output_dir = output_path / split_name / category_name / video_name
                
                # 提取 frames
                n_frames = extract_frames(video_path, video_output_dir, target_size)
                
                if n_frames == 0:
                    print(f"    Warning: No frames extracted from {video_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess videos to frames (RGB only)")
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Raw video directory, e.g., data/raw/tennis')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory, e.g., data/processed/tennis')
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Train/Val/Test split ratio (default: 0.7 0.15 0.15)')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                        help='Target image size (width height), default: 224 224')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 設定隨機種子
    random.seed(args.seed)
    
    # 檢查分割比例
    if abs(sum(args.split_ratio) - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(args.split_ratio)}")
    
    print("="*60)
    print("Video Preprocessing (RGB Only)")
    print("="*60)
    print(f"Raw directory:    {args.raw_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratio:      Train={args.split_ratio[0]:.1%}, Val={args.split_ratio[1]:.1%}, Test={args.split_ratio[2]:.1%}")
    print(f"Target size:      {args.target_size[0]}x{args.target_size[1]}")
    print(f"Random seed:      {args.seed}")
    print("="*60)
    
    # 執行處理
    process_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        split_ratio=tuple(args.split_ratio),
        target_size=tuple(args.target_size)
    )
    
    print("\n" + "="*60)
    print("✓ Preprocessing completed!")
    print("="*60)

if __name__ == "__main__":
    main()
