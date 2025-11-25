import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np

class VideoEventDataset(Dataset):
    """
    影片事件辨識資料集 - RGB Only 版本
    從預處理後的資料載入 frame 序列
    """
    
    def __init__(self, processed_dir, seq_length=16, transform=None, stride=8):
        """
        Args:
            processed_dir: 處理後的資料目錄，如 data/processed/tennis/train
            seq_length: 序列長度（幀數），預設 16
            transform: 資料增強轉換
            stride: sliding window 的步長，預設 8（50% overlap）
        """
        self.processed_dir = Path(processed_dir)
        self.seq_length = seq_length
        self.stride = stride
        
        # 設定預設 transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # 掃描資料並建立樣本列表
        self.samples = []
        self.class_to_idx = {}
        self._scan_dataset()
        
        print(f"Dataset: {processed_dir}")
        print(f"  Classes: {len(self.class_to_idx)} - {list(self.class_to_idx.keys())}")
        print(f"  Total samples: {len(self.samples)}")
    
    def _scan_dataset(self):
        """
        掃描資料集目錄，建立樣本列表
        結構: processed_dir/category/video_name/frame_xxxx.jpg
        """
        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Processed directory not found: {self.processed_dir}")
        
        # 掃描所有類別資料夾
        category_dirs = sorted([d for d in self.processed_dir.iterdir() if d.is_dir()])
        
        if len(category_dirs) == 0:
            raise ValueError(f"No category directories found in {self.processed_dir}")
        
        # 建立類別索引映射
        for idx, cat_dir in enumerate(category_dirs):
            self.class_to_idx[cat_dir.name] = idx
        
        # 掃描每個類別下的影片
        for cat_dir in category_dirs:
            category_name = cat_dir.name
            category_idx = self.class_to_idx[category_name]
            
            # 掃描該類別下的所有影片資料夾
            video_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()])
            
            for video_dir in video_dirs:
                # 取得該影片的所有 frames
                frame_files = sorted(video_dir.glob("frame_*.jpg"))
                
                if len(frame_files) < self.seq_length:
                    # 影片太短，跳過
                    continue
                
                # 使用 sliding window 建立多個樣本
                num_frames = len(frame_files)
                for start_idx in range(0, num_frames - self.seq_length + 1, self.stride):
                    end_idx = start_idx + self.seq_length
                    sample = {
                        'frames': frame_files[start_idx:end_idx],
                        'label': category_idx,
                        'category': category_name,
                        'video_name': video_dir.name
                    }
                    self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        回傳一個訓練樣本
        
        Returns:
            frames: Tensor of shape (seq_length, 3, H, W)
            label: int
        """
        sample = self.samples[idx]
        frame_paths = sample['frames']
        label = sample['label']
        
        # 載入並轉換所有 frames
        frames = []
        for frame_path in frame_paths:
            # 使用 PIL 載入圖片
            img = Image.open(frame_path).convert('RGB')
            
            # 應用 transform
            if self.transform:
                img = self.transform(img)
            
            frames.append(img)
        
        # Stack 成 (seq_length, C, H, W)
        frames = torch.stack(frames, dim=0)
        
        return frames, label


# 為了向後兼容，保留原有的類別名稱
class BadmintonDataset(VideoEventDataset):
    """向後兼容的別名"""
    pass


def get_dataloaders(data_root, batch_size=8, seq_length=16, num_workers=4):
    """
    便捷函數：建立 train/val/test dataloaders
    
    Args:
        data_root: 資料根目錄，如 data/processed/tennis
        batch_size: batch 大小
        seq_length: 序列長度
        num_workers: DataLoader 的工作執行緒數
    
    Returns:
        dict: {'train': train_loader, 'val': val_loader, 'test': test_loader, 'num_classes': int}
    """
    data_root = Path(data_root)
    
    # 建立資料增強
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 建立 datasets
    dataloaders = {}
    num_classes = None
    
    for split in ['train', 'val', 'test']:
        split_dir = data_root / split
        
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping...")
            continue
        
        transform = train_transform if split == 'train' else val_transform
        
        dataset = VideoEventDataset(
            processed_dir=split_dir,
            seq_length=seq_length,
            transform=transform,
            stride=8 if split == 'train' else seq_length  # test/val 不重疊
        )
        
        if num_classes is None:
            num_classes = len(dataset.class_to_idx)
        
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    
    dataloaders['num_classes'] = num_classes
    dataloaders['class_to_idx'] = dataset.class_to_idx if dataset else {}
    
    return dataloaders
