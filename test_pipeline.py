#!/usr/bin/env python3
"""
快速測試腳本 - 用於驗證訓練流程
使用小樣本測試所有組件是否正常運作
"""

import sys
import os
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent))

def test_preprocessing():
    """測試預處理是否正常"""
    print("="*60)
    print("測試 1: 預處理腳本")
    print("="*60)
    
    try:
        from src.data.preprocess_videos import extract_frames
        print("✓ 預處理模組載入成功")
        return True
    except Exception as e:
        print(f"✗ 預處理模組載入失敗: {e}")
        return False

def test_dataset():
    """測試 Dataset 是否正常"""
    print("\n" + "="*60)
    print("測試 2: Dataset 類別")
    print("="*60)
    
    try:
        from src.data.dataset import VideoEventDataset
        print("✓ Dataset 模組載入成功")
        
        # 檢查是否有預處理資料
        test_dir = Path("data/processed/tennis/train")
        if test_dir.exists():
            try:
                dataset = VideoEventDataset(test_dir, seq_length=8)
                print(f"✓ Dataset 初始化成功")
                print(f"  - 樣本數: {len(dataset)}")
                print(f"  - 類別數: {len(dataset.class_to_idx)}")
                print(f"  - 類別: {list(dataset.class_to_idx.keys())}")
                
                if len(dataset) > 0:
                    frames, label = dataset[0]
                    print(f"  - 樣本形狀: {frames.shape}")
                    print(f"  - 標籤: {label}")
                
                return True
            except Exception as e:
                print(f"✗ Dataset 初始化失敗: {e}")
                return False
        else:
            print(f"⚠ 預處理資料不存在: {test_dir}")
            print("  請先運行預處理腳本")
            return True  # 不算失敗
            
    except Exception as e:
        print(f"✗ Dataset 模組載入失敗: {e}")
        return False

def test_model():
    """測試模型是否正常"""
    print("\n" + "="*60)
    print("測試 3: 模型架構")
    print("="*60)
    
    try:
        import torch
        from src.models.cnn_lstm import CNNLSTM
        
        print("✓ 模型模組載入成功")
        
        # 測試 RGB only 模式
        model = CNNLSTM(num_classes=7, hidden_size=256, use_optical_flow=False)
        print("✓ RGB-only 模型初始化成功")
        
        # 測試前向傳播
        dummy_input = torch.randn(2, 8, 3, 224, 224)  # (batch, seq, C, H, W)
        output = model(dummy_input)
        print(f"✓ 前向傳播成功")
        print(f"  - 輸入形狀: {dummy_input.shape}")
        print(f"  - 輸出形狀: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script():
    """測試訓練腳本導入"""
    print("\n" + "="*60)
    print("測試 4: 訓練腳本")
    print("="*60)
    
    try:
        from src.train import train
        print("✓ 訓練腳本載入成功")
        return True
    except Exception as e:
        print(f"✗ 訓練腳本載入失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """測試配置文件"""
    print("\n" + "="*60)
    print("測試 5: 配置文件")
    print("="*60)
    
    try:
        import yaml
        config_path = Path("configs/experiments/tennis_baseline.yaml")
        
        if not config_path.exists():
            print(f"✗ 配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ 配置文件讀取成功")
        print(f"  - 類別數: {config['data']['num_classes']}")
        print(f"  - Batch size: {config['training']['batch_size']}")
        print(f"  - 序列長度: {config['model']['seq_length']}")
        print(f"  - 使用光流: {config['model']['use_optical_flow']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置文件測試失敗: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("🧪 訓練流程測試")
    print("="*60)
    
    results = []
    
    results.append(("預處理模組", test_preprocessing()))
    results.append(("Dataset 類別", test_dataset()))
    results.append(("模型架構", test_model()))
    results.append(("訓練腳本", test_training_script()))
    results.append(("配置文件", test_config()))
    
    print("\n" + "="*60)
    print("📊 測試結果")
    print("="*60)
    
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✅ 所有測試通過！可以開始訓練了")
        print("\n下一步:")
        print("1. 將影片放入 data/raw/tennis/ 的各個類別資料夾")
        print("2. 運行預處理: python src/data/preprocess_videos.py --raw_dir data/raw/tennis --output_dir data/processed/tennis")
        print("3. 開始訓練: python src/train.py --config configs/experiments/tennis_baseline.yaml")
    else:
        print("\n❌ 部分測試失敗，請檢查錯誤訊息")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
