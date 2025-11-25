#!/usr/bin/env python3
"""
簡化的測試腳本 - 測試模型但不下載預訓練權重
"""

import sys
import os
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent))

def test_model_no_pretrained():
    """測試模型（不載入預訓練權重）"""
    print("="*60)
    print("測試: 模型架構 (不載入預訓練權重)")
    print("="*60)
    
    try:
        import torch
        from src.models.cnn_lstm import CNNLSTM
        
        print("✓ 模型模組載入成功")
        
        # 測試 RGB only 模式 (不載入預訓練)
        model = CNNLSTM(num_classes=7, hidden_size=256, use_optical_flow=False, pretrained=False)
        print("✓ RGB-only 模型初始化成功 (隨機權重)")
        
        # 測試前向傳播
        dummy_input = torch.randn(2, 8, 3, 224, 224)  # (batch, seq, C, H, W)
        output = model(dummy_input)
        print(f"✓ 前向傳播成功")
        print(f"  - 輸入形狀: {dummy_input.shape}")
        print(f"  - 輸出形狀: {output.shape}")
        print(f"  - 參數總數: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    """測試 Dataset"""
    print("\n" + "="*60)
    print("測試: Dataset 類別")
    print("="*60)
    
    try:
        from src.data.dataset import VideoEventDataset
        print("✓ Dataset 模組載入成功")
        
        # 檢查是否有預處理資料
        test_dir = Path("data/processed/tennis/train")
        if test_dir.exists():
            dataset = VideoEventDataset(test_dir, seq_length=8)
            print(f"✓ Dataset 初始化成功")
            print(f"  - 樣本數: {len(dataset)}")
            print(f"  - 類別數: {len(dataset.class_to_idx)}")
            print(f"  - 類別: {list(dataset.class_to_idx.keys())}")
            
            if len(dataset) > 0:
                frames, label = dataset[0]
                print(f"  - 樣本形狀: {frames.shape}")
                print(f"  - 標籤: {label}")
        else:
            print(f"ℹ 預處理資料尚未建立: {test_dir}")
            print("  請先運行預處理腳本")
        
        return True
            
    except Exception as e:
        print(f"✗ Dataset 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("🧪 快速流程測試 (無網路連線)")
    print("="*60)
    
    results = []
    results.append(("模型架構", test_model_no_pretrained()))
    results.append(("Dataset 類別", test_dataset()))
    
    print("\n" + "="*60)
    print("📊 測試結果")
    print("="*60)
    
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✅ 核心組件測試通過！")
        print("\n📝 下一步流程:")
        print("1. 將影片放入 data/raw/tennis/ 的對應資料夾中")
        print("   - flat_service/")
        print("   - slice_service/")
        print("   - smash/")
        print("   - ... 等")
        print("\n2. 運行預處理:")
        print("   python3 src/data/preprocess_videos.py \\")
        print("     --raw_dir data/raw/tennis \\")
        print("     --output_dir data/processed/tennis")
        print("\n3. 開始訓練:")
        print("   python3 src/train.py --config configs/experiments/tennis_baseline.yaml")
    else:
        print("\n❌ 部分測試失敗，請檢查錯誤訊息")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
