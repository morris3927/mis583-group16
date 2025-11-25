#!/usr/bin/env python3
"""
THETIS 資料集處理工具
將 THETIS 資料集的影片按照事件類別整理到專案資料夾
"""

import json
import shutil
from pathlib import Path
import argparse

def load_event_mapping():
    """載入事件映射配置"""
    mapping_path = Path(__file__).parent.parent / "configs" / "event_mapping.json"
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def organize_thetis_videos(thetis_dir, output_dir, dry_run=True):
    """
    整理 THETIS 影片到專案資料夾
    
    Args:
        thetis_dir: THETIS 下載目錄（包含 VIDEO_RGB/）
        output_dir: 輸出目錄（通常是 data/raw/tennis/）
        dry_run: 如果為 True，只顯示會執行的操作，不實際複製
    """
    thetis_path = Path(thetis_dir)
    output_path = Path(output_dir)
    
    # 載入事件映射
    event_mapping = load_event_mapping()
    
    # 取得網球的所有類別
    tennis_categories = []
    for event_type, sports in event_mapping.items():
        tennis_categories.extend(sports['tennis'])
    
    print("="*60)
    print("THETIS 資料集整理工具")
    print("="*60)
    print(f"來源目錄: {thetis_path}")
    print(f"目標目錄: {output_path}")
    print(f"模式: {'預覽模式（不實際複製）' if dry_run else '執行模式'}")
    print("")
    print(f"網球類別（共 {len(tennis_categories)} 個）:")
    for cat in tennis_categories:
        print(f"  - {cat}")
    print("="*60)
    print("")
    
    # 檢查 THETIS VIDEO_RGB 目錄
    video_rgb_dir = thetis_path / "VIDEO_RGB"
    if not video_rgb_dir.exists():
        print(f"❌ 錯誤: 找不到 {video_rgb_dir}")
        print("請確認 THETIS 資料集已正確下載")
        return
    
    # 建立輸出目錄結構
    for category in tennis_categories:
        category_dir = output_path / category
        if not dry_run:
            category_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ 建立目錄: {category_dir}")
    
    print("")
    print("="*60)
    print("⚠️  手動整理步驟說明")
    print("="*60)
    print("")
    print("THETIS 資料集需要手動整理，因為:")
    print("1. 需要參考 THETIS 的標註檔案來識別每個影片的動作類型")
    print("2. THETIS 的類別名稱可能與我們的定義不同")
    print("")
    print("建議步驟:")
    print("")
    print("步驟 1: 查看 THETIS 的標註檔案")
    print(f"  cd {thetis_path}")
    print("  # 尋找 annotations, labels 或類似的檔案")
    print("")
    print("步驟 2: 了解 THETIS 的影片命名和分類方式")
    print("  # 檢查影片檔名是否包含動作類型資訊")
    print("")
    print("步驟 3: 手動或編寫腳本複製影片")
    print("  # 範例: 複製發球影片")
    print(f"  cp {video_rgb_dir}/serve_*.mp4 {output_path}/flat_service/")
    print("")
    print("步驟 4: 驗證整理結果")
    print("  # 確認每個類別資料夾都有影片")
    print(f"  ls -lh {output_path}/*/")
    print("")
    
    # 掃描 VIDEO_RGB 目錄
    print("="*60)
    print("THETIS 影片檔案掃描")
    print("="*60)
    
    video_files = list(video_rgb_dir.glob("*.mp4")) + \
                  list(video_rgb_dir.glob("*.avi")) + \
                  list(video_rgb_dir.glob("*.mov"))
    
    if len(video_files) == 0:
        print("⚠️  未找到影片檔案，可能需要進一步檢查 THETIS 的目錄結構")
    else:
        print(f"找到 {len(video_files)} 個影片檔案")
        print("")
        print("前 10 個影片檔案:")
        for i, video_file in enumerate(video_files[:10], 1):
            print(f"  {i}. {video_file.name}")
        
        if len(video_files) > 10:
            print(f"  ... 還有 {len(video_files) - 10} 個檔案")
    
    print("")
    print("="*60)
    print("下一步")
    print("="*60)
    print("1. 查看 THETIS 文檔了解資料結構")
    print("2. 根據標註檔案將影片分類")
    print("3. 複製/移動影片到對應的類別資料夾")
    print("4. 運行預處理腳本:")
    print("   python3 src/data/preprocess_videos.py \\")
    print("     --raw_dir data/raw/tennis \\")
    print("     --output_dir data/processed/tennis")

def main():
    parser = argparse.ArgumentParser(description="THETIS 資料集整理工具")
    parser.add_argument('--thetis_dir', type=str, 
                        default=str(Path.home() / "Downloads" / "thetis_rgb"),
                        help='THETIS 下載目錄')
    parser.add_argument('--output_dir', type=str,
                        default="data/raw/tennis",
                        help='輸出目錄（專案資料夾）')
    parser.add_argument('--execute', action='store_true',
                        help='實際執行（預設為預覽模式）')
    
    args = parser.parse_args()
    
    organize_thetis_videos(
        thetis_dir=args.thetis_dir,
        output_dir=args.output_dir,
        dry_run=not args.execute
    )

if __name__ == "__main__":
    main()
