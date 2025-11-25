#!/bin/bash
# THETIS 資料集下載腳本
# 自動下載並整理 THETIS 網球資料集

set -e  # 遇到錯誤立即停止

echo "======================================"
echo "THETIS 資料集下載工具"
echo "======================================"

# 檢查參數
DOWNLOAD_DIR="${1:-$HOME/Downloads/thetis_rgb}"

echo "下載目錄: $DOWNLOAD_DIR"
echo ""

# 建立下載目錄
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# 檢查是否已經下載
if [ -d ".git" ]; then
    echo "⚠️  檢測到已存在的 Git 倉庫"
    read -p "是否要重新下載？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消下載"
        exit 0
    fi
    rm -rf .git VIDEO_RGB
fi

echo "🔧 設定 Git sparse checkout..."
git init
git config core.sparseCheckout true
echo "VIDEO_RGB" >> .git/info/sparse-checkout

echo "📡 連接到 THETIS 儲存庫..."
git remote add origin https://github.com/THETIS-dataset/dataset.git

echo "⬇️  下載 VIDEO_RGB 資料（這可能需要幾分鐘）..."
git pull origin main

echo ""
echo "✅ 下載完成！"
echo ""
echo "📂 影片位置: $DOWNLOAD_DIR/VIDEO_RGB/"
echo ""
echo "📋 下一步："
echo "1. 檢查下載的影片檔案"
echo "2. 參考 THETIS 的標註檔，將影片按動作類型分類"
echo "3. 複製到專案的 data/raw/tennis/ 對應類別資料夾"
echo ""
echo "專案支援的類別："
echo "  - flat_service    (平擊發球)"
echo "  - slice_service   (切削發球)"
echo "  - smash          (扣殺/殺球)"
echo "  - forehand_flat  (正手平擊)"
echo "  - backhand       (反手擊球)"
echo "  - forehand_volley (正手截擊)"
echo "  - backhand_volley (反手截擊)"
echo ""
echo "範例整理指令："
echo "  cp $DOWNLOAD_DIR/VIDEO_RGB/[某個發球影片].mp4 <專案路徑>/data/raw/tennis/flat_service/"
