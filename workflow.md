# 專案執行流程 (Workflow)

本文件詳細說明從資料收集到模型評估的完整執行步驟。

## 階段一：資料準備 (Data Preparation)

### 1. 資料收集 (Data Collection)
- [ ] **網球 (Source Domain)**:
    - 下載 THETIS 或 OpenTTGames 資料集。
    - 或從 YouTube 下載網球賽事影片 (建議 1080p, 30fps+)。
    - 存放至 `data/raw/tennis/`。
- [ ] **羽球 (Target Domain)**:
    - 從 YouTube BWF 頻道下載賽事精華 (Highlights)。
    - 存放至 `data/raw/badminton/`。

### 2. 資料標註 (Annotation)
- [ ] 使用標註工具 (如 LabelImg, CVAT 或自製簡單腳本) 標記事件時間點 (Start Time, End Time, Event Class)。
- [ ] 輸出格式建議為 JSON:
    ```json
    {
        "video_name.mp4": [
            {"start": 10.5, "end": 12.0, "label": "serve"},
            {"start": 12.1, "end": 15.0, "label": "rally"}
        ]
    }
    ```
- [ ] 儲存至 `data/annotations/`。

### 3. 預處理 (Preprocessing)
- [ ] **Frame Extraction**: 將影片轉為 Frame 序列 (建議 Resize 至 224x224 或 320x320)。
- [ ] **Optical Flow**: 計算 Dense Optical Flow (使用 TV-L1 或 Farneback 算法)。
- [ ] **Normalization**: 將 RGB 與 Optical Flow 數值正規化至 [0, 1] 或 [-1, 1]。
- [ ] 執行腳本: `python src/data/optical_flow.py`

---

## 階段二：模型建置 (Model Implementation)

### 1. 建置 CNN Backbone
- [ ] 實作 `src/models/backbones.py`。
- [ ] 使用 `torchvision.models.resnet50(pretrained=True)`。
- [ ] 修改第一層 Conv2d 以接受 6-channel 輸入 (RGB + Flow)，或採用雙流架構 (Two-Stream) 後融合。

### 2. 建置 LSTM 模組
- [ ] 實作 `src/models/cnn_lstm.py`。
- [ ] 定義 Bi-directional LSTM 層。
- [ ] 加入 Fully Connected Layer 進行分類 (4 classes)。

### 3. 定義 Dataset 與 DataLoader
- [ ] 實作 `src/data/dataset.py`。
- [ ] 支援 Sliding Window 機制，將長影片切分為固定長度的 Clips (e.g., 16 frames)。
- [ ] 實作 Data Augmentation (Random Crop, Horizontal Flip)。

---

## 階段三：訓練與遷移學習 (Training & Transfer Learning)

### 1. 實驗 A: Baseline (Badminton Only)
- [ ] 設定 `configs/experiments/baseline_badminton.yaml`。
- [ ] 僅使用羽球資料集進行訓練。
- [ ] 記錄 Loss 與 Accuracy 曲線。

### 2. 實驗 B: Pre-training (Tennis)
- [ ] 設定 `configs/experiments/pretrain_tennis.yaml`。
- [ ] 使用網球資料集訓練模型，直到收斂。
- [ ] 儲存權重至 `weights/tennis_best.pth`。

### 3. 實驗 C: Transfer Learning (Tennis -> Badminton)
- [ ] **Strategy 1 (Frozen)**: 載入網球權重，凍結 CNN 層，僅訓練 LSTM 與 FC 層。
- [ ] **Strategy 2 (Fine-tuning)**: 載入網球權重，解凍所有層，使用較小 LR (e.g., 1e-5) 微調。
- [ ] 比較不同策略的收斂速度與最終準確率。

---

## 階段四：評估與視覺化 (Evaluation & Visualization)

### 1. 定量評估
- [ ] 計算 Test Set 的 Accuracy, Precision, Recall, F1-Score。
- [ ] 繪製 Confusion Matrix 以分析各類別的混淆情況 (例如：Serve 容易被誤判為 Dead?)。

### 2. 定性評估 (Grad-CAM)
- [ ] 實作 `src/utils/visualization.py`。
- [ ] 選取關鍵 Frame，繪製 Grad-CAM 熱力圖。
- [ ] 驗證模型關注區域是否為「球」或「球員動作」。

### 3. 網頁展示 (Optional)
- [ ] 使用 Streamlit 或 Flask 建立簡單 Demo 頁面。
- [ ] 上傳影片，即時顯示辨識結果與信心分數。
