# 深度期末 跨運動事件辨識：從網球到羽球——基於 CNN-LSTM 的遷移學習研究

**CSE544 Final Group Project — Proposal**

**組員與學號：**
* **M144020038** 謝睿恩
* **M144020057** 楊翊愷

---

## 一、問題定義與研究構想

### 1. 研究背景
體育賽事的自動化標記能顯著提升轉播與戰術分析效率。Wang 等人（2025）提出的 **CNN-LSTM 融合架構** 已證實能有效結合「空間特徵」與「時序動態」，在足球賽事（SoccerNet）中達到 **92.3%** 的分類準確率 。

然而，針對**球拍類運動（Racket Sports）**（如網球、羽球），其物件（球）極小且移動速度極快（羽球殺球可達 400km/h），與足球的視覺特徵差異巨大。此外，為每一種球類運動重新蒐集大規模標註資料成本極高。

### 2. 研究目的
本計畫旨在驗證 CNN-LSTM 架構在高速球拍運動中的適用性，並探討**遷移學習（Transfer Learning）** 技術是否能利用資料較豐富的「網球」模型，來提升資料較稀缺的「羽球」事件辨識效能。

### 3. 研究問題 (Research Questions)
1.  **架構適應性：** 結合光流法（Optical Flow）的 CNN-LSTM 架構，是否足以捕捉羽球/網球這類高速微小物件的運動軌跡？
2.  **遷移效益：** 透過網球數據預訓練（Pre-training）的模型，在遷移至羽球領域時，相較於從頭訓練（From Scratch），能否在小樣本下達到更高的收斂速度與準確率？
3.  **可解釋性：** 模型在判斷「殺球」或「界外」時，是關注球員的肢體動作（Pose）還是球的落點？

---

## 二、方法概要

本研究將復現並改良 Wang 等人（2025）的架構，並設計遷移學習實驗。

### 1. 模型架構 (Base Model)
我們將構建如下模型：

* **輸入層 (Input)：** 堆疊 RGB 影格與 Dense Optical Flow（光流），形成 **6-channel** 輸入，以強化動態捕捉 。
* **空間特徵 (Spatial)：** 使用 **ResNet-50** 作為 Backbone 提取每一幀的特徵向量 。
* **時序建模 (Temporal)：** 將特徵序列輸入 **Bi-directional LSTM**，捕捉擊球動作的前後因果關係 。

**針對球拍運動的調整：**
* 調整 Input Size 與 Crop 策略，移除觀眾席雜訊，專注於場地內。
* 縮短 Sliding Window 的時間跨度（例如 0.5s - 1s），以適應羽球快速的攻防節奏。

### 2. 實驗設計：遷移學習策略
我們將比較三種訓練策略在羽球測試集上的表現：
* **Baseline (Scratch)：** 僅使用羽球資料，從隨機初始化開始訓練。
* **Strategy A (Frozen Feature)：** 載入「網球」預訓練權重，凍結 CNN Backbone，僅訓練 LSTM 與分類層。
* **Strategy B (Fine-tuning)：** 載入「網球」預訓練權重，並以較小的 Learning Rate 微調全模型。

### 3. 事件定義
定義通用的球拍運動事件類別：
* Serve (發球)
* mash (殺球/得分)
* Rally (對打/過渡)
* Defense/Receive (防守/接發)

---

## 三、資料來源與取得

### 1. 來源領域 (Source Domain) - 網球
* **來源：** 使用公開資料集（如 **THETIS** 或 **OpenTTGames** 的網球部分），或 YouTube 上的完整網球賽事影片。
* **目標：** 取得約 500-1000 個 Clip 進行預訓練。

### 2. 目標領域 (Target Domain) - 羽球
* **來源：** YouTube BWF 官方頻道的賽事精華（Highlights）。
* **優勢：** Highlights 多為得分鏡頭，只需剪輯並標註具體事件類型，可快速建立高品質的小型資料集（約 50-100 個 Clip）。

---

## 四、評估指標與風險備援

### 1. 評估指標
* **Quantitative:** Accuracy, F1-Score, Confusion Matrix 。
* **Qualitative:** 使用 **Grad-CAM** 繪製熱力圖，視覺化模型在時間與空間上的關注點（如：確認模型是否真的在看「球」）。

### 2. 風險備援
* **風險：** 網球與羽球的特徵差異過大（如場地顏色、球的物理反彈），導致遷移效果不佳。
* **備援方案：** 若遷移失敗，則退回至**「單一運動優化」**。專注於調整 CNN-LSTM 架構（如引入 Attention 機制），使其在純羽球資料集上達到最佳效能，證明此論文架構在高速小球運動的有效性。

---

## 五、關鍵參考文獻

1.  **Wang, Y. (2025).** *Research on Match Event Recognition Method Based on LSTM and CNN Fusion*. 2025 5th International Conference on Automation Control, Algorithm and Intelligent Bionics (ACAIB). 
2.  *(其他網球/羽球相關資料集與 TrackNet 相關文獻將於報告中補上)*

---

## 六、組員分工與背景關聯

* **楊翊愷 (M144020057)：**
    * **背景：** 熟悉 Deep Learning、PyTorch、Docker 部署。
    * **職責：** 建置 CNN-LSTM 模型、實作光流法預處理 、執行遷移學習訓練實驗。
* **謝睿恩 (M144020038)：**
    * **背景：** 資料處理與分析。
    * **職責：** 網球/羽球影片蒐集與剪輯、資料標註 (Ground Truth Generation)、Grad-CAM 視覺化實作與網頁 Demo 介面展示。