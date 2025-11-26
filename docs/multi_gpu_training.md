# 🚀 多GPU訓練指南

## ✅ 已支持多GPU訓練

訓練代碼已添加 **DataParallel** 支持，可以自動使用多張GPU進行訓練。

---

## 📊 使用方式

### 單GPU訓練（預設）

```bash
# 使用 GPU 0
CUDA_VISIBLE_DEVICES=0 python src/train.py --config configs/experiments/tennis_baseline.yaml

# 使用 GPU 1
CUDA_VISIBLE_DEVICES=1 python src/train.py --config configs/experiments/tennis_baseline.yaml
```

### 多GPU訓練 ⭐

```bash
# 使用 GPU 0 和 GPU 1
CUDA_VISIBLE_DEVICES=0,1 python src/train.py \
    --config configs/experiments/tennis_baseline.yaml \
    --experiment_name "tennis_4event_multigpu"

# 使用所有可用的GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py \
    --config configs/experiments/tennis_baseline.yaml

# 後台訓練（多GPU）
CUDA_VISIBLE_DEVICES=0,1 nohup python src/train.py \
    --config configs/experiments/tennis_baseline.yaml \
    --experiment_name "tennis_4event_2gpu" \
    > training_2gpu.log 2>&1 &
```

---

## 🔍 訓練時的輸出

### 單GPU
```
Using device: cuda
Train set: 7775 samples
...
```

### 多GPU
```
Using device: cuda
Using 2 GPUs for training!
GPU IDs: [0, 1]
Train set: 7775 samples
...
```

---

## ⚙️ 工作原理

當檢測到多張GPU時：
1. 模型自動使用 `nn.DataParallel` 包裝
2. 每個 batch 自動分割到不同GPU
3. 前向傳播在所有GPU並行執行
4. 梯度在 GPU 0 聚合
5. 反向傳播更新參數

---

## 📈 性能提升

| GPU數量 | 相對速度 | 有效Batch Size | 建議 |
|---------|---------|---------------|------|
| 1 GPU | 1x | 32 | 基準 |
| 2 GPU | ~1.7x | 64 (32×2) | ✅ 推薦 |
| 4 GPU | ~3.0x | 128 (32×4) | 適合大數據集 |

**注意**: 加速比不是線性的，因為有通信開銷。

---

## 💡 最佳實踐

### 1. Batch Size 調整
多GPU訓練時，**有效 batch size = 單卡 batch size × GPU 數量**

```yaml
# 配置文件 (tennis_baseline.yaml)
training:
  batch_size: 32  # 2 GPU → 有效 batch_size = 64
```

如果想保持相同的有效 batch size：
```yaml
# 單GPU
batch_size: 64

# 2 GPU
batch_size: 32  # 有效 batch_size = 32 × 2 = 64
```

### 2. Learning Rate 調整（可選）
有效 batch size 增大時，可以考慮增加學習率：

```yaml
# 單GPU (batch_size=32)
learning_rate: 0.0001

# 2 GPU (有效batch_size=64)
learning_rate: 0.0002  # 或保持 0.0001
```

### 3. 監控GPU使用

```bash
# 實時監控
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
watch -n 1 gpustat
```

---

## 🐛 疑難排解

### Q: 為什麼沒有使用多GPU？

**檢查**：
```bash
# 確認CUDA可見的GPU
echo $CUDA_VISIBLE_DEVICES

# Python中檢查
python -c "import torch; print(torch.cuda.device_count())"
```

**解決**：確保環境變數設定正確
```bash
export CUDA_VISIBLE_DEVICES=0,1
python src/train.py --config ...
```

### Q: GPU記憶體不均衡（GPU 0負載太重）

這是 **DataParallel 的已知限制**，GPU 0會:
- 收集所有梯度
- 執行參數更新
- 儲存完整模型

**解決方案**：
1. 可以接受（小模型影響不大）
2. 改用 DistributedDataParallel（需要更多代碼修改）

### Q: 訓練變慢了？

可能原因：
1. **Batch size 太小** - 通信開銷大於計算
   - 建議：每張卡至少 batch_size ≥ 16
2. **數據載入瓶頸** - 增加 `num_workers`
   ```yaml
   training:
     num_workers: 8  # 增加到 8 或更多
   ```

### Q: 載入舊模型報錯？

代碼已經處理 `module.` 前綴問題，應該能正常載入：
- 單GPU訓練的模型 → 多GPU使用 ✅
- 多GPU訓練的模型 → 單GPU使用 ✅

---

## 🎯 推薦配置

### 實驗室伺服器（2-4 GPU）

```bash
# 使用2張GPU，後台訓練
CUDA_VISIBLE_DEVICES=0,1 nohup python src/train.py \
    --config configs/experiments/tennis_baseline.yaml \
    --experiment_name "tennis_4event_2gpu" \
    > training_2gpu.log 2>&1 &

# 監控訓練
tail -f training_2gpu.log
watch -n 1 nvidia-smi
```

### Colab（單GPU T4）

Colab 通常只有1張GPU，無需特別設定：
```python
!python src/train.py --config configs/experiments/tennis_colab.yaml
```

---

## 📝 注意事項

1. **DataParallel vs DistributedDataParallel**
   - 當前使用：DataParallel（簡單，單機多卡）
   - 未來可升級：DistributedDataParallel（高效，支持多機）

2. **模型保存**
   - 已自動處理：保存的是原始模型（無 `module.` 前綴）
   - 可以在單GPU/多GPU間自由切換

3. **Batch Size**
   - 有效batch size會變大，可能影響收斂
   - 建議先用預設值，觀察效果

4. **記憶體**
   - 每張GPU需要足夠記憶體
   - 如果OOM，降低 `batch_size`

---

祝訓練順利！🚀
