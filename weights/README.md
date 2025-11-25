# Weights 目錄說明

這個目錄用於儲存訓練好的模型權重。

## 目錄結構

```
weights/
├── experiments/        # 實驗模型（按實驗名稱和時間）
│   ├── tennis_7class_20251125_104500/
│   │   ├── best_model.pth
│   │   ├── config.yaml
│   │   └── training_log.txt
│   └── badminton_transfer_20251126_153000/
│       └── ...
├── checkpoints/        # 訓練中的 checkpoints（每 N epoch）
│   └── tennis_epoch_10.pth
└── final/             # 最終選定的模型
    ├── tennis_final.pth
    └── badminton_final.pth
```

## 命名規範

### 實驗模型
格式：`{dataset}_{strategy}_{timestamp}/`
- `dataset`: tennis, badminton
- `strategy`: 7class, 4class, baseline, transfer, finetune
- `timestamp`: YYYYMMDD_HHMMSS

範例：
- `tennis_7class_20251125_104500/`
- `badminton_transfer_frozen_20251125_110000/`

### Checkpoint
格式：`{experiment_name}_epoch_{N}.pth`

範例：
- `tennis_7class_epoch_10.pth`
- `tennis_7class_epoch_20.pth`

### 最終模型
格式：`{purpose}_final.pth`

範例：
- `tennis_final.pth` - 最佳的網球預訓練模型
- `badminton_final.pth` - 最佳的羽球模型

## 使用方式

### 訓練時指定實驗名稱
```bash
python3 src/train.py \
    --config configs/experiments/tennis_baseline.yaml \
    --experiment_name tennis_7class  # 會自動加上時間戳
```

### 載入特定模型
```bash
python3 src/evaluate.py \
    --model_path weights/experiments/tennis_7class_20251125_104500/best_model.pth
```

### 遷移學習時載入
```yaml
# configs/experiments/badminton_transfer.yaml
training:
  pretrained_weights: "weights/final/tennis_final.pth"
```

## .gitignore

模型權重檔案不應上傳到 Git：
```
weights/**/*.pth
!weights/.gitkeep
```

只保留目錄結構和說明文件。
