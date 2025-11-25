import argparse
import torch
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from models.cnn_lstm import CNNLSTM
from data.dataset import BadmintonDataset
from utils.metrics import calculate_metrics

def evaluate(model_path, test_data_dir, config_path, output_dir="results"):
    """
    評估模型並生成報告。
    """
    # 1. 載入設定
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. 載入資料
    test_dataset = BadmintonDataset(
        data_dir=test_data_dir,
        transform=None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    print(f"Test set size: {len(test_dataset)}")

    # 3. 載入模型
    model = CNNLSTM(
        num_classes=4,
        hidden_size=config['model']['hidden_size'],
        pretrained=False # 評估時不需要下載 ImageNet 權重，因為會載入 checkpoint
    ).to(device)
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 4. 推論 (Inference)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch[0].to(device), batch[-1].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. 計算指標
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    
    # 6. 詳細報告 (Classification Report)
    target_names = ['Serve', 'Rally', 'Winner', 'Dead'] # 需與 Dataset 定義一致
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\n=== Classification Report ===")
    print(report)
    
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # 7. 混淆矩陣 (Confusion Matrix)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    print(f"\nConfusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()
    
    evaluate(args.model_path, args.test_data, args.config, args.output_dir)

