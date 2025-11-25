from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    計算評估指標。
    
    Args:
        y_true (torch.Tensor or np.array): 真實標籤。
        y_pred (torch.Tensor or np.array): 預測標籤 (Logits 或 Class Indices)。
        
    Returns:
        dict: 包含 Accuracy, Precision, Recall, F1 的字典。
    """
    # 如果是 Tensor，轉為 numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    # 如果 y_pred 是 logits (2D array)，取 argmax 得到類別索引
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
        
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    return metrics

