import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

class GradCAM:
    """
    Grad-CAM 實作，用於視覺化 CNN 關注區域。
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # 註冊 Hook
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        """
        生成 Heatmap。
        x: 輸入張量 (batch, seq, c, h, w) -> 這裡簡化為處理單張影像或將 seq 視為 batch
        """
        # 確保模型在 eval 模式，但需要梯度
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activation
        
        # Global Average Pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.data.cpu().numpy()

def plot_grad_cam(frame, heatmap, alpha=0.5, save_path=None):
    """
    將 Heatmap 疊加在原始 Frame 上。
    frame: 原始影像 (H, W, 3) numpy array, range [0, 255]
    heatmap: Grad-CAM 輸出 (1, 1, H', W')
    """
    # Resize heatmap to frame size
    heatmap = cv2.resize(heatmap[0, 0], (frame.shape[1], frame.shape[0]))
    
    # Apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose
    superimposed_img = heatmap * alpha + frame * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    if save_path:
        cv2.imwrite(save_path, superimposed_img)
        
    return superimposed_img

