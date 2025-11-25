import torch
import torch.nn as nn
import torchvision.models as models

def get_backbone(pretrained=True, input_channels=6):
    """
    取得 ResNet-50 Backbone，並修改第一層卷積以支援 6-channel 輸入 (RGB + Optical Flow)。
    
    Args:
        pretrained (bool): 是否載入 ImageNet 預訓練權重。
        input_channels (int): 輸入影像的通道數 (預設為 6: 3 RGB + 3 Optical Flow)。
        
    Returns:
        nn.Module: 修改後的 ResNet-50 特徵提取器 (移除最後的 FC 層)。
        int: 輸出特徵的維度 (ResNet50 為 2048)。
    """
    # 1. 載入預訓練的 ResNet-50
    resnet = models.resnet50(pretrained=pretrained)
    
    # 2. 修改第一層卷積 (Conv1) 以適應 input_channels
    # 原始 ResNet 的 Conv1 為: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if input_channels != 3:
        original_conv1 = resnet.conv1
        
        # 建立新的 Conv1 層
        new_conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # 初始化權重
        with torch.no_grad():
            # 複製原始 RGB 權重到前 3 個 channel
            new_conv1.weight[:, :3, :, :] = original_conv1.weight
            # 對於額外的 channel (如 Optical Flow)，可以使用原始權重的平均值或隨機初始化
            # 這裡我們使用平均值來保持初始輸出的分佈
            if input_channels > 3:
                new_conv1.weight[:, 3:, :, :] = torch.mean(original_conv1.weight, dim=1, keepdim=True).repeat(1, input_channels - 3, 1, 1)
        
        resnet.conv1 = new_conv1

    # 3. 移除最後的全連接層 (FC) 和分類器，只保留特徵提取部分
    # ResNet 的架構是: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
    # 我們需要保留到 avgpool
    modules = list(resnet.children())[:-1] # 移除最後的 fc 層
    backbone = nn.Sequential(*modules)
    
    feature_dim = resnet.fc.in_features # ResNet50 通常是 2048
    
    return backbone, feature_dim

