import torch
import torch.nn as nn
from .backbones import get_backbone

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_size=256, num_layers=2, pretrained=True, use_optical_flow=False):
        """
        CNN-LSTM 模型架構，用於影片事件辨識。
        
        Args:
            num_classes (int): 分類類別數量 (預設 4: Serve, Rally, Winner, Dead)。
            hidden_size (int): LSTM 隱藏層維度。
            num_layers (int): LSTM 層數。
            pretrained (bool): 是否使用預訓練的 Backbone。
            use_optical_flow (bool): 是否使用光流 (False=RGB only 3 channels, True=RGB+Flow 6 channels)。
        """
        super(CNNLSTM, self).__init__()
        
        self.use_optical_flow = use_optical_flow
        
        # 1. 定義 CNN Backbone (ResNet-50)
        # RGB only = 3 channels, RGB + Optical Flow = 6 channels
        input_channels = 6 if use_optical_flow else 3
        self.backbone, self.feature_dim = get_backbone(pretrained=pretrained, input_channels=input_channels)
        
        # 2. 定義 LSTM 層
        # batch_first=True: 輸入格式為 (batch, seq, feature)
        # bidirectional=True: 使用雙向 LSTM 以捕捉前後文資訊
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. 定義分類層 (Classification Head)
        # 由於是雙向 LSTM，輸出維度是 hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        前向傳播 (Forward Pass)
        
        Args:
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, seq_len, channels, height, width)
            
        Returns:
            torch.Tensor: 預測結果 (Logits)，形狀為 (batch_size, num_classes)
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # 1. 調整形狀以輸入 CNN
        # CNN 需要 (N, C, H, W)，所以將 batch 和 seq_len 合併
        c_in = x.view(batch_size * seq_len, C, H, W)
        
        # 2. 提取特徵 (CNN Feature Extraction)
        # 輸出形狀: (batch_size * seq_len, feature_dim, 1, 1) -> squeeze -> (batch_size * seq_len, feature_dim)
        c_out = self.backbone(c_in)
        c_out = c_out.view(batch_size * seq_len, -1)
        
        # 3. 調整形狀以輸入 LSTM
        # 還原為 (batch_size, seq_len, feature_dim)
        r_in = c_out.view(batch_size, seq_len, -1)
        
        # 4. 時序建模 (LSTM Temporal Modeling)
        # r_out 形狀: (batch_size, seq_len, hidden_size * 2)
        # h_n, c_n 為最後的 hidden state
        r_out, (h_n, c_n) = self.lstm(r_in)
        
        # 5. 分類 (Classification)
        # 取最後一個時間點的輸出作為整個序列的代表
        # 或者可以使用 Attention 機制，這裡先採用最簡單的 Last Step
        last_output = r_out[:, -1, :] 
        
        out = self.fc(last_output)
        
        return out

