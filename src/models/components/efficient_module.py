import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lightning as L

class EfficientNetModule(nn.Module):
    def __init__(self, weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1, num_classes=1000):
        super(EfficientNetModule, self).__init__()
        self.model = models.efficientnet_b0(weights=weights)
        
    def forward(self, x):
        features = self.forward_features(x)
        logists = self.forward_head(features)
        return logists, features

    def forward_features(self, x):
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        return x
    
    def forward_head(self, x):
        x = self.model.classifier(x)
        return x# touch 520
