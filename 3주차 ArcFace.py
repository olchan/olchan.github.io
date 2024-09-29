import torch
import torch.nn as nn
import torch.nn.functional as F

'''
추출된 특징 맵을 기반으로 얼굴 간의 차이를 각도 기반으로 구분한다.
ArcFace는 VGG19에서 추출한 특징 맵을 입력으로 받아,
정규화 및 각도 계산을 통해 얼굴 인식 분류를 수행한다.
즉, ArcFace는 VGG19에서 얻은 특징 벡터에 대해 분류를 담당한다.
'''

'''
Class ArcFace
    1. def __init__(self, in_dim, out_dim, s, m):
        - s and m are parameters derived from "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".
        - Matrix W:
            1) The matrix W has dimensions in_dim x out_dim.
            2) W is initialized using Xavier initialization.
            3) in_dim: Dimensionality of the tensor resulting from flattening the forward pass of VGG19.
            4) out_dim: Number of classes.
            
    2. def forward(self, x):
        - the forward pass of the ArcFace model.

'''

class ArcFace(nn.Module):
    def __init__(self, in_dim, out_dim, s, m):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))

        nn.init.kaiming_uniform_(self.W)
        
    def forward(self, x):
        normalized_x = F.normalize(x, p=2, dim=1)
        normalized_W = F.normalize(self.W, p=2, dim=0)
    
        cosine = torch.matmul(normalized_x.view(normalized_x.size(0), -1), normalized_W)
        
        # Using torch.clamp() to ensure cosine values are within a safe range,
        # preventing potential NaN losses.
        
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        probability = self.s * torch.cos(theta+self.m)
        
        return probability