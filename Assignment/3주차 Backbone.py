import torch.nn as nn

'''
VGG19 모델은 얼굴 인식의 초기 단계에서 입력 이미지에서 중요한 시각적 특징을 추출하여 이후의 ArcFace 모델에 전달될 특징 맵을 생성한다.
입력된 이미지에서 저차원 특징을 고차원 특징으로 변환해준다.
'''

'''
Class VGG19
    1. def __init__(self, base_dim = 64):
        1) Hyper parameter
        - base_dim is derived from "VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION."
        - The architecture of this class is partially based on VGG19, with the notable exception that it does not include fully connected layers.
    
    2. def forward(self, x):
        - The forward pass of the VGG19 model.
        - EXAMPLE ) input : 224 x 224 x 3 -> output :  7 x 7 x 512
        
'''


def conv_2(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_4(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

class VGG19(nn.Module):
    def __init__(self, base_dim=64):
        super(VGG19, self).__init__()
        self.feature = nn.Sequential(
        conv_2(3, base_dim),
        conv_2(base_dim, base_dim*2),
        conv_4(base_dim*2, base_dim*4),
        conv_4(base_dim*4, base_dim*8),
        conv_4(base_dim*8, base_dim*8)
        )
        
    def forward(self, x):
        x = self.feature(x)
        
        return x