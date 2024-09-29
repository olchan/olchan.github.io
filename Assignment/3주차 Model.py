from Backbone import VGG19
from ArcFace import ArcFace
import torch.nn as nn

class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.VGG19 = VGG19()
        self.ArcFace = ArcFace(in_dim = 25088, out_dim = 20, s = 64, m = 0.6)
    
    def forward(self, x):
        x = self.VGG19(x)
        x = self.ArcFace(x)
        
        return x
    

"""
VGG19를 통해 이미지로부터 특징 맵을 추출하고, 그 후 ArcFace를 사용해 최종적으로 얼굴 분류를 수행한다.
"""