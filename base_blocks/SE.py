from torch import nn
import torch

'''
Squeeze-and-Excitation Networks
'''
class se_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(se_module, self).__init__()
        assert channel >= reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x): 
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)        
        y = y.view(b, c, 1, 1)
        return x * y
    

if __name__ == '__main__':
    temp = torch.randn(1,32,64,64)
    model = se_module(32)
    out = model(temp)
    print(out.shape)