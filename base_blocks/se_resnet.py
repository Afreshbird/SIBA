import torch
import torch.nn as nn
from base_blocks.SE import se_module as SEBlock

'''
SE-ResNet block
'''
class Res_SE(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1, reduction=16, use_res=True, use_se=True):
        super(Res_SE, self).__init__()
        self.use_res = use_res
        if in_channel != out_channel:
            self.res_conv = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        else:
            self.res_conv = nn.Identity()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, stride, padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel, stride, padding)
        self.act = nn.PReLU()
        if use_se:
            self.se = SEBlock(out_channel,reduction)
        else:
            self.se = nn.Identity()


    def forward(self, input): 
        res = self.res_conv(input)

        out = self.act(self.conv1(input))
        out = self.conv2(out)
        out = self.se(out)

        if self.use_res:
            out = self.act(out+res)
        else:
            out = self.act(out)
        return out



if __name__ == '__main__':
    data = torch.randn(1,32,128,128)
    model = Res_SE(32,32)
    out = model(data)
    print(out.shape)
