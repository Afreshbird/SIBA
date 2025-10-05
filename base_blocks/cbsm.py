import torch.nn as nn
from base_blocks.SE import se_module as SEBlock

'''
channel boosting and space mapping module (CBSM)
'''
class CBSM(nn.Module):
    def __init__(self, boosting_channel, reduction=16):
        super(CBSM, self).__init__()
        self.conv1 = nn.Conv2d(1, boosting_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(boosting_channel, boosting_channel, 3, 1, 1)
        self.act = nn.PReLU()
        self.se = SEBlock(boosting_channel, reduction)

    def forward(self, source_image): 
        tmp = self.conv2(self.act(self.conv1(source_image)))
        return self.act(self.se(tmp))
