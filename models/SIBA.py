import torch
import torch.nn as nn
from base_blocks.se_resnet import Res_SE
from base_blocks.cbsm import CBSM
from base_blocks.restormer import TransformerBlock_SA as SA
from base_blocks.restormer import TransformerBlock_CA as CA
from timm.models.layers import trunc_normal_


# the Source Image is the Best Attention for infrared and visible image fusion
class SIBA(nn.Module):
    def __init__(self,in_cha=1,mid_cha=48,out_cha=1,SA_depths=1,CA_depths=1):
        
        super(SIBA, self).__init__()

        self.ir_conv = Res_SE(in_cha, mid_cha)
        self.vi_conv = Res_SE(in_cha, mid_cha)

        self.ir_sa = nn.ModuleList()
        self.vi_sa = nn.ModuleList()
        for i in range(SA_depths):
            self.ir_sa.append(SA(mid_cha,mid_cha))
            self.vi_sa.append(SA(mid_cha,mid_cha))


        self.weight_ir = CBSM(mid_cha)
        self.weight_irI = CBSM(mid_cha)
        self.weight_vi = CBSM(mid_cha)
        self.weight_viI = CBSM(mid_cha)


        self.ir2vi_ca = nn.ModuleList()
        self.irI2vi_ca = nn.ModuleList()
        self.vi2ir_ca = nn.ModuleList()
        self.viI2ir_ca = nn.ModuleList()
        for i in range(CA_depths):
            self.ir2vi_ca.append(CA(mid_cha,mid_cha))
            self.irI2vi_ca.append(CA(mid_cha,mid_cha))
            self.vi2ir_ca.append(CA(mid_cha,mid_cha))
            self.viI2ir_ca.append(CA(mid_cha,mid_cha))


        self.fuse_conv = nn.Sequential(Res_SE(mid_cha*4,mid_cha*2),
                                        Res_SE(mid_cha*2,mid_cha))
        
        self.out_conv = Res_SE(mid_cha,out_cha,use_se=False)

        self.apply(self._init_weights)
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, ir, vi):
        ir_raw = ir
        ir_raw_invert = 1-ir
        vi_raw = vi
        vi_raw_invert = 1-vi

        ir = self.ir_conv(ir) 
        vi = self.vi_conv(vi)

        # self attention
        ir_sa = ir
        vi_sa = vi
        for layer_ir, layer_vi in zip(self.ir_sa, self.vi_sa):
            ir_sa = layer_ir(ir_sa)
            vi_sa = layer_vi(vi_sa)


        w_ir = self.weight_ir(ir_raw)
        w_irI = self.weight_irI(ir_raw_invert)

        w_vi = self.weight_vi(vi_raw)
        w_viI = self.weight_viI(vi_raw_invert)

        # cross attention
        # w_ir, w_irI = q; vi_sa = k, v
        ir2vi_ca = vi_sa
        irI2vi_ca = vi_sa
        for layer_ir, layer_irI in zip(self.ir2vi_ca, self.irI2vi_ca):
            ir2vi_ca = layer_ir(ir2vi_ca, w_ir)
            irI2vi_ca = layer_irI(irI2vi_ca, w_irI)


        vi2ir_ca = ir_sa
        viI2ir_ca = ir_sa
        for layer_vi, layer_viI in zip(self.vi2ir_ca, self.viI2ir_ca):
            vi2ir_ca = layer_vi(vi2ir_ca, w_vi)
            viI2ir_ca = layer_viI(viI2ir_ca, w_viI)


        mixed = torch.cat([ir2vi_ca,vi2ir_ca,irI2vi_ca,viI2ir_ca],dim=1)
        mixed = self.fuse_conv(mixed)
        out = self.out_conv(mixed)

        return out


if __name__ == '__main__':
    ir = torch.randn(4,1,128,128)
    vi = torch.randn(4,1,128,128)
    model = SIBA()
    out = model(ir,vi)
    print(out.shape)