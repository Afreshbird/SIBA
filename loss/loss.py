import kornia.losses
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
From CHITNet: A Complementary to Harmonious Information Transfer Network for Infrared and Visible Image Fusion
'''
class JointGrad(nn.Module):
    def __init__(self):
        super(JointGrad, self).__init__()
        self.laplacian = kornia.filters.laplacian
        self.l1_loss = nn.L1Loss()

    def forward(self, im_fus, im_ir, im_vi):
        ir_grad = torch.abs(self.laplacian(im_ir, 3))
        vi_grad = torch.abs(self.laplacian(im_vi, 3))
        fus_grad = self.laplacian(im_fus, 3)
        JGrad = torch.where(ir_grad-vi_grad >= 0, self.laplacian(im_ir, 3), self.laplacian(im_vi, 3))
        loss_JGrad = self.l1_loss(JGrad, fus_grad)
        return loss_JGrad


'''
From CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion
'''
class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        
    def forward(self,generate_img,image_ir,image_vis):
        
        # int calculation
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)

        # grad calculation
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        
        return loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self,x):

        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)