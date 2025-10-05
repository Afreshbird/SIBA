import sys
sys.path.append('./')
from models.SIBA import SIBA
from loader.train_loader import TrainLoader
import os
import time
import datetime
import torch
import torch.nn as nn
from args.args_SIBA import args
from torch.utils.data import DataLoader
import pathlib
from loss.loss import JointGrad,Fusionloss
torch.backends.cudnn.benchmark = True


# 训练设置
os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu_number
model_save_path = args.model_save_path
num_epochs = args.epochs
lr = args.init_lr
weight_decay = args.weight_decay
batch_size = args.batch_size
clip_grad_norm_value = 0.01
optim_step = args.optim_step
optim_gamma = args.optim_gamma


model = SIBA().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)

# loss
JointGradLoss = JointGrad()
Intensity_Grad = Fusionloss()

ir_path = pathlib.Path(args.ir_path)
vi_path = pathlib.Path(args.vi_path)
data = TrainLoader(ir_path, vi_path, args.patch_size)

trainloader = DataLoader(data, batch_size, True, pin_memory=True)

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
prev_time = time.time()


for epoch in range(num_epochs):
    for i, (ir, vi) in enumerate(trainloader):
        ir, vi = ir.cuda(), vi.cuda()
        model.train()
        optimizer.zero_grad()

        fuse = model(ir, vi) 
        
        # loss_Laplae
        loss_2 = JointGradLoss(fuse, ir, vi)
        # loss_intensity, loss_Sobel
        loss_3, loss_4 = Intensity_Grad(fuse, ir, vi)
        loss = 10*loss_2 + 0.1*loss_3 + loss_4
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        optimizer.step()  
        
        # determine approximate time left
        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if i % 50 == 0:
            print("[Epoch {}/{}] [lr {}] [loss: {}] ETA: {}".format(
                    epoch,
                    num_epochs,
                    optimizer.param_groups[0]['lr'],
                    round(loss.item(),4),
                    time_left
                )
            )
    scheduler.step()  

    if optimizer.param_groups[0]['lr'] < 1e-6:
        optimizer.param_groups[0]['lr'] = 1e-6

model_save_path = os.path.join(model_save_path,timestamp)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

if True:
    checkpoint = {
        'model': model.state_dict(),
    }
    save_path = os.path.join(model_save_path, 'SIBA_epoch'+ str(num_epochs) + '.pth')
    torch.save(checkpoint, save_path)

print('done')
