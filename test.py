import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from loader.test_loader import TestLoader
from utils.RGB2YCrBb import YCrCb2RGB, clamp
from models.SIBA import SIBA
import torch
import time

torch.cuda.set_device(0)

model_path = "checkpoint/SIBA_epoch60.pth"
testdata_path = '/home/ws/datasets/image_fusion/MSRS'
result_save_path = '/home/ws/datasets/image_fusion/MSRS_fusion_result'

if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

model = SIBA().cuda()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])

total = sum([params.nelement() for params in model.parameters()])
print("Number of params: {%.3f M}" % (total / 1e6))
model.eval()


test_dataset = TestLoader(testdata_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

sum_time = 0
with torch.no_grad():
    for _, vis_y_image, cb, cr, ir_image, img_name, _ in tqdm(test_loader, total=len(test_loader)):
        vis_y_image = vis_y_image.cuda()
        cb = cb.cuda()
        cr = cr.cuda()
        ir_image = ir_image.cuda()

        start = time.time()
        image_fused = model(ir_image,vis_y_image)
        end = time.time()

        sum_time+=(end-start)
        
        image_fused = clamp(image_fused[0])
        image_fused = YCrCb2RGB(image_fused, cb[0], cr[0])
        image_fused = transforms.ToPILImage()(image_fused)
        image_fused.save(f'{result_save_path}/{img_name[0]}')

print('use time: ', sum_time)