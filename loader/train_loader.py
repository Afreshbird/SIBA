import pathlib
import cv2
import numpy as np
import kornia.utils
import torch.utils.data



# from CHITNet: A Complementary to Harmonious Information Transfer Network for Infrared and Visible Image Fusion
class TrainLoader(torch.utils.data.Dataset):
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, patch_size=128):
        super(TrainLoader, self).__init__()
        self.ps = patch_size
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def get_patch(self, ir, vis):
        H, W = ir.shape[1], ir.shape[2]
        x, y = np.random.randint(10, H-10-self.ps+1), np.random.randint(10, W-10-self.ps+1)
        ir_crop = ir[:, x:x+self.ps, y:y+self.ps]
        vis_crop = vis[:, x:x+self.ps, y:y+self.ps]
        return ir_crop, vis_crop
    
    def __getitem__(self, index):
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]
        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)
        ir_crop, vis_crop = self.get_patch(ir, vi)
        return ir_crop, vis_crop

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts