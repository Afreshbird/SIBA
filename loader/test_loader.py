import os
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils.RGB2YCrBb import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])

# from DePF: A Novel Fusion Approach based on Decomposition Pooling for Infrared and Visible Images
class TestLoader(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path  
            if sub_dir == 'vi':
                self.vis_path = temp_path 

        self.name_list = os.listdir(self.inf_path) 
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]
        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')
        vis_image = Image.open(os.path.join(self.vis_path, name))
        image_size = inf_image.size
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name, image_size

    def __len__(self):
        return len(self.name_list)