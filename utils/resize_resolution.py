from PIL import Image
import os

def resize_image(input_path, output_path):
    with Image.open(input_path) as img:
        original_width, original_height = img.size
        new_width = original_width // 2
        new_height = original_height // 2
        resized_img = img.resize((new_width, new_height), resample=Image.LANCZOS)
        resized_img.save(output_path)
        

if __name__ == '__main__':
    ir_dir = '/home/ws/datasets/image_fusion/M3FD/ir'
    vi_dir = '/home/ws/datasets/image_fusion/M3FD/vi'
    ir_out_dir = '/home/ws/datasets/image_fusion/M3FD_2x/ir'
    vi_out_dir = '/home/ws/datasets/image_fusion/M3FD_2x/vi'

    os.makedirs(ir_out_dir,exist_ok=True)
    os.makedirs(vi_out_dir,exist_ok=True)

    for name in os.listdir(ir_dir):
        ir_path = os.path.join(ir_dir,name)
        ir_out_path = os.path.join(ir_out_dir,name)
        resize_image(ir_path,ir_out_path)

        vi_path = os.path.join(vi_dir,name)
        vi_out_path = os.path.join(vi_out_dir,name)
        resize_image(vi_path,vi_out_path)

