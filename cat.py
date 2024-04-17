from PIL import Image
import os
from tqdm import tqdm

def merge_images(input_path1, input_path2, output_path, spacing=10):
    # 获取两个路径下的所有文件名
    files1 = os.listdir(input_path1)
    files2 = os.listdir(input_path2)

    # 确保两个路径下的文件名一一对应
    files1.sort()
    files2.sort()

    # 创建输出路径
    os.makedirs(output_path, exist_ok=True)

    # 遍历文件并拼接图片
    for file1, file2 in tqdm(zip(files1, files2)):
        img1 = Image.open(os.path.join(input_path1, file1))
        img2 = Image.open(os.path.join(input_path2, file2))

        # 获取图片的宽度和高度
        width1, height1 = img1.size
        width2, height2 = img2.size

        # 计算新图片的宽度和高度
        new_width = width1 + width2 + spacing
        new_height = max(height1, height2)

        # 创建新图片
        new_img = Image.new('RGB', (new_width, new_height), color='white')

        # 将两张图片拼接到新图片中
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (width1 + spacing, 0))

        # 保存新图片
        output_file = os.path.join(output_path, file1)
        new_img.save(output_file)

if __name__ == "__main__":
    input_path1 = '/home/wangyl/Code/mmdetection/outputs/tire_diffusiondet_r50/20240102_163742/imgs'
    input_path2 = '/home/wangyl/Code/mmdetection/outputs/tire_frcnn_r50/new_dataset_test/imgs'
    output_path = 'outputs/cats_new'
    spacing = 20  # 设置两张图片之间的间隔

    merge_images(input_path1, input_path2, output_path, spacing)