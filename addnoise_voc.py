import json
import random
import numpy as np

def add_noise_to_coco_annotations(coco_annotations, class_mapping, noise_scale=10.0, proportion=0.2):
    # 计算需要处理的样本数量
    num_samples_to_modify = int(len(coco_annotations['annotations']) * proportion)

    # 随机选择要处理的样本索引
    samples_to_modify = random.sample(range(len(coco_annotations['annotations'])), num_samples_to_modify)

    # 对选择的样本进行分类标签的随机变化和标注框的高斯噪声扰动
    modified_image_ids = set()  # 用于存储被修改的图片ID
    for idx in samples_to_modify:
        annotation = coco_annotations['annotations'][idx]

        # 记录被修改的图片ID
        modified_image_ids.add(annotation['image_id'])

        # 随机变化分类标签
        annotation['category_id'] = random.choice(class_mapping[annotation['category_id']])

        # 添加高斯噪声扰动到标注框
        annotation['bbox'] = add_gaussian_noise_to_bbox(annotation['bbox'], noise_scale)

        # 截断框的坐标，确保不超出图片范围
        annotation['bbox'] = truncate_bbox(coco_annotations, annotation['bbox'], annotation['image_id'])

    # 打印被修改的图片ID
    print("Modified Image IDs:", modified_image_ids)
    
    return coco_annotations

def add_gaussian_noise_to_bbox(bbox, noise_scale):
    # 添加高斯噪声扰动
    noise = np.random.normal(0, noise_scale, 4)
    # 将NumPy数组转换为列表并取整
    bbox = list(map(int, bbox + noise))
    return bbox

def truncate_bbox(coco_annotations, bbox, image_id):
    # 获取图片的宽和高
    image_info = next((image for image in coco_annotations['images'] if image['id'] == image_id), None)
    if image_info:
        image_width = image_info['width']
        image_height = image_info['height']

        # 截断框的坐标，确保不超出图片范围
        bbox[0] = max(0, min(bbox[0], image_width - 1))
        bbox[1] = max(0, min(bbox[1], image_height - 1))
        bbox[2] = min(image_width - 1, max(bbox[2], 0))
        bbox[3] = min(image_height - 1, max(bbox[3], 0))

    return bbox

def generate_class_mapping(num_classes):
    class_mapping = {}
    for i in range(num_classes):
        other_classes = [j for j in range(num_classes) if j != i]
        class_mapping[i] = other_classes
    return class_mapping

def main():
    # 读取原始的COCO JSON文件
    with open('/home/wangyl/Code/mmdetection/data/voc2007trainval.json', 'r') as f:
        original_annotations = json.load(f)

    num_classes = 20
    class_mapping = generate_class_mapping(num_classes)
    print(class_mapping)

    # 复制一份原始标签，避免直接修改原始数据
    modified_annotations = original_annotations.copy()

    # 添加噪声，指定 proportion 参数为需要处理的样本比例
    modified_annotations = add_noise_to_coco_annotations(modified_annotations, class_mapping, proportion=0.6)

    # 保存修改后的标签到新的JSON文件
    with open('/home/wangyl/Code/mmdetection/data/voc2007trainval_0.6.json', 'w') as f:
        json.dump(modified_annotations, f)

if __name__ == "__main__":
    main()