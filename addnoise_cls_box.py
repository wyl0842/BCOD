import json
import random
import numpy as np

def add_noise_to_coco_annotations(coco_annotations, class_mapping, cls_proportion=0.2, box_proportion=0):
    # 计算需要处理的样本数量
    num_samples_to_modify = int(len(coco_annotations['annotations']) * cls_proportion)

    # 随机选择要处理的样本索引
    samples_to_modify = random.sample(range(len(coco_annotations['annotations'])), num_samples_to_modify)

    # 对选择的样本进行分类标签的随机变化
    modified_image_ids = set()  # 用于存储被修改的图片ID
    for idx in samples_to_modify:
        annotation = coco_annotations['annotations'][idx]

        # 记录被修改的图片ID
        modified_image_ids.add(annotation['image_id'])

        # 随机变化分类标签
        if annotation['category_id'] < 6:
            annotation['category_id'] = random.choice(class_mapping[annotation['category_id']])

    
    for i in range(len(coco_annotations['annotations'])):
        annotation = coco_annotations['annotations'][i]
        # 添加高斯噪声扰动到标注框
        annotation['bbox'] = add_gaussian_noise_to_bbox(annotation['bbox'], box_proportion)

        # 截断框的坐标，确保不超出图片范围
        annotation['bbox'] = truncate_bbox(coco_annotations, annotation['bbox'], annotation['image_id'])

    # 打印被修改类别的图片ID
    print("Modified Image IDs:", modified_image_ids)
    
    return coco_annotations

def add_gaussian_noise_to_bbox(bbox, noise_percentage):
    # 计算四个坐标分别的百分比
    x1_percentage = random.uniform(-noise_percentage, noise_percentage)
    y1_percentage = random.uniform(-noise_percentage, noise_percentage)
    x2_percentage = random.uniform(-noise_percentage, noise_percentage)
    y2_percentage = random.uniform(-noise_percentage, noise_percentage)

    # 添加均匀分布噪声扰动
    bbox[0] = bbox[0] + bbox[2] * x1_percentage  # x
    bbox[1] = bbox[1] + bbox[3] * y1_percentage  # y
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]
    x2 = x2 + bbox[2] * x2_percentage
    y2 = y2 + bbox[3] * y2_percentage
    bbox[2] = x2 - bbox[0]  # width
    bbox[3] = y2 - bbox[1]  # height

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
        bbox[2] = min(image_width - 1 - bbox[0], max(bbox[2], 0))
        bbox[3] = min(image_height - 1 - bbox[1], max(bbox[3], 0))

    return bbox

def main():
    # 读取原始的COCO JSON文件
    with open('/home/wangyl/Code/mmdetection/data/voc2007trainval.json', 'r') as f:
        original_annotations = json.load(f)

    # 定义分类标签映射，每个类别映射到一个列表，随机选择一个进行替换
    class_mapping = {
        0: [1, 2, 3, 4, 5],   # Example: Original class 1 may become 2, 3, or 4
        1: [0, 2, 3, 4, 5],   # Example: Original class 2 may become 1, 3, or 4
        2: [0, 1, 3, 4, 5],   # Example: Original class 1 may become 2, 3, or 4
        3: [0, 1, 2, 4, 5],   # Example: Original class 2 may become 1, 3, or 4
        4: [0, 1, 2, 3, 5],   # Example: Original class 1 may become 2, 3, or 4
        5: [0, 1, 2, 3, 4],   # Example: Original class 2 may become 1, 3, or 4
        # Add mappings for other classes as needed
    }

    # 复制一份原始标签，避免直接修改原始数据
    modified_annotations = original_annotations.copy()

    # 添加噪声，指定 proportion 参数为需要处理的样本比例
    modified_annotations = add_noise_to_coco_annotations(modified_annotations, class_mapping, cls_proportion=0.2, box_proportion=0)

    # 保存修改后的标签到新的JSON文件
    with open('/home/wangyl/Code/mmdetection/data/voc2007trainval_cls0.2_box0.json', 'w') as f:
        json.dump(modified_annotations, f)

if __name__ == "__main__":
    main()