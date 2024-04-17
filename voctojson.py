import os
import xml.etree.ElementTree as ET
import json
from PIL import Image
from tqdm import tqdm  # 导入tqdm库，用于创建进度条

def xml_to_dict(xml_root):
    """将XML转换为字典"""
    xml_dict = {}
    for child in xml_root:
        if child.tag == 'object':
            obj_dict = {}
            for obj_child in child:
                if obj_child.tag == 'bndbox':
                    bbox_dict = {}
                    for bbox_child in obj_child:
                        bbox_dict[bbox_child.tag] = int(bbox_child.text)
                    obj_dict[obj_child.tag] = bbox_dict
                else:
                    obj_dict[obj_child.tag] = obj_child.text
            xml_dict[child.tag] = obj_dict
        else:
            xml_dict[child.tag] = child.text
    return xml_dict

def convert_to_coco(image_folder, xml_folder, output_path):
    """将图像和对应的XML文件转换为COCO格式的JSON文件"""
    images = []
    annotations = []
    categories = []
    image_id = 1
    annotation_id = 1

    # 添加类别信息
    categories.append({'id': 1, 'name': 'object', 'supercategory': 'object'})

    # 遍历图像文件夹
    for image_file in tqdm(os.listdir(image_folder), desc="Converting to COCO"):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_file)
            image = {
                'id': image_id,
                'file_name': image_file,
                'width': 0,
                'height': 0,
                'date_captured': '',
                'license': 0,
                'coco_url': '',
                'flickr_url': ''
            }

            # 获取图像尺寸
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                width, height = img.size
                image['width'] = width
                image['height'] = height

            images.append(image)

            # 解析对应的XML标注文件
            xml_file = image_file.replace('.jpg', '.xml').replace('.jpeg', '.xml').replace('.png', '.xml')
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            annotations_info = xml_to_dict(root)

            # 添加注释信息
            if 'object' in annotations_info:
                annotation_info = annotations_info['object']
                bbox = [
                    int(annotation_info['bndbox']['xmin']),
                    int(annotation_info['bndbox']['ymin']),
                    int(annotation_info['bndbox']['xmax']) - int(annotation_info['bndbox']['xmin']),
                    int(annotation_info['bndbox']['ymax']) - int(annotation_info['bndbox']['ymin'])
                ]

                annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0,
                    'segmentation': [],
                    'ignore': 0
                })

                annotation_id += 1

            image_id += 1

    # 构建COCO格式的JSON数据
    coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    # 保存JSON文件
    with open(output_path, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

if __name__ == "__main__":
    image_folder = '/home/wangyl/Code/mmdetection/data/voctire/trainfiles/train_png'
    xml_folder = '/home/wangyl/Code/mmdetection/data/voctire/trainfiles/train_xml'
    output_path = '/home/wangyl/Code/mmdetection/data/train.json'

    convert_to_coco(image_folder, xml_folder, output_path)