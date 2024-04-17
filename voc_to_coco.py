import sys
import os
import json
import random
import xml.etree.ElementTree as ET

START_IMAGE_ID = 0
START_BBOX_ID = 0
# If necessary, pre-define category and its id
# PRE_DEFINE_CATEGORIES = {"crazing": 0, "inclusion": 1, "patches": 2, "pitted_surface": 3, "rolled-in_scale": 4, "scratches": 5}
# PRE_DEFINE_CATEGORIES = {'61': 0, '62': 1, '63': 2, '64': 3, '71': 4,
#                  '72': 5, '73': 6, '75': 7, '77': 8, '80': 9}
PRE_DEFINE_CATEGORIES = {'61': 0, '62': 1, '63': 2, '71': 3,
                 '72': 4, '73': 5}
# PRE_DEFINE_CATEGORIES = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10,
#                "dog": 11, "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19}

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def convert(xml_dir, json_file):
    xml_list = []
    for root_dir, _, files in os.walk(xml_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_list.append(file)
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES

    i_id = START_IMAGE_ID
    b_id = START_BBOX_ID
    for line in xml_list:
        # 使用random模块生成一个随机数，决定是否跳过当前迭代
        if random.random() < 0.8:
            continue
        line = line.strip()
        print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        filename = line[:-4] + ".png"
        ## The filename must be a number
        image_id = i_id
        i_id += 1
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text)
            ymin = int(get_and_check(bndbox, 'ymin', 1).text)
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': b_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            b_id += 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    # if len(sys.argv) <= 1:
    #     print('3 auguments are need.')
    #     print('Usage: %s XML_LIST.txt XML_DIR OUTPU_JSON.json'%(sys.argv[0]))
    #     exit(1)
    xml_dir = '/home/wangyl/Code/mmdetection/data/voctire/testfiles/test_xml'
    # xml_dir = '/home/wangyl/Code/mmdetection/data/VOCdevkit/VOCdevkit/VOC2007/Annotations'
    json_file = '/home/wangyl/Code/mmdetection/data/mysub_test.json'
    # json_file = '/home/wangyl/Code/mmdetection/data/voc_train.json'
    convert(xml_dir, json_file)
    # import numpy as np
    #
    # np.random.seed(1)
    # ran = np.random.choice( [i for i in range(0, 1800)], 360, replace=False)
    # map = {}
    # for num in ran:
    #     map[num] = True
    #
    # import shutil
    # from glob import glob
    # i = 0
    # for root_dir, _, files in os.walk('../NEU-DET/IMAGES'):
    #     for file in files:
    #         file_name = os.path.join(root_dir, file)
    #         if not i in map:
    #             shutil.copy(file_name, '../neu-det-coco/train/')
    #         else:
    #             shutil.copy(file_name, '../neu-det-coco/test/')
    #         i += 1