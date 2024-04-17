import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread

# 标签ID到名称的映射
# label_names = [
#     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
#     'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
#     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
# ]
label_names = [
    '61', '62', '63', '71', '72', '73'
]

# 假设JSON数据已经保存在'annotations.json'文件中
# json_file_path = 'correct_save_correction.json' # voc
json_file_path = 'tire3_save_correction.json'

# 读取JSON文件
with open(json_file_path, 'r') as file:
    annotations = json.load(file)

# 绘制图像和标注的函数
def draw_image_with_annotations(data, save_path):
    # 加载图片
    img = imread(data["img_path_1"])
    
    # 创建图和坐标轴
    fig, ax = plt.subplots(1)
    ax.imshow(img)  # 显示图片
    ax.axis('off')  # 去掉坐标轴
    text_offset = 20  # 文本偏移量
    
    # 绘制bboxes_1并在左下角标注标签名称
    for bbox, label_id in zip(data["bboxes_true_1"], data["labels_1"]):
        label_name = label_names[label_id]  # -1 因为列表是从0开始的索引
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # ax.text(bbox[0], bbox[3], label_name, color='white', fontsize=12, verticalalignment='bottom',
        #         bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.3'))
        ax.text(bbox[0], bbox[3] + text_offset, label_name, color='white', fontsize=6, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.1'))

    # 绘制bboxes_2并在左上角标注标签名称及scores_2
    for bbox, label_id, score in zip(data["bboxes_true_2"], data["labels_2"], data["scores_2"]):
        label_name = label_names[label_id]  # -1 因为列表是从0开始的索引
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
                                 linewidth=1.5, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        # ax.text(bbox[0], bbox[1], f"{label_name} {score:.2f}", color='white', fontsize=12, verticalalignment='top',
        #         bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.3'))
        ax.text(bbox[0], bbox[1] - text_offset, f"{label_name} {score:.2f}", color='white', fontsize=6, verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.1'))

    # 保存图像到指定路径
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)  # 关闭图形，避免在内存中累积

# 在你的本地环境执行时，请确保图片路径正确，且你的环境可以访问这些图片。
for index, data in enumerate(annotations):
    # save_path = f"outputs/voc_correction/image_{index}.png"  # 定义保存路径和文件名
    save_path = f"outputs/tire_correction/image_{index}.png"  # 定义保存路径和文件名
    draw_image_with_annotations(data, save_path)
