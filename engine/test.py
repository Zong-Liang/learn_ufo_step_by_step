import os
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from models.model import build_model
from config.config import *


def test_step(
    device,
    model_best_pth,
    test_data_dir,
    test_output_dir,
    group_size,
    image_size,
    image_dir_name,
):
    # 构建模型
    net = build_model(device).to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_best_pth))
    net.eval()

    # 图像预处理
    img_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_transform_gray = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ]
    )
    with torch.no_grad():
        # 遍历数据路径
        for p in range(len(test_data_dir)):
            all_class = os.listdir(os.path.join(test_data_dir[p], image_dir_name))

            image_list, save_list = [], []

            # 遍历每个类别
            for s in range(len(all_class)):
                image_path = sorted(
                    os.listdir(
                        os.path.join(test_data_dir[p], image_dir_name, all_class[s])
                    )
                )

                # 确定每个组的索引
                idx = []
                block_size = (len(image_path) + group_size - 1) // group_size
                for ii in range(block_size):
                    cur = ii
                    while cur < len(image_path):
                        idx.append(cur)
                        cur += block_size

                new_image_path = []
                for ii in range(len(image_path)):
                    new_image_path.append(image_path[idx[ii]])
                image_path = new_image_path
                # print(len(image_path))

                # 构建图像和保存路径列表
                image_list.append(
                    list(
                        map(
                            lambda x: os.path.join(
                                test_data_dir[p], image_dir_name, all_class[s], x
                            ),
                            image_path,
                        )
                    )
                )
                save_list.append(
                    list(
                        map(
                            lambda x: os.path.join(
                                test_output_dir[p], all_class[s], x[:-4] + ".png"
                            ),
                            image_path,
                        )
                    )
                )
            # 遍历每个类别的图像
            for i in range(len(image_list)):
                cur_class_all_image = image_list[i]
                cur_class_rgb = torch.zeros(
                    len(cur_class_all_image), 3, image_size, image_size
                )

                # 处理每张图像
                for m in range(len(cur_class_all_image)):
                    rgb_ = Image.open(cur_class_all_image[m])
                    # 判断图像模式是RGB还是灰度
                    if rgb_.mode == "RGB":
                        rgb_ = img_transform(rgb_)
                    else:
                        rgb_ = img_transform_gray(rgb_)
                    cur_class_rgb[m, :, :, :] = rgb_

                cur_class_mask = torch.zeros(
                    len(cur_class_all_image), image_size, image_size
                )
                divided = len(cur_class_all_image) // group_size
                rested = len(cur_class_all_image) % group_size
                # 处理整除部分
                if divided != 0:
                    for k in range(divided):
                        group_rgb = cur_class_rgb[
                            (k * group_size) : ((k + 1) * group_size)
                        ]
                        group_rgb = group_rgb.to(device)
                        _, pred_mask = net(group_rgb)
                        cur_class_mask[
                            (k * group_size) : ((k + 1) * group_size)
                        ] = pred_mask
                # 处理余数部分
                if rested != 0:
                    group_rgb_tmp_l = cur_class_rgb[-rested:]
                    group_rgb_tmp_r = cur_class_rgb[: group_size - rested]
                    group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
                    group_rgb = group_rgb.to(device)
                    _, pred_mask = net(group_rgb)
                    cur_class_mask[(divided * group_size) :] = pred_mask[:rested]

                # 创建保存路径
                class_save_path = os.path.join(test_output_dir[p], all_class[i])
                if not os.path.exists(class_save_path):
                    os.makedirs(class_save_path)

                # 保存预测结果
                for j in range(len(cur_class_all_image)):
                    exact_save_path = save_list[i][j]
                    result = cur_class_mask[j, :, :].numpy()
                    result = Image.fromarray(result * 255)
                    w, h = Image.open(image_list[i][j]).size
                    result = result.resize((w, h), Image.BILINEAR)
                    result.convert("L").save(exact_save_path)

            print("done")


if __name__ == "__main__":
    test_step(
        device,
        model_best_pth,
        test_data_dir,
        test_output_dir,
        group_size,
        image_size,
        image_dir_name,
    )
