from utils.utils import img_norm
import torch
import os
import numpy as np
from PIL import Image
from imageio.v2 import imread
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class VideoDataset(Dataset):
    def __init__(self, dir_, epochs, size=224, group=5):
        self.img_list = []
        self.label_list = []

        self.group = group

        dir_img = os.path.join(dir_, "image")
        dir_gt = os.path.join(dir_, "mask")

        self.dir_list = sorted(os.listdir(dir_img))

        self.length = 0
        for i in range(len(self.dir_list)):
            tmp_list = []
            cur_dir = sorted(os.listdir(os.path.join(dir_img, self.dir_list[i])))
            for j in range(len(cur_dir)):
                tmp_list.append(os.path.join(dir_img, self.dir_list[i], cur_dir[j]))
            self.length += len(tmp_list)
            self.img_list.append(tmp_list)

            tmp_list = []
            cur_dir = sorted(os.listdir(os.path.join(dir_gt, self.dir_list[i])))
            for j in range(len(cur_dir)):
                tmp_list.append(os.path.join(dir_gt, self.dir_list[i], cur_dir[j]))
            self.label_list.append(tmp_list)

        self.img_size = size
        self.dataset_length = epochs
        # self.count = 0

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        rd = np.random.randint(0, len(self.img_list))
        rd2 = np.random.permutation(len(self.img_list[rd]))

        cur_img = []
        cur_gt = []

        for i in range(self.group):
            cur_img.append(self.img_list[rd][rd2[i % len(self.img_list[rd])]])
            cur_gt.append(self.label_list[rd][rd2[i % len(self.img_list[rd])]])

        group_img = []
        group_gt = []

        for i in range(self.group):
            tmp_img = imread(cur_img[i])

            tmp_img = torch.from_numpy(img_norm(tmp_img.astype(np.float32) / 255.0))
            tmp_img = F.interpolate(
                tmp_img.unsqueeze(0).permute(0, 3, 1, 2),
                size=(self.img_size, self.img_size),
            )
            group_img.append(tmp_img)

            tmp_gt = np.array(Image.open(cur_gt[i]).convert("L"))
            tmp_gt = torch.from_numpy(tmp_gt.astype(np.float32) / 255.0)
            tmp_gt = F.interpolate(
                tmp_gt.view(1, tmp_gt.shape[0], tmp_gt.shape[1], 1).permute(0, 3, 1, 2),
                size=(self.img_size, self.img_size),
            ).squeeze()
            tmp_gt = tmp_gt.view(1, tmp_gt.shape[0], tmp_gt.shape[1])
            group_gt.append(tmp_gt)

        # self.count += 1
        group_img = torch.cat(group_img, 0)
        # print(group_img.shape)
        group_gt = torch.cat(group_gt, 0)
        # print(group_gt.shape)
        # print("$" * 50 + f" [{self.count}] " + "$" * 50)

        return group_img, group_gt


if __name__ == "__main__":
    train_dataset = VideoDataset(
        r"D:\Desktop\zongliang\processed_data\our_busv_1\train", 100000, 224, 5
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=0,
        batch_size=4,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    i = 0
    for image, mask in train_loader:
        i += 1
        print(image.shape)  # torch.Size([4, 5, 3, 224, 224])
        print(mask.shape)  # torch.Size([4, 5, 224, 224])
        image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
        mask = mask.view(-1, mask.shape[2], mask.shape[3])
        print(image.shape)  # torch.Size([20, 3, 224, 224])
        print(mask.shape)  # torch.Size([20, 224, 224])
        print("*" * 50 + f" {i} " + "*" * 50)
    print(i)
