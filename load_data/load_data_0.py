import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from imageio.v2 import imread
from utils.utils import img_norm
import torch.nn.functional as F
from PIL import Image


class VideoDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()

        self.image_list, self.mask_list = [], []

        image_dir = os.path.join(root_dir, "image")
        mask_dir = os.path.join(root_dir, "mask")

        self.dir_list = sorted(os.listdir(image_dir))

        self.length = 0

        for i in range(len(self.dir_list)):
            temp_list = []
            cur_dir = sorted(os.listdir(os.path.join(image_dir, self.dir_list[i])))
            for j in range(len(cur_dir)):
                temp_list.append(os.path.join(image_dir, self.dir_list[i], cur_dir[j]))
            self.length += len(temp_list)
            self.image_list.append(temp_list)

            temp_list = []
            cur_dir = sorted(os.listdir(os.path.join(mask_dir, self.dir_list[i])))
            for j in range(len(cur_dir)):
                temp_list.append(os.path.join(mask_dir, self.dir_list[i], cur_dir[j]))
            self.mask_list.append(temp_list)

        self.image_size = 224

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        m = np.random.randint(len(self.image_list))
        n = np.arange(len(self.image_list[m]))

        cur_image, cur_mask = [], []

        for i in range(len(n)):
            cur_image.append(self.image_list[m][n[i % len(self.image_list[m])]])
            cur_mask.append(self.mask_list[m][n[i % len(self.image_list[m])]])

            temp_image = imread(cur_image[i])

            temp_image = torch.from_numpy(
                img_norm(temp_image.astype(np.float32) / 255.0)
            )
            temp_image = F.interpolate(
                temp_image.unsqueeze(0).permute(0, 3, 1, 2),
                size=(self.image_size, self.image_size),
            )
            image = temp_image

            temp_mask = np.array(Image.open(cur_mask[i]).convert("L"))
            temp_mask = torch.from_numpy(temp_mask.astype(np.float32) / 255.0)
            temp_mask = F.interpolate(
                temp_mask.view(1, temp_mask.shape[0], temp_mask.shape[1], 1).permute(
                    0, 3, 1, 2
                ),
                size=(self.image_size, self.image_size),
            ).squeeze()
            temp_mask = temp_mask.view(1, temp_mask.shape[0], temp_mask.shape[1])
            mask = temp_mask

        return image.squeeze(), mask.squeeze()


if __name__ == "__main__":
    train_dataset = VideoDataset("D:\\Desktop\\zongliang\\processed_data\\our_busv")
    train_loader = DataLoader(
        train_dataset,
        num_workers=0,
        batch_size=4,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    count = 0
    for image, mask in train_loader:
        count += 1
        print(image.shape)
        print(mask.shape)
        print("-" * 100)
        # image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
        # mask = mask.view(-1, mask.shape[2], mask.shape[3])
        # print(image.shape)
        # print(mask.shape)
    print("-" * 100)
    print(count)
