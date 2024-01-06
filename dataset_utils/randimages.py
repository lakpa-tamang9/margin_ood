import torch
import random
import numpy as np


class RandomImages(torch.utils.data.Dataset):
    def __init__(self, transform=None, exclude_cifar=True, data_num=10000):
        self.transform = transform

        self.data = np.load(
            "data/300K_random_images/300k/300K_random_images.npy"
        ).astype(np.uint8)

        # print(111111111111111111)

        if data_num != -1:
            all_id = list(range(len(self.data)))
            sample_id = random.sample(all_id, data_num)
            self.data = self.data[sample_id]

    def __getitem__(self, index):
        # id = self.id_sample[index]
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, 0, index  # 0 is the class

    def __len__(self):
        return len(self.data)
