# author: Mohammad Minhazul Haq
# target dataset loader

import numpy as np
from torch.utils import data
from PIL import Image
import os


class TargetUnlabelledDataSet(data.Dataset):
    def __init__(self, root_dir, image_folder='train_images', max_iters=None,
                 resize_size=None, mean=0.0, std=1.0):
        images_dir = os.path.join(root_dir, image_folder)
        image_files = sorted(os.listdir(images_dir))
        image_filenames = []

        for file in image_files:
            image_filenames.append(os.path.join(images_dir, file))

        if not max_iters == None:
            image_filenames_len = len(image_filenames)
            image_filenames = image_filenames * int(np.ceil(float(max_iters) / image_filenames_len))

        self.files = []

        for image_filename in image_filenames:
            self.files.append({"image": image_filename,
                               "name": image_filename.split('/')[-1]})

        self.resize_size = resize_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["image"])
        if not self.resize_size == None:
            image = image.resize(self.resize_size, Image.BICUBIC)
        image = np.array(image).astype('float32')
        image = ((image - self.mean) / self.std)
        image = image.transpose((2, 0, 1))

        return image, datafiles["name"]
