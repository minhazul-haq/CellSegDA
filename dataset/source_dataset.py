# author: Mohammad Minhazul Haq
# source dataset loader

import numpy as np
from torch.utils import data
from PIL import Image
import os


class SourceDataSet(data.Dataset):
    def __init__(self, root_dir, image_folder='train_images', mask_folder='train_masks', max_iters=None,
                 resize_size=None, mean=0.0, std=1.0):
        dataset = root_dir.split('/')[-1]

        if dataset.startswith('kirc'):
            image_extension = '.tiff'
        elif dataset.startswith('tnbc'):
            image_extension = '.png'

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
            mask_filename = image_filename.replace(image_folder, mask_folder)
            mask_filename = mask_filename.replace('_image' + image_extension, '_mask.png')

            self.files.append({"image": image_filename,
                               "mask": mask_filename,
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

        mask = Image.open(datafiles["mask"])
        if not self.resize_size == None:
            mask = mask.resize(self.resize_size, Image.NEAREST)
        mask = np.expand_dims(np.array(mask.convert('1')), 2).astype('float32')
        mask = mask.transpose((2, 0, 1))

        return image, mask, datafiles["name"]
