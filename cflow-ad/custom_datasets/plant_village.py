import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


PLANT_VILLAGE_CLASS_NAMES =['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
NUM_TO_CLASS = {0: 'Pepper__bell___Bacterial_spot', 1: 'Pepper__bell___healthy', 2: 'Potato___Early_blight', 3: 'Potato___Late_blight', 4: 'Potato___healthy', 5: 'Tomato_Bacterial_spot', 6: 'Tomato_Early_blight', 7: 'Tomato_Late_blight', 8: 'Tomato_Leaf_Mold', 9: 'Tomato_Septoria_leaf_spot', 10: 'Tomato_Spider_mites_Two_spotted_spider_mite', 11: 'Tomato__Target_Spot', 12: 'Tomato__Tomato_YellowLeaf__Curl_Virus', 13: 'Tomato__Tomato_mosaic_virus', 14: 'Tomato_healthy'}
CLASS_TO_NUM = {'Pepper__bell___Bacterial_spot': 0, 'Pepper__bell___healthy': 1, 'Potato___Early_blight': 2, 'Potato___Late_blight': 3, 'Potato___healthy': 4, 'Tomato_Bacterial_spot': 5, 'Tomato_Early_blight': 6, 'Tomato_Late_blight': 7, 'Tomato_Leaf_Mold': 8, 'Tomato_Septoria_leaf_spot': 9, 'Tomato_Spider_mites_Two_spotted_spider_mite': 10, 'Tomato__Target_Spot': 11, 'Tomato__Tomato_YellowLeaf__Curl_Virus': 12, 'Tomato__Tomato_mosaic_virus': 13, 'Tomato_healthy': 14}

class PlantVillageDataset(Dataset):
    def __init__(self, c, is_train=True):
        #
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        # Checking
        acceptable_plants = list(dict.fromkeys([label.split('__')[0] for label in PLANT_VILLAGE_CLASS_NAMES]))
        plant_names = list(filter(lambda label: label.find('_') == -1, acceptable_plants))
        # assert c.class_name in PLANT_VILLAGE_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, PLANT_VILLAGE_CLASS_NAMES)
        assert c.class_name in plant_names, f'class_name: {c.class_name}, should be in {plant_names}'
        # Storing healthy label name for this plant
        self.classes_of_plant = [label for label in PLANT_VILLAGE_CLASS_NAMES if label.lower().find(c.class_name.lower()) != -1]
        self.healthy_classname = [item for item in self.classes_of_plant if item.find('healthy')][0]
        assert self.healthy_classname, f'There must be a healthy class for plant {c.class_name}'
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        #x = Image.open(x).convert('RGB')
        x = Image.open(x)
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        if phase == 'train':
            # load images
            img_type_dir = os.path.join(self.dataset_path, self.healthy_classname)
            if not os.path.isdir(img_type_dir):
                raise ValueError(f'no dir exist with name {img_type_dir}')
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.JPG')])
            x.extend(img_fpath_list)

            # load gt labels
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))

        elif phase == 'test':
            img_types = sorted(self.classes_of_plant)
            for img_type in img_types:
                # load images
                img_type_dir = os.path.join(self.dataset_path, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                        for f in os.listdir(img_type_dir)
                                        if f.endswith('.JPG')])
                x.extend(img_fpath_list)

                # load gt labels
                if img_type == self.healthy_classname:
                    y.extend([0] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
                else:
                    y.extend([1] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
        assert len(x) > 0, 'Dataset should not be null (No sample)'
        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
