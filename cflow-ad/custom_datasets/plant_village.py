import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


PLANT_VILLAGE_CLASS_NAMES =['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
NUM_TO_CLASS = {0: 'Pepper__bell___Bacterial_spot', 1: 'Pepper__bell___healthy', 2: 'Potato___Early_blight', 3: 'Potato___Late_blight', 4: 'Potato___healthy', 5: 'Tomato_Bacterial_spot', 6: 'Tomato_Early_blight', 7: 'Tomato_Late_blight', 8: 'Tomato_Leaf_Mold', 9: 'Tomato_Septoria_leaf_spot', 10: 'Tomato_Spider_mites_Two_spotted_spider_mite', 11: 'Tomato__Target_Spot', 12: 'Tomato__Tomato_YellowLeaf__Curl_Virus', 13: 'Tomato__Tomato_mosaic_virus', 14: 'Tomato_healthy'}
CLASS_TO_NUM = {'Pepper__bell___Bacterial_spot': 0, 'Pepper__bell___healthy': 1, 'Potato___Early_blight': 2, 'Potato___Late_blight': 3, 'Potato___healthy': 4, 'Tomato_Bacterial_spot': 5, 'Tomato_Early_blight': 6, 'Tomato_Late_blight': 7, 'Tomato_Leaf_Mold': 8, 'Tomato_Septoria_leaf_spot': 9, 'Tomato_Spider_mites_Two_spotted_spider_mite': 10, 'Tomato__Target_Spot': 11, 'Tomato__Tomato_YellowLeaf__Curl_Virus': 12, 'Tomato__Tomato_mosaic_virus': 13, 'Tomato_healthy': 14}

TOTAL_SAMPLE_NUM = 500

class PlantVillageDataset(Dataset):
    def __init__(self, c, phase='train', split_ratio=0.8):
        #
        self.c = c
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.phase = phase
        self.cropsize = c.crp_size
        self.split_ratio = split_ratio
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
        total_num = c.sample_num if isinstance(c.sample_num, int) else TOTAL_SAMPLE_NUM
        self.x, self.y, self.mask = self.load_dataset_folder(total_sample_num=total_num)
        # set transforms
        if self.phase == 'train':
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
        mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        # else:
        #     mask = Image.open(mask)
        #     mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)
    def get_random_samples_with_limit(self, x,y,mask, limit, seed=123):
        num_sample_total = len(x)
        limit = 100 if 100 < num_sample_total else num_sample_total
        ran_idx = np.random.randint(0, num_sample_total, limit)
        # Mix up the array (To make sure that there are both healthy and disease samples)
        x_trimmed = [x[idx] for idx in ran_idx]
        y_trimmed = [y[idx] for idx in ran_idx]
        m_trimmed = [mask[idx] for idx in ran_idx]
        # Cut off by the limit
        return x_trimmed, y_trimmed, m_trimmed

    def take_random(self, data_pool, num, seed=123):
        if num > len(data_pool):
            return data_pool
        ran_idx = np.random.randint(0, len(data_pool), num)
        return [data_pool[idx] for idx in ran_idx]

    def get_samples_file_path_of_class(self, class_name):
        img_type_dir = os.path.join(self.dataset_path, class_name)
        if not os.path.isdir(img_type_dir):
            raise ValueError(f'no dir exist with name {img_type_dir}')
        img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith('.JPG')])
        return img_fpath_list

    def check_validity(self, x, y):
        assert len(x) > 0, 'Dataset should not be null (No sample)'
        assert len(x) == len(y), 'number of x and y should be same'

    def handle_add_samples(self, x, y, mask, samples, label):
        x.extend(samples)
        y.extend([label] * len(samples))
        mask.extend([None] * len(samples))

    # For creating test and validation dataset quickly
    def create_mixed_class_dataset(self, x,y,mask,norm_pool, ano_pool,total, norm_ratio=0.5, seed=111):
        norm_num =int(total * norm_ratio)
        ano_num = total - norm_num
        rand_norm_samples = self.take_random(norm_pool, norm_num, seed=11)
        self.handle_add_samples(x,y,mask, rand_norm_samples, 0)
        ran_ano_samples = self.take_random(ano_pool, ano_num, seed=12)
        self.handle_add_samples(x,y,mask, ran_ano_samples, 1)

    def load_dataset_folder(self, total_sample_num=TOTAL_SAMPLE_NUM, split_ratios=(0.6, 0.2, 0.2), seed=123):
        phase = self.phase
        train_ratio, val_ratio, test_ratio = split_ratios
        # HEALTHY GUYS
        healthy_samples = self.get_samples_file_path_of_class(self.healthy_classname)
        # DISEAED GUYS
        diseased_samples = []
        disased_classes = [c_type for c_type in self.classes_of_plant if c_type != self.healthy_classname]
        for diased_type in disased_classes:
            samples =  self.get_samples_file_path_of_class(diased_type)
            diseased_samples.extend(samples)
        #
        SAMPLE_SIZE = min(total_sample_num, len(healthy_samples) + len(diseased_samples))
        train_sample_num = int(SAMPLE_SIZE * train_ratio)
        val_sample_num = int(SAMPLE_SIZE * val_ratio)
        test_sample_num = int(SAMPLE_SIZE * test_ratio)
        #
        x, y, mask = [], [], []

        # LOAD DATA BY APPROPRIATE SPLITS
        if phase == 'train':
            # load images
            rand_norm_samples = self.take_random(healthy_samples, train_sample_num)
            self.handle_add_samples(x,y,mask, rand_norm_samples, 0)
        elif phase== 'val':
            self.create_mixed_class_dataset(x,y,mask,healthy_samples,diseased_samples,val_sample_num,0.7,seed=seed+33)
        elif phase == 'test':
            self.create_mixed_class_dataset(x,y,mask,healthy_samples,diseased_samples,test_sample_num,0.7,seed=seed+22)
            assert (0 in y) and (1 in y), "There shuold be samples of both healthy and disease"
        else:
            raise KeyError("Unknown running phase (Must be train/test)")

        return list(x), list(y), list(mask)