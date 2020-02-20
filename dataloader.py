import numpy as np
from torch.utils.data import Dataset
import random
import os


class Sequence_Dataset(Dataset):
    def __init__(self, normal_data_path, collision_data_path, normal_map_path, collision_map_path, image_size):
        self.normal_map_path = normal_map_path
        self.collision_map_path = collision_map_path

        self.data_normal = np.load(normal_data_path)
        self.data_collision = np.load(collision_data_path)

        self.map_normal = np.load(self.normal_map_path)
        self.map_normal = np.expand_dims(self.map_normal, axis=1)
        self.map_collision = np.load(self.collision_map_path)
        self.map_collision = np.expand_dims(self.map_collision, axis=1)
        self.indices = range(len(self))

    def __getitem__(self, index):
        ref_index = random.randint(0, len(self.data_normal)-1)
        return self.data_normal[index], self.data_collision[ref_index], self.map_normal[index], self.map_collision[ref_index]
    
    def __len__(self):
        return len(self.data_normal)


class DPGMM_Dataset(Dataset):
    def __init__(self, extended_data_path, normal_data_path, collision_data_path):
        self.data_extend = np.load(os.path.join(extended_data_path, 'data.extended.train.npy'))
        self.data_normal = np.load(normal_data_path)
        self.data_collision = np.load(collision_data_path)
        self.indices = range(len(self))

    def __getitem__(self, index):
        return self.data_normal[index], self.data_collision[index], self.data_extend[index]
    
    def __len__(self):
        return len(self.data_normal)
