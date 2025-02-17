import torch
from torch.utils.data import Dataset
import pandas as pd
from util import farthest_point_sample, index_points
import os
import numpy as np
import laspy
import h5py
import glob
torch.manual_seed(42)
    
class tald(Dataset):
    def __init__(self, device, grid_size, points_taken, partition='train'):
       
        path = os.path.join("data", "tald", f"{partition}", f"norm_{grid_size}_{points_taken}.npz")
        
        if os.path.exists(path): # this starts from the system's path
            tiles = np.load(path)
            self.data = tiles['x']
            self.label = tiles['y']
        else:
            self.data = None
            self.label = None
            for fl in glob.glob(os.path.join("data", "tald", f"{partition}","*.laz")):
                las = laspy.read(fl)
                # las_classification = las_label_replace(las)
                # df["Classification"].replace([3., 2., 6., 5. , 4., 7.], [0, 1, 2, 3, 4, 5], inplace=True)
                # print(las.x.max() - las.x.min())
                las_classification = las_label_replace(las, 'tald')
                data, label = grid_als(device, grid_size, points_taken, las.xyz, las_classification)

                if self.data is None and self.label is None:
                    self.data = data
                    self.label = label
                else:
                    self.data = np.append(self.data, data, axis = 0)

                    self.label = np.append(self.label, label, axis = 0)

            if partition == 'test':
                np.savez(os.path.join("data", "tald", f"{partition}", f"not_norm_{grid_size}_{points_taken}.npz"), x = self.data, y = self.label)
            mn = np.min(self.data, axis = 1, keepdims=True)
            mx = np.max(self.data, axis = 1, keepdims=True)
            self.data = (self.data - mn)/(mx - mn)
            np.savez(os.path.join("data", "tald", f"{partition}", f"norm_{grid_size}_{points_taken}.npz"), x = self.data, y = self.label)
    
    def __getitem__(self, item):
        pointcloud = torch.tensor(self.data[item]).float()
        label = torch.tensor(self.label[item])

        return pointcloud, label

    def __len__(self):
        return len(self.data)

class modelnet40(Dataset):
    def __init__(self):
        with h5py.File(os.path.join("data", "modelnet40_ply_hdf5_2048", "ply_data_train0.h5")) as F:
            self.data, self.label = F['data'][()], F['label'][()]
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return self.data.shape[0]

class Dales(Dataset):
    def __init__(self, device, grid_size, points_taken, partition='train'):
       
        path = os.path.join("data", "Dales", f"{partition}", f"norm_{grid_size}_{points_taken}.npz")
        
        if os.path.exists(path): # this starts from the system's path
            tiles = np.load(path)
            self.data = tiles['x']
            self.label = tiles['y']
        else:
            self.data = None
            self.label = None
            for fl in glob.glob(os.path.join("data", "Dales", f"{partition}","*.las")):
                las = laspy.read(fl)
                las_classification = las_label_replace(las, 'Dales')
                data, label = grid_als(device, grid_size, points_taken, las.xyz, las_classification)
             
                if self.data is None and self.label is None:
                    self.data = data
                    self.label = label
                else:
                    self.data = np.append(self.data, data, axis = 0)

                    self.label = np.append(self.label, label, axis = 0)

            if partition == 'test':
                np.savez(os.path.join("data", "Dales", f"{partition}", f"not_norm_{grid_size}_{points_taken}.npz"), x = self.data, y = self.label)
            mn = np.min(self.data, axis = 1, keepdims=True)
            mx = np.max(self.data, axis = 1, keepdims=True)
            self.data = (self.data - mn)/(mx - mn)
            np.savez(os.path.join("data", "Dales", f"{partition}", f"norm_{grid_size}_{points_taken}.npz"), x = self.data, y = self.label)

    def __getitem__(self, item):
        pointcloud = torch.tensor(self.data[item]).float()
        label = torch.tensor(self.label[item])

        return pointcloud, label

    def __len__(self):
        return len(self.data)

def las_label_replace(las, dataset):
    # DALES
    if dataset == 'Dales':
        las_classification = np.asarray(las.classification)
        mapping = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
        for old, new in mapping.items():
            las_classification[las_classification == old] = new

    # TALD
    # df["Classification"].replace([3., 2., 6., 5. , 4., 7.], [0, 1, 2, 3, 4, 5], inplace=True)
    elif dataset == 'tald':
        las_classification = np.asarray(las.classification)
        mapping = {2:0, 3:1, 4:1, 5:7}
        for old, new in mapping.items():
            las_classification[las_classification == old] = new
    
    return las_classification

def grid_als(device, grid_size, points_taken, data, classification):
    grid_point_clouds = {}
    grid_point_clouds_label = {}
    for point, label in zip(data, classification):
        grid_x = int(point[0] / grid_size)
        grid_y = int(point[1] / grid_size)

        if (grid_x, grid_y) not in grid_point_clouds:
            grid_point_clouds[(grid_x, grid_y)] = []
            grid_point_clouds_label[(grid_x, grid_y)] = []
        
        grid_point_clouds[(grid_x, grid_y)].append(point)
        grid_point_clouds_label[(grid_x, grid_y)].append(label)

    tiles = []
    tiles_labels = []

    grid_lengths = [len(i) for i in grid_point_clouds.values()]
    mn_points = min(grid_lengths)
    mx_points = max(grid_lengths)
    min_grid_points = (mx_points - mn_points) * 0.1

    for grid, label in zip(grid_point_clouds.values(), grid_point_clouds_label.values()):

        len_grid = len(grid)

        if(len_grid>min_grid_points): # This is for excluding points which are at the boundry at the edges of the tiles

            grid = np.asarray(grid)
            label = np.asarray(label)

            if(len_grid<points_taken): # This is for if the points in the grid are less then the required points for making the grid 
                for _ in range(points_taken-len_grid):
                    grid = np.append([grid, grid[0]], axis=0)
                    label = np.append([label, label[0]], axis=0)

            
            grid = torch.tensor(grid).unsqueeze(0).to(device)
            label = torch.tensor(label).unsqueeze(0).unsqueeze(2).to(device)

            tiles_idx = farthest_point_sample(grid, points_taken) # using fps

            tiles.append(index_points(grid, tiles_idx).squeeze().cpu().numpy())
            tiles_labels.append(index_points(label, tiles_idx).squeeze().cpu().numpy())

    tiles_np = np.asarray(tiles)
    tiles_np_labels = np.asarray(tiles_labels)

    return tiles_np, tiles_np_labels

dataset = {'tald': tald, 'Dales': Dales}

if __name__ == '__main__':
    import time
    start = time.time()
    train = Dales('cuda', 25, 4096)
    end = time.time()
    print(end-start)

    # train = tald('cuda', 25, 4096, partition='test')
    # print(train[0][1].size())
    pass