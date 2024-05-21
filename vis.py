import numpy as np
import open3d as o3d
import laspy 
import os 
import torch
from models import model
from datasets import dataset
from torch.utils.data import DataLoader
from train import test_loop
from sklearn.metrics import classification_report
from datasets import tald
np.random.seed(42)


colors = np.random.randn(8,3)
def visualize(data, label):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    color = np.zeros((len(data), 3))
    for j in range(8):
        color[label == j] += colors[j]

    pcd.colors = o3d.utility.Vector3dVector(color)

    return pcd
import torch.nn as nn

# model = {'NPCT': NaivePointTransformer, 'SPCT': SimplePointTransformer, 'PCT': PointTransformer, 
#         'PCT_FP': PointTransformer_FP, 'PCT_FPMOD': PointTransformer_FPMOD, 'PCT_FPADV': PointTransformer_FPADV}

def visualize_model(model_path, model_name, dataset_name):
    dataset = dataset[dataset_name]
    
    loader = DataLoader(dataset('cuda', 25, 4096, partition='test'), batch_size = 8)
    tiles = np.load(os.path.join("data", f"{dataset_name}" , 'test', f"not_norm_25_4096.npz"))
    data = tiles['x'].reshape(-1, 3)
    # print(data.shape)
    label = tiles['y'].reshape(-1)
    model = model[model_name]().to('cuda')
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    loss_fn = nn.CrossEntropyLoss()
    _,acc,bal_acc,preds = test_loop(loader, loss_fn, model, 'cuda')
    print(f'{acc=}')
    print(f'{bal_acc=}')
    targets_names = ['ground', 'vegetation', 'cars', 'trucks', 'power_lines', 'fences', 'poles', 'buildings']
    preds = np.asarray(preds).reshape(-1)
    print(label.shape, preds.shape)
    print(classification_report(label, preds, target_names=targets_names))
    pcd = visualize(data, preds)
    return pcd

def res_plot(csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(10,5)
    df[['train_loss', 'val_loss']].plot(ax=ax[0])
    df[['train_acc', 'val_acc']].plot(ax=ax[1])
    plt.savefig(csv_path.split(".")[0]+'.png')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset = 'Dales'
    model_name = "PCT_FPADV"
    model_no = 5
    epoch = 160
    model_path = os.path.join("checkpoints", f"model_{model_no}", f"{model_name}_{model_no}_{epoch}.pt")
    csv_path = os.path.join("checkpoints", f"model_{model_no}", f"{model_name}_{model_no}_{epoch}.csv")
    pcd = visualize_model(model_path, model_name, dataset)
    o3d.visualization.draw_geometries([pcd])
    res_plot(csv_path)
