import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import PointTransformer, NaivePointTransformer, SimplePointTransformer, PointTransformer_FP
from model import PointTransformer_FPMOD
from model import PointTransformer_FPADV
from dataset import Dales, tald
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import argparse
import os
import pandas as pd
import glob
torch.manual_seed(42)

#Training the model
def train_loop(model, optimizer, loss_fn, loader, device, see_batch_loss = False):
    model.train()
    total_loss = 0
    y_true = []
    y_preds = []
    for batch, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device).squeeze()

        logits = model(data)

        optimizer.zero_grad()

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.view(-1))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        preds = logits.max(dim = -1)[1].view(-1)

        y_true.extend(label.view(-1).cpu().tolist())
        y_preds.extend(preds.detach().cpu().tolist())
        
        if see_batch_loss:
            if batch%100 == 0:
                print(f'Batch_Loss_{batch} : {loss.item()}')

    return total_loss/len(loader), accuracy_score(y_true, y_preds), balanced_accuracy_score(y_true, y_preds)
        
@torch.no_grad()
def test_loop(model, loss_fn, loader, device):
    model.eval()
    total_loss = 0
    y_true = []
    y_preds = []
    for data, label in loader:
        data, label = data.to(device), label.to(device).squeeze()
        # data = data.transpose(1,2)

        logits = model(data)

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.view(-1))
        
        total_loss+=loss.item()
        preds = logits.max(dim = -1)[1].view(-1)
        
        y_true.extend(label.view(-1).cpu().tolist())
        y_preds.extend(preds.detach().cpu().tolist())

    # print(f'val_loss: {total_loss/len(test_loader)}, val_acc: {accuracy_score(y_true, y_preds)}')  
    return total_loss/len(loader), accuracy_score(y_true, y_preds), balanced_accuracy_score(y_true, y_preds), y_preds

class define():
    def __init__(self, model, dataset, args):

        # Setting the device
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Splitting the data
        dataset = dataset[args.dataset]
        dataset = dataset(self.device, args.grid_size, args.points_taken, partition='train')
        print("Dataset Read Complete")
        train_dataset, test_dataset = random_split(dataset, [0.7, 0.3]) 

        # Setting the path for saving the checkpoint       
        self.path = path = os.path.join("checkpoints", f"model_{args.model_name}")

        # Setting Up Loader
        self.train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last=True)

        # Loading the Model
        if args.model == 'PCT_FPMOD' or args.model == 'PCT_FPADV':
            self.model = model[args.model](args.embd, dp = args.dp)
        else:
            self.model = model[args.model](args.embd)

        # loss, Optimizer, Scheduler
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = args.step_size, gamma = 0.6)
        self.model = self.model.to(self.device)

        try:
            if args.load_checkpoint:
                checkpoint = torch.load(os.path.join(path, f"{args.model}_{args.model_name}_{self.recent_epoch()}.pt"))
                self.model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print('Checkpoint Loaded')
        except Exception as e:
            print(e)
            ans = input('Do you want to continue without loading checkpoint (y/n):')
            if ans == 'n':
                quit()
    
    def training(self):
        print("Running Epochs")
        print(f'{self.device = }, {args.grid_size = }, {args.points_taken = }, {args.epoch = }, {args.embd = }, {args.batch_size = }, {args.lr = }, {args.dp = }')
        start = time.time()
        df = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        for _epoch in range(1, args.epoch+1): 
            train_loss, train_acc, bal_avg_acc = train_loop(self.model, self.optimizer, self.loss_fn, self.train_loader, self.device)
            self.scheduler.step()
            if _epoch%args.eval==0:
                val_loss, val_acc, bal_val_acc, _ = test_loop(self.model, self.loss_fn, self.test_loader, self.device)
                df['train_loss'].append(train_loss)
                df['train_acc'].append(train_acc)
                df['val_loss'].append(val_loss)
                df['val_acc'].append(val_acc)
                print(f'Epoch {_epoch} | lr: {self.scheduler.get_last_lr()}:\n train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | bal_train_acc: {bal_avg_acc:.4f}\n val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | bal_val_acc: {bal_val_acc:.4f}')
            
            if _epoch%args.save_freq == 0:
                self.save_progress(df, _epoch)
                df = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        end = time.time()
        print(f'Total_time: {end-start}')
        
        # Saving the results
        if args.epoch%args.save_freq != 0:
            self.save_progress(df, args.epoch)

    def save_progress(self, df, epoch):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        mx = self.recent_epoch()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch' : args.epoch
            }, os.path.join(self.path, f"{args.model}_{args.model_name}_{args.save_freq+mx}.pt"))
        # torch.save(model.state_dict(), os.path.join("checkpoints", f"{args.model}_{args.model_name}.pt"))
        self.save_csv(df, mx)
        print(f" Model Saved at {epoch} epochs, named: {args.model}_{args.model_name}_{args.save_freq+mx}.pt")

    def save_csv(self, df, mx):
        df = pd.DataFrame(df)
        if mx>0:
            mx_file = os.path.join(self.path, f"{args.model}_{args.model_name}_{mx}.csv")
            dfex = pd.read_csv(mx_file)
            df = pd.concat([dfex, df], ignore_index=True)
        df.to_csv(os.path.join(self.path,  f"{args.model}_{args.model_name}_{args.save_freq+mx}.csv"), index=False)

    def recent_epoch(self):
        path = os.path.join("checkpoints", f"model_{args.model_name}")
        mx = 0
        for x in glob.glob(os.path.join(path, f"{args.model}_{args.model_name}_*.csv")):
            n = int(x.split('\\')[-1].split('.')[0].split('_')[-1])
            mx = n if mx<n else mx
        return mx

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--epoch', type = int, default = 10)
    parser.add_argument('--step_size', type = int, default = 100)
    parser.add_argument('--model_name', default = '42')
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--grid_size', type = int, default = 25)
    parser.add_argument('--points_taken', type = int, default = 4096)
    parser.add_argument('--eval', type = int, default = 1)
    parser.add_argument('--embd', type = int, default = 64)
    parser.add_argument('--model', type = str, default = 'PCT_FPADV')
    parser.add_argument('--load_checkpoint', type = bool, default = False)
    parser.add_argument('--dp', type = float, default = 0.5)
    parser.add_argument('--save_freq', type = int, default = 20)
    parser.add_argument('--dataset', type = str, default = 'Dales')
    args = parser.parse_args()

    # Initialize the model
    model = {'NPCT': NaivePointTransformer, 'SPCT': SimplePointTransformer, 'PCT': PointTransformer, 
             'PCT_FP': PointTransformer_FP, 'PCT_FPMOD': PointTransformer_FPMOD, 'PCT_FPADV': PointTransformer_FPADV}
    
    dataset = {'tald': tald, 'Dales': Dales}
    
    model = define(model, dataset, args)
    model.training()
    





