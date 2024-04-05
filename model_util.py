import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from util.pointnet import sample_and_group
torch.manual_seed(42)


class pointnet(nn.Module):
    def __init__(self, n_embd, dropout):
        super(pointnet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.ll1 = nn.Linear(1024, 512)
        self.ll2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 40) # This 40 is for modelnet40

        self.t1 = T1(dropout)
        self.t2 = T2(dropout)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    
    def forward(self, x):
        
        t1 = self.t1(x)
        x =  torch.bmm(t1, x)

        x = F.relu(self.bn1(self.conv1(x)))
       
        t2 = self.t2(x)
        x = torch.bmm(t2, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        
        x = F.relu(self.bn4(self.ll1(x)))
        x = F.relu(self.bn5(self.ll2(x)))
        x = self.output(x)
        return x
    
class pointnet2cls(nn.Module):
    def __init__(self, n_embd, dropout):
        super(pointnet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.ll1 = nn.Linear(1024, 512)
        self.ll2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 40) # This 40 is for modelnet40

        self.t1 = T1(dropout)
        self.t2 = T2(dropout)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    
    def forward(self, x):
        xyz = x
        
        
        t1 = self.t1(x)
        x =  torch.bmm(t1, x)

        x = sample_and_group(npoint=1024, nsample=32, xyz = xyz, points=x)

        x = F.relu(self.bn1(self.conv1(x)))
       
        t2 = self.t2(x)
        x = torch.bmm(t2, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        
        x = F.relu(self.bn4(self.ll1(x)))
        x = F.relu(self.bn5(self.ll2(x)))
        x = self.output(x)
        return x
    
class pointnet_seg(nn.Module):
    def __init__(self, n_embd, dropout):
        super(pointnet_seg, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.conv4 = nn.Conv1d(1088, 512, kernel_size=1)
        self.conv5 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv6 = nn.Conv1d(256, 128, kernel_size=1)

        self.output = nn.Conv1d(128, 8, kernel_size=1) # This 8 is for Dales


        self.t1 = T1(dropout)
        self.t2 = T2(dropout)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)


    
    def forward(self, x):
        n_points = x.size()[2]
        t1 = self.t1(x)
        x =  torch.bmm(t1, x)

        x = F.relu(self.bn1(self.conv1(x)))
       
        t2 = self.t2(x)
        feature_transformed = torch.bmm(t2, x)

        x = F.relu(self.bn2(self.conv2(feature_transformed)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        x = torch.cat((feature_transformed, x.unsqueeze(-1).repeat(1, 1, n_points)), dim = 1)



        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = self.output(x)
        return x

class T1(nn.Module):
    def __init__(self, dropout):
        super(T1, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.ll1 = nn.Linear(1024, 512)
        self.ll2 = nn.Linear(512, 256)

        self.t1 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.adaptive_max_pool1d(x, 1)

        x = F.relu(self.bn4(self.ll1(x.view(x.size(0), -1))))
        x = F.relu(self.bn5(self.ll2(x)))
        x = self.t1(x) 
        iden = Variable(torch.from_numpy(np.eye(3).flatten().astype(np.float32)))
        if x.is_cuda:
            iden = iden.to('cuda')
        elif x.is_mps:
            iden = iden.to('mps')
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class T2(nn.Module):
    def __init__(self, dropout):
        super(T2, self).__init__()
        self.conv1 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.ll1 = nn.Linear(1024, 512)
        self.ll2 = nn.Linear(512, 256)

        self.t2 = nn.Linear(256, 4096)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.adaptive_max_pool1d(x, 1)

        x = F.relu(self.bn4(self.ll1(x.view(x.size(0), -1))))
        x = F.relu(self.bn5(self.ll2(x)))
        x = self.t2(x) 
        iden = Variable(torch.from_numpy(np.eye(64).flatten().astype(np.float32)))
        if x.is_cuda:
            iden = iden.to('cuda')
        elif x.is_mps:
            iden = iden.to('mps')
        x = x + iden
        x = x.view(-1, 64, 64)
        return x

class pct(nn.Module):
    def __init__(self, n_embd, n_heads, n_layers, n_labels = 8):
        super(pct, self).__init__()
        self.l1 = nn.Linear(3, n_embd)
               
        self.blocks = nn.Sequential(*[blocks(n_embd, n_heads) for _ in range(n_layers)])

        self.logits = nn.Linear(n_embd, n_labels)

    def forward(self, x):
        
        x = self.l1(x)

        x = self.blocks(x)

        logits = self.logits(x)
        
        return logits

class blocks(nn.Module):
    def __init__(self, n_embd, n_heads):
        super(blocks, self).__init__()
    
        head_size = n_embd // n_heads
        self.ln1 = nn.LayerNorm(n_embd)        
        self.multihead = multihead_attention(n_embd, head_size, n_heads)
        
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffw = nn.Sequential(
                    nn.Linear(n_embd, n_embd * 4),
                    nn.ReLU(),
                    nn.Linear(n_embd * 4, n_embd),
        )
    def forward(self, x):
        multihead = x + self.multihead(self.ln1(x))
        return multihead + self.ffw(self.ln2(multihead))

class multihead_attention(nn.Module):
    def __init__(self, n_embd, head_size, n_heads):
        super(multihead_attention, self).__init__()
        self.heads = nn.ModuleList([sa_layer(n_embd, head_size) for _ in range(n_heads)])
        self.ll = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], axis = -1)
        out = self.ll(out)
        
        return out

# Single Head
class sa_layer(nn.Module):
    def __init__(self, n_embd, head_size):
        
        super(sa_layer, self).__init__()
        self.q = nn.Linear(n_embd, head_size, bias = False)
        self.k = nn.Linear(n_embd, head_size, bias = False)
        self.v = nn.Linear(n_embd, head_size, bias = False)

    def forward(self, x):
    
        q = self.q(x)
        k = self.k(x)

        wei = q @ k.transpose(1,2) * (k.size(-1)  ** -0.5)
        wei = F.softmax(wei, dim = -1)

        v = self.v(x)

        out = wei @ v

        return out
    
