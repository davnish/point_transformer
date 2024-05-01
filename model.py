import torch
import torch.nn as nn
from util.pct import StackedAttention
from util.pointnet import sample_and_group_all, Local_op
from util.pointnet import PointNetFeaturePropagation, sample_and_group
import torch.nn.functional as F

torch.manual_seed(42)


class NaivePointTransformer(nn.Module):
    def __init__(self, embd = 64, with_oa = False):
        super().__init__()
        output_channels = 8
        d_points = 3
        self.conv1 = nn.Conv1d(d_points, embd, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(embd, embd, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(embd)
        self.bn2 = nn.BatchNorm1d(embd)
        
        self.StackedAttention = StackedAttention(channels = embd, with_oa = with_oa)
        
        self.conv5 = nn.Conv1d(embd*4*2, embd*2, 1)
        self.bn5 = nn.BatchNorm1d(embd*2)
        self.dp5 = nn.Dropout(p=0.2)
        self.conv6 = nn.Conv1d(embd*2, embd*2, 1)
        self.bn6 = nn.BatchNorm1d(embd*2)

        self.logits = nn.Conv1d(embd*2, output_channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N

        x = self.StackedAttention(x)  
        x1 = torch.max(x, 2)[0].unsqueeze(dim = -1).repeat(1, 1, x.size(2)) # Global features

        x = torch.cat([x, x1], dim = 1)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dp5(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.logits(x)
        x = x.permute(0, 2, 1)
        return x

class SimplePointTransformer(nn.Module):
    def __init__(self, embd = 64, with_oa = True):
        super().__init__()
        output_channels = 8
        d_points = 3
        self.conv1 = nn.Conv1d(d_points, embd, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(embd, embd, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(embd)
        self.bn2 = nn.BatchNorm1d(embd)
        
        self.StackedAttention = StackedAttention(channels = embd, with_oa = with_oa)
        
        self.conv5 = nn.Conv1d(embd*4*2, embd*2, 1)
        self.bn5 = nn.BatchNorm1d(embd*2)
        self.dp5 = nn.Dropout(p=0.2)
        self.conv6 = nn.Conv1d(embd*2, embd*2, 1)
        self.bn6 = nn.BatchNorm1d(embd*2)

        self.logits = nn.Conv1d(embd*2, output_channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N

        x = self.StackedAttention(x)  
        x1 = torch.max(x, 2)[0].unsqueeze(dim = -1).repeat(1, 1, x.size(2)) # Global features

        x = torch.cat([x, x1], dim = 1)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dp5(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.logits(x)
        x = x.permute(0, 2, 1)
        return x

class PointTransformer(nn.Module):
    def __init__(self, embd = 64, with_oa = True):
        super().__init__()
        output_channels = 8
        d_points = 3
        self.conv1 = nn.Conv1d(d_points, embd, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(embd, embd, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(embd)
        self.bn2 = nn.BatchNorm1d(embd)

        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)
        
        self.gather_local_0 = Local_op(in_channels = embd*2, out_channels = embd*2)
        self.gather_local_1 = Local_op(in_channels = embd*4, out_channels = embd*4)
       
        self.pt_last = StackedAttention(channels = embd*4, with_oa = with_oa)
        self.dp_pt = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()

        self.conv_fuse = nn.Sequential(nn.Conv1d(embd*4*4 + embd*4 + embd*2, embd*4*4, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(embd*4*4),
                                   nn.LeakyReLU(negative_slope=0.2))


        self.conv3 = nn.Conv1d(embd*4*4*2, embd*4*4, 1)
        self.bn3 = nn.BatchNorm1d(embd*4*4)
        self.dp3 = nn.Dropout(p=0.2)        
        
        self.conv4 = nn.Conv1d(embd*4*4, embd*4*2, 1)
        self.bn4 = nn.BatchNorm1d(embd*4*2)
        self.dp4 = nn.Dropout(p=0.2)
        
        self.conv5 = nn.Conv1d(embd*4*2, embd*4, 1)
        self.bn5 = nn.BatchNorm1d(embd*4)
        self.dp5 = nn.Dropout(p=0.2)
        
        self.conv6 = nn.Conv1d(embd*4, embd*2, 1)
        self.bn6 = nn.BatchNorm1d(embd*2)
        
        self.conv7 = nn.Conv1d(embd*2, embd, 1)
        self.bn7 = nn.BatchNorm1d(embd)

        self.logits = nn.Conv1d(embd, output_channels, 1)


    def forward(self, x):
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N

        x = x.permute(0, 2, 1)

        new_xyz, new_feature = sample_and_group_all(nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)

        new_xyz, new_feature = sample_and_group_all(nsample=32, xyz=new_xyz, points=feature_0.permute(0, 2, 1)) 
        feature_1 = self.gather_local_1(new_feature) # B, C, N

        x = self.pt_last(feature_1)
        
        x = torch.concat([x, feature_1, feature_0], dim=1)

        x = self.conv_fuse(x)
        
        x1 = torch.max(x, 2)[0].unsqueeze(dim = -1).repeat(1, 1, x.size(2)) # Global features

        x = torch.cat([x, x1], dim = 1)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dp3(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dp4(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.dp5(x)

        x = self.relu(self.bn6(self.conv6(x + feature_1)))

        x = self.relu(self.bn7(self.conv7(x + feature_0)))

        x = self.logits(x)
        
        x = x.permute(0, 2, 1)
        return x
    
class PointTransformer_fp(nn.Module):
    def __init__(self, n_embd = 64, with_oa = True):
        super().__init__()
        output_channels = 8
        d_points = 3
        self.conv1 = nn.Conv1d(d_points, n_embd, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(n_embd, n_embd, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(n_embd)
        self.bn2 = nn.BatchNorm1d(n_embd)

        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)
        
        self.gather_local_1 = Local_op(in_channels = n_embd*2, out_channels = n_embd*2)
        self.gather_local_2 = Local_op(in_channels = n_embd*4, out_channels = n_embd*4)
       
        self.pt_last = StackedAttention(channels = n_embd*4, with_oa = with_oa)
        self.dp_pt = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()

        self.conv_fuse = nn.Sequential(nn.Conv1d(n_embd*4*4*2, n_embd*4*4*2, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(n_embd*4*4*2),
                                   nn.LeakyReLU(negative_slope=0.2))


        self.fp1 = PointNetFeaturePropagation(in_channel=(n_embd*4*4*2 + n_embd*2), mlp=[n_embd*4*4, n_embd*4*2])
        self.fp2 = PointNetFeaturePropagation(in_channel=(n_embd*4*2 + n_embd), mlp=[n_embd*4, n_embd*2])
        
        # self.conv3 = nn.Conv1d(n_embd*4*4*2, n_embd*4*4, 1)
        # self.bn3 = nn.BatchNorm1d(n_embd*4*4)
        # self.dp3 = nn.Dropout(p=0.2)        
        
        # self.conv4 = nn.Conv1d(n_embd*4*4, n_embd*4*2, 1)
        # self.bn4 = nn.BatchNorm1d(n_embd*4*2)
        # self.dp4 = nn.Dropout(p=0.2)
        
        # self.conv5 = nn.Conv1d(n_embd*4*2, n_embd*4, 1)
        # self.bn5 = nn.BatchNorm1d(n_embd*4)
        # self.dp5 = nn.Dropout(p=0.2)
        
        self.conv6 = nn.Conv1d(n_embd*2, n_embd, 1)
        self.bn6 = nn.BatchNorm1d(n_embd)
        
        self.conv7 = nn.Conv1d(n_embd, n_embd//2, 1)
        self.bn7 = nn.BatchNorm1d(n_embd//2)

        self.logits = nn.Conv1d(n_embd//2, output_channels, 1)


    def forward(self, x):
        N = x.size(1)
        xyz1 = x
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N

        feature_0 = x

        xyz2, new_feature = sample_and_group(npoint=N//2, nsample=32, xyz=xyz1, points=x.permute(0, 2, 1))         
        feature_1 = self.gather_local_1(new_feature)

        xyz3, new_feature = sample_and_group(npoint=N//4, nsample=32, xyz=xyz2, points=feature_1.permute(0, 2, 1)) 
        feature_2 = self.gather_local_2(new_feature) # B, C, N

        x = self.pt_last(feature_2)
        
        # x = torch.concat([x, feature_1, feature_0], dim=1)

        
        x1 = torch.max(x, 2)[0].unsqueeze(dim = -1).repeat(1, 1, x.size(2)) # Global features

        x = torch.cat([x, x1], dim = 1)
 
        x = self.conv_fuse(x)


        x = self.fp1(xyz2.transpose(1,2), xyz3.transpose(1,2), feature_1, x)
        x = self.fp2(xyz1.transpose(1,2), xyz2.transpose(1,2), feature_0, x)

        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x + feature_0)))

        x = self.logits(x)
        
        x = x.permute(0, 2, 1)
        return x
    

    
    
if __name__ == '__main__':

    x = torch.rand((8,128,3))

    model = NaivePointTransformer(embd=64)
    y = model(x)
    print(y.size())

    model = SimplePointTransformer(embd=64)
    y = model(x)
    print(y.size())

    model = PointTransformer(embd=64, with_oa=False)
    y = model(x)
    print(y.size())
    pass