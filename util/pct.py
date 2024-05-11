import torch
import torch.nn as nn
import time
torch.manual_seed(42)

class OA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + torch.sum(attention, dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    
class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 

        energy = energy / x_k.size(1) ** -0.5
        attention = self.softmax(energy)
        x = x_v @ attention # b, c, n 
        x = self.act(x)
        return x
    
class StackedAttention(nn.Module):
    def __init__(self, channels, with_oa):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        if with_oa:
            self.sa = nn.ModuleList([OA_Layer(channels) for _ in range(4)])
        else:
            self.sa = nn.ModuleList([SA_Layer(channels) for _ in range(4)])

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x_concat = None
        for sa in self.sa:
            x = sa(x)
            if x_concat is None:
                x_concat = x
            else:
                x_concat = torch.concat([x_concat, x], dim = 1)
        
        return x_concat

def calc_wtime(model):
    def wrapper(x):   
        start = time.time()
        y = model(x)
        end = time.time()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Size: {y.size()}, Time Taken: {end-start}, Total Param: {total_params}")
    return wrapper

if __name__ == "__main__":
    x = torch.rand(2,64,128) # B, C, N

    model = StackedAttention(channels=64, with_oa=False)
    y = model(x)
    print(y.size())

    model = StackedAttention(channels=64, with_oa = True)
    y = model(x)
    print(y.size())
