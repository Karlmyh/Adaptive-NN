import torch

class MLP(torch.nn.Module):
    def __init__(self,d):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(d,400)
        self.linear2 = torch.nn.Linear(400,20)
        self.linear3 = torch.nn.Linear(20,1)

    def forward(self, x):
        layer1_out = torch.relu(self.linear1(x))
        layer2_out = torch.relu(self.linear2(layer1_out))
        out        = torch.sigmoid(self.linear3(layer2_out))
        return out, layer1_out, layer2_out 
    
    
    
class generate_sample_MLP(torch.nn.Module):
    def __init__(self,d):
        super(generate_sample_MLP, self).__init__()
        torch.manual_seed(1)
        self.linear1 = torch.nn.Linear(d,3*d)
        self.linear2 = torch.nn.Linear(3*d,1)

    def forward(self, x):
        layer1_out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(layer1_out))
        
        return out