import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(784, 128)  
        self.layer_2 = nn.Linear(128, 64)  
        self.layer_3 = nn.Linear(64, 10)    
        
        self.Relu = nn.ReLU() 

    def forward(self, x):
        x = x.view(-1, 28*28)  
        x = self.Relu(self.layer_1(x))  
        x = self.Relu(self.layer_2(x))  
        x = self.layer_3(x) 
        
        return x