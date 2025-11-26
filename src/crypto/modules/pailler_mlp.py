from functools import partial
import torch.nn as nn
import multiprocessing as mp
from tqdm import tqdm

def h_mul(row, x):
    assert len(x) == row.shape[0]
    return sum(w_val * x_val for w_val, x_val in zip(row, x))

def parallel_mul(w, x, layer="Performing mul"):
    assert len(w.shape) == 2
    w = w.detach().cpu().numpy()
    num_cpus = min(28, mp.cpu_count()) 
    
    results = []
    func = partial(h_mul, x=x)

    with mp.Pool(num_cpus) as pool:
        for result in tqdm(pool.imap(func, w), total=w.shape[0], desc=layer):
            results.append(result)

    return results

def h_add(vector, bias):
    assert len(vector) == len(bias)
    new_vector = []
    for v_val, b_val in zip(vector, bias):
        new_vector.append(v_val + b_val)
    return new_vector

def h_relu(vector):
    new_vector = []
    
    for x in vector:
        val = x * x 
        new_vector.append(val)
        
    return new_vector

class PaillerMLP(nn.Module):
    def __init__(self, mlp) -> None:
        super().__init__()
        self.mlp = mlp
        
    def forward(self, input):
        x = parallel_mul(self.mlp.layer_1.weight, input, layer="Layer 1 Mul")
        x = h_add(x, self.mlp.layer_1.bias)
        x = h_relu(x)

        x = parallel_mul(self.mlp.layer_2.weight, x, layer="Layer 2 Mul")
        x = h_add(x, self.mlp.layer_2.bias)
        x = h_relu(x)

        x = parallel_mul(self.mlp.layer_3.weight, x, layer="Layer 3 Mul")
        x = h_add(x, self.mlp.layer_3.bias)
        
        return x
    
        