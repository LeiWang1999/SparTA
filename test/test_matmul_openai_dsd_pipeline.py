import os
import time

import torch
import numpy as np

os.sys.path.append(os.getcwd())

from sparta.common import tesa
from sparta import specializer

np.random.seed(2022)

cfg = {
    'GLOBAL_M_VALUE': 1024,
    'GLOBAL_K_VALUE': 256,
    'GLOBAL_N_VALUE': 512,
    # 'GLOBAL_M_VALUE': 1024,
    # 'GLOBAL_K_VALUE': 1024,
    # 'GLOBAL_N_VALUE': 1024,
}

factory = specializer.get_factory('sparse_linear_openai_dsd')

# Prepare Data
def prepare_data():
    A = np.random.uniform(size=(cfg['GLOBAL_M_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)
    B = np.random.uniform(size=(cfg['GLOBAL_K_VALUE'], cfg['GLOBAL_N_VALUE'])).astype(np.float32)
    B_mask = np.random.uniform(size=(
        cfg['GLOBAL_K_VALUE'] // 64,
        cfg['GLOBAL_N_VALUE'] // 32,
    )) < 0.2
    B_tesa = tesa.BCSR(
        dense=B,
        mask=B_mask,
        block_size=(64, 32),
        mode='V'
    ).sparse
    B_val = B_tesa['val']

    B_mask_tiled = np.zeros((cfg['GLOBAL_K_VALUE'], cfg['GLOBAL_N_VALUE']))
    for row_idx in range(B_mask.shape[0]):
        for col_idx in range(B_mask.shape[1]):
            row_start = row_idx * 64
            row_end = row_start + 64
            col_start = col_idx * 32
            col_end = col_start + 32
            B_mask_tiled[row_start:row_end, col_start:col_end] = B_mask[row_idx, col_idx]

    B *= B_mask_tiled
    C_tgt = A @ B
    return A, B, B_val, B_mask, C_tgt

A, B, B_val, B_mask, C_tgt = prepare_data()

# Test Function
test_func = factory.get_test_interface(cfg, mask={'B': B_mask})
print(f'NVCC Latency: {test_func(inputs={"A": A, "B": B}, num_iters=1000)} ms')

# PyTorch Module
module_interface = factory.get_module_interface(cfg, mask={'B': B_mask})
module_code = module_interface.get_module_code()
with open('./test/module.cu', 'w') as f:
    f.write(module_code)

f = module_interface.get_module().forward

device = torch.device(f'cuda:3')
A = torch.from_numpy(A).to(device)
B = torch.from_numpy(B_val).to(device)

for _ in range(10):
    C = f(A, B)
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    C = f(A, B)
torch.cuda.synchronize()
print(f'PyTorch Latency: {(time.time() - start)} ms')

C = C.cpu().numpy()
print(f'Sum_C: {C.sum()}')
print(f'Sum_C_tgt: {C_tgt.sum()}')
print(f'Error: {np.sum(np.abs(C - C_tgt))}')
