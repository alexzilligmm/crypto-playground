import torch
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from functools import partial

# torch utils

def apply_transform(batch, transform):
    images = batch["image"]
    batch["image"] = [transform(img) for img in images]
    return batch

def collate_fn(batch):

    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]

    images = torch.stack(images)

    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels

def get_apply_transform(transform):
    return lambda x: apply_transform(x, transform)

# crypto utils

def encrypt_row(row, public_key):
    """Encrypt a single row of the matrix."""
    return [public_key.encrypt(x.item()) for x in row]

def encrypt_matrix_parallel(a, public_key):
    if hasattr(a, 'detach'):
        a = a.detach().cpu().numpy()
        
    num_cpus = min(mp.cpu_count() - 1, 8)

    encrypted_matrix = []

    worker_func = partial(encrypt_row, public_key=public_key)

    with mp.Pool(num_cpus) as pool:
        for encrypted_row in tqdm.tqdm(pool.imap(worker_func, a),
                                       total=a.shape[0]):
            encrypted_matrix.append(encrypted_row)

    return encrypted_matrix

# plot utils

def plot_encrypted_matrix(enc_matrix):

    cipher_matrix = [
        [num.ciphertext() for num in row] 
        for row in enc_matrix
    ]

    img_data = np.array(
        [[val % 255 for val in row] for row in cipher_matrix], 
        dtype=float
    )

    # 3. Plot
    plt.figure(figsize=(2, 2))
    plt.imshow(img_data, cmap='gray', vmin=0, vmax=255)
    plt.title("Ciphered image (mod 255)")
    plt.axis('off')
    plt.show()