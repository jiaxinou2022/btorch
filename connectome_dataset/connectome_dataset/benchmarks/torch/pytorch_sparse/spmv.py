"""SpMV setup helpers — torch-sparse (PyG / torch-geometric)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def make_fn(mat: sp.csr_matrix, *, device: str, batch_size: int):
    """Return a torch-sparse SpMV callable."""
    import torch
    from torch_sparse import SparseTensor

    coo = mat.tocoo().astype(np.float32)
    a = SparseTensor(
        row=torch.tensor(coo.row, device=device, dtype=torch.long),
        col=torch.tensor(coo.col, device=device, dtype=torch.long),
        value=torch.tensor(coo.data, device=device),
        sparse_sizes=mat.shape,
    )
    x = torch.tensor(np.random.rand(mat.shape[0], batch_size).astype(np.float32), device=device)
    return lambda: a.matmul(x)
