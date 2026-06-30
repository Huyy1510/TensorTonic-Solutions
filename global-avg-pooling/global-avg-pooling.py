import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.asarray(x)
    if x.ndim not in (3, 4):
        raise ValueError("Input must be 3D or 4D tensor.")
    return np.mean(x, axis=(-2, -1))
    pass