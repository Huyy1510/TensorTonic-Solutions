import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    
    return np.array([val if val >=0 else alpha * val for val in x])
    pass