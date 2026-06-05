import numpy as np
def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    X = np.array(X)
    H_in, W_in = X.shape

    H_out = int(((H_in - pool_size)/stride)+1)
    W_out = int(((W_in - pool_size)/stride)+1)

    X_out = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = (i * stride) + pool_size
            w_start = j * stride
            w_end = (j * stride) + pool_size

            window = X[h_start:h_end, w_start:w_end]

            X_out[i,j] = np.max(window)

    return X_out.tolist()