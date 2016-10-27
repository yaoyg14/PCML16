# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x

def batch_iter(y, tx, batch_size, num_batches, shuffle=True):
    """
    Modified!
    Will yield in total num_batches batches of size batch_size.
    If num_batches is bigger than the size of the dataset, if will wrap around it.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size % data_size
        end_index = (start_index + batch_size) % data_size
        if end_index <= start_index:
            r = range(batch_num * batch_size, (batch_num + 1) * batch_size)
            yield shuffled_y.take(r, axis = 0, mode = 'wrap'), shuffled_tx.take(r, axis = 0, mode = 'wrap')
        else:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
