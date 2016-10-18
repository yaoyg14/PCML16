
import numpy as np

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    I = np.eye(len(tx.T))

    weight=np.linalg.solve(tx.T @ tx + lamb*I,tx.T@ y)
    
    return weight
