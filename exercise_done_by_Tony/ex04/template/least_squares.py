# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np



def least_squares(y, tx):
    """calculate the least squares solution."""
    I = np.eye(len(tx.T))
    w=np.linalg.solve(tx.T @ tx, tx.T @ y)
    
    return w