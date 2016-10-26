import numpy as np
from scipy.special import expit
from helpers import *

def gradient_descent(y, tx, initial_w, gamma, max_iters, loss_func, grad_func):
    """Gradient descent algorithm."""
    w = initial_w

    for n_iter in range(max_iters):
        grad = grad_func(y, tx, w)
        w -= gamma * grad
        
    return w, loss_func(y, tx, w)

# optimized for batch_size=1, as required...
def stochastic_gradient_descent(y, tx, initial_w, gamma, max_iters, loss_func, grad_func):
    """Stochastic gradient descent algorithm."""
    w, sample_count = initial_w, len(y)

    for n_iter in range(max_iters):
        rnd = np.random.randint(sample_count)
        grad = grad_func(y[rnd : rnd + 1], tx[rnd : rnd + 1], w)
        w -= gamma * grad
            
    return w, loss_func(y, tx, w)

def compute_mse(y, tx, w):
    """Calculate the loss using mse"""
    e = y - tx @ w
    return e.T @ e * 0.5 / len(y)

def compute_mse_gradient(y, tx, w):
    """Compute the gradient of mse"""
    return -tx.T @ (y - tx @ w) / len(y)

def least_squares(y, tx):
    """calculate the least squares solution."""
    tx_t = tx.T
    return np.linalg.solve(tx_t @ tx, tx_t @ y)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm with mse."""
    return gradient_descent(y, tx, initial_w, gamma, max_iters, compute_mse, compute_mse_gradient)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm with mse."""
    return stochastic_gradient_descent(y, tx, initial_w, gamma, max_iters, compute_mse, compute_mse_gradient)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    tx_t = tx.T
    return np.linalg.solve(tx_t @ tx + lambda_ * np.eye(len(tx_t)), tx_t @ y)
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    w, sample_count = initial_w, len(y)
    
    for i in range(max_iters):
        shuffle_indices = np.random.permutation(np.arange(sample_count))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
        i = 0
        while i < sample_count:
            y_rnd, tx_rnd = shuffled_y[i], shuffled_tx[i]
            w -= tx_rnd.T * (gamma * (expit(tx_rnd @ w) - y_rnd))
            i += 1
    
    #gradient_descent(y, tx, initial_w, gamma, max_iters, lambda y, tx, w: 1, lambda y, tx, w: tx.T @ (expit(tx @ w) - y) / len(y))
    tx_w = tx @ w
    return w, np.abs(np.sum(np.log(1 + np.exp(tx_w)) - y * tx_w)) / len(y)
