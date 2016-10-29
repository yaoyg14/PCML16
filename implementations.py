import numpy as np
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

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    w, sample_count, batch_size = initial_w, len(y), 1000
    batch_count = int(sample_count / batch_size) * max_iters

    fact = gamma / batch_size
    for mini_y, mini_tx in batch_iter(y, tx, batch_size, batch_count, shuffle = False):
        grad = mini_tx.T @ (sigmoid(mini_tx @ w) - mini_y)
        w -= fact * grad
    
    tx_w = tx @ w
    return w, np.abs(np.sum(np.log(1 + np.exp(tx_w)) - y * tx_w)) / len(y)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression"""
    if lambda_ == 0: # small optimization
        return logistic_regression(y, tx, initial_w, max_iters, gamma)
    
    w, sample_count, batch_size = initial_w, len(y), 1000
    batch_count = int(sample_count / batch_size) * max_iters

    fact = gamma / batch_size
    for mini_y, mini_tx in batch_iter(y, tx, batch_size, batch_count, shuffle = False):
        grad = mini_tx.T @ (sigmoid(mini_tx @ w) - mini_y) - 2 * lambda_ * w
        w -= fact * grad
    
    tx_w = tx @ w
    return w, np.abs(np.sum(np.log(1 + np.exp(tx_w)) - y * tx_w)) / len(y)