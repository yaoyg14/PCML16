import numpy as np
from scipy.special import expit
from helpers import *

def gradient_descent(y, tx, initial_w, gamma, max_iters, loss_func, grad_func):
    """Gradient descent algorithm."""
    w = initial_w

    for n_iter in range(max_iters):
        grad = grad_func(y, tx, w)
        w -= gamma * grad
        
    return loss_func(y, tx, w), w

def stochastic_gradient_descent(y, tx, initial_w, gamma, max_iters, batch_size, loss_func, grad_func):
    """Stochastic gradient descent algorithm."""
    w = initial_w

    for mini_y, mini_tx in batch_iter(y, tx, batch_size, max_iters):
        grad = grad_func(mini_y, mini_tx, w)
        w -= gamma * grad
            
    return loss_func(y, tx, w), w

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

def least_squares_GD(y, tx, gamma, max_iters):
    """Gradient descent algorithm with mse."""
    return gradient_descent(y, tx, np.zeros(tx.shape[1]), gamma, max_iters, compute_mse, compute_mse_gradient)

def least_squares_SGD(y, tx, gamma, max_iters, batch_size):
    """Stochastic gradient descent algorithm with mse."""
    return stochastic_gradient_descent(y, tx, np.zeros(tx.shape[1]), gamma, max_iters, batch_size, compute_mse, compute_mse_gradient)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    tx_t = tx.T
    return np.linalg.solve(tx_t @ tx + lambda_ * np.eye(len(tx_t)), tx_t @ y)
    
def logistic_regression(y, tx, gamma, max_iters):
    """Logistic regression"""
    def loss(y, tx, w):
        #txn_t_x_w = tx @ w
        #return np.abs(np.sum(np.log(1 + np.exp(txn_t_x_w)) - y * txn_t_x_w)) / len(y)
        return 1
    
    def grad(y, tx, w):
        return (tx.T @ (expit(tx @ w) - y)-2*0.022*w) / len(y)
    
    return stochastic_gradient_descent(y, tx, np.zeros(tx.shape[1]), gamma, max_iters,1000, loss, grad)

def reg_logistic_regression(y, tx, gamma, max_iters):
    """Logistic Ridge regression"""
    def loss(y, tx, w):
        #txn_t_x_w = tx @ 
        #return np.abs(np.sum(np.log(1 + np.exp(txn_t_x_w)) - y * txn_t_x_w)) / len(y)
        return 1
    
    def grad(y, tx, w):
        return (tx.T @ (expit(tx @ w) - y)-2*0.1*w) / len(y)   #lambda_=0.1
    
    return stochastic_gradient_descent(y, tx, np.zeros(tx.shape[1]), gamma, max_iters,1000, loss, grad)