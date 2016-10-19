import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using mse"""
    e = y - tx @ w
    return e.T @ e * 0.5 / len(y)

def compute_mse_gradient(y, tx, w):
    """Compute the gradient of mse"""
    return -tx.T @ (y - tx @ w) / len(y)

def gradient_descent(y, tx, initial_w, gamma, max_iters, loss_func, grad_func, epsilon = 1e-6):
    """Gradient descent algorithm."""
    w = initial_w

    for n_iter in range(max_iters):
        loss = loss_func(y, tx, w)
        
        if loss < epsilon:
            break
        
        grad = grad_func(y, tx, w)
        w -= gamma * grad
        
    return loss, w

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, loss_func, grad_func, epsilon = 1e-6):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    gen = batch_iter(y, tx, batch_size)
    
    for n_iter in range(max_iters):
        loss = loss_func(y, tx, w)
        
        if loss < epsilon:
            break
        
        yp, txp = next(gen)
        grad = grad_func(yp, txp, w)
        w -= gamma * grad
        
    return loss, w

def least_squares(y, tx):
    """calculate the least squares solution."""
    weight=np.linalg.solve(tx.T @ tx, tx.T @ y)
    return weight

def least_squares_GD(y, tx, initial_w, gamma, max_iters):
    """Gradient descent algorithm with mse."""
    return gradient_descent(y, tx, initial_w, gamma, compute_mse, compute_mse_gradient)

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm with mse."""
    return stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, compute_mse, compute_mse_gradient)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    tx_t = tx.T
    return np.linalg.solve(tx_t @ tx + lambda_ * np.eye(len(tx_t)), tx_t @ y)
