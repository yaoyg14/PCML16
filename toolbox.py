import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.
        
        You can calculate the loss using mse or mae.
        """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    return 1/2/len(y) *(y-tx@ w).T @ (y-tx@ w)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and loss
    # ***************************************************
    
    
    N=len(y)
    
    #coefficient of gradient is -1/N,but coefficient of compute_loss is 1/(2N)
    
    gradient=-1/N * (tx.T)  @ (y-tx@ w)
    return gradient

def least_squares(y, tx):
    """calculate the least squares solution."""
    weight=np.linalg.solve(tx.T @ tx, tx.T @ y)
    return weight

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.
    # ***************************************************
    
    
    stoch_gradient=compute_gradient(y,tx,w)
    loss=compute_loss(y,tx,w)
    return stoch_gradient,loss


def least_squares_GD (y, tx, initial_w, gamma, max_iters):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    w = initial_w
    losses = []
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        
        loss=compute_loss(y,tx,w)
        gradient=compute_gradient(y,tx,w)
        
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        
        w=w-gamma*gradient
        
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
    return losses, ws

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    w=initial_w
    ws=[initial_w]
    losses=[]
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            stoch_gradient,loss=compute_stoch_gradient(minibatch_y,minibatch_tx,w)
        w=w-gamma*stoch_gradient;
        ws.append(np.copy(w))
        losses.append(loss)
    return losses, ws

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    I = np.eye(len(tx.T))
    weight=np.linalg.solve(tx.T @ tx + lambda_*I,tx.T@ y)
    return weight
