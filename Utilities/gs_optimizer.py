from tensorflow import Variable, get_static_value, norm
from tensorflow.keras.optimizers import Adam
import numpy as np
import io
import time


def optimize(X: np.ndarray, D: np.ndarray, T: np.ndarray,
             f_size: int, alpha: float, epsilon: float,
             test_indices: np.ndarray, log_file: io.TextIOWrapper) -> np.ndarray:
    I, M = X.shape
    f = f_size

    U = Variable(np.random.rand(I, f))
    A = Variable(np.random.rand(f, I))
    V = Variable(np.random.rand(f, M))
    B = Variable(np.random.rand(M, f))

    optimizer = Adam(learning_rate=alpha, epsilon=1e-8)
    prev_loss = np.Inf
    converged = False
    
    total_time = 0
    iteration = 0
    t1 = time.time()
    while not converged:
        grad = gradient(U, A, V, B, X, T, D)
        optimizer.apply_gradients(zip(grad, [U, A, V, B]))
        loss = loss_function(U, V, X, test_indices)
        if np.math.isclose(prev_loss, loss, abs_tol=epsilon):
            converged = True
            
        if log_file and (iteration % 50 == 0 or converged == True):
            t2 = time.time()
            total_time += t2 - t1
            log_file.write('iteration: {}  loss: {:.3f}  time: {:.2f}\n'.format(iteration, loss, t2-t1))
            t1 = time.time()
            
        prev_loss = loss
        iteration += 1
        
    if log_file:
        log_file.write('{} iterations in {:.2f} minutes\n\n\n'.format(iteration, total_time/60))
 
    return np.matmul(U, V)
    


def loss_function(U, V, X, test_indices):
    X_hat = np.matmul(U, V)
    X_hat[test_indices] = 0
    
    return get_static_value(0.5 * (norm(X - X_hat, 'fro', axis=[-2,-1]) ** 2))

def gradient(U, A, V, B, X, T, D):
    grad_U = ( np.matmul(np.matmul(U, A), np.transpose(A)) - np.matmul(T, np.transpose(A)) ) +\
             ( np.matmul(np.matmul(U, V), np.transpose(V)) - np.matmul(X, np.transpose(V)) )
             
    grad_A =  np.matmul(np.matmul(np.transpose(U), U), A) - np.matmul(np.transpose(U), T)
    
    grad_V = ( np.matmul(np.matmul(np.transpose(U), U), V) - np.matmul(np.transpose(U), X) ) +\
             ( np.matmul(np.matmul(np.transpose(B), B), V) - np.matmul(np.transpose(B), D) )

    grad_B = np.matmul(np.matmul(B, V), np.transpose(V)) - np.matmul(D, np.transpose(V))

    return [grad_U, grad_A, grad_V, grad_B]   
