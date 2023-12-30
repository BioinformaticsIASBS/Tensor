from tensorflow import Variable, get_static_value, norm
from tensorflow.keras.optimizers import Adam
import numpy as np
import scipy as sp
import io
import time


def optimize(X: np.ndarray, script_D: np.ndarray, script_T: np.ndarray,
             form_no: int, f_size: int, alpha: float, epsilon: float,
             test_indices: np.ndarray, log_file: io.TextIOWrapper) -> np.ndarray:
    I, M = X.shape
    K = script_D.shape[-1]
    L = script_T.shape[-1]
    f = f_size

    D = Variable(np.random.rand(I, f))
    A = Variable(np.random.rand(I, f))
    B = Variable(np.random.rand(K, f))
    T = Variable(np.random.rand(M, f))
    C = Variable(np.random.rand(M, f))
    E = Variable(np.random.rand(L, f))

    I, J, K = script_D.shape
    D1 = script_D.reshape(I, K*I)
    D2 = D1
    D3 = script_D.reshape(K, I*I)

    M, N, L = script_T.shape
    T1 = script_T.reshape(M, L*M)
    T2 = T1
    T3 = script_T.reshape(L, M*M)

    optimizer = Adam(learning_rate=alpha, epsilon=1e-8)
    prev_loss = np.Inf
    converged = False
    
    total_time = 0
    iteration = 0
    t1 = time.time() 
    while not converged:
        if form_no == 1:
            gradient = gradient1(D, T, X)
            optimizer.apply_gradients(zip(gradient, [D, T]))
            
        elif form_no == 2:
            gradient = gradient2(D, A, B, T, D1, D2, D3, X)
            optimizer.apply_gradients(zip(gradient, [D, A, B, T]))
            
        elif form_no == 3:
            gradient = gradient3(D, T, C, E, T1, T2, T3, X)
            optimizer.apply_gradients(zip(gradient, [D, T, C, E]))
            
        else:
            gradient = gradient4(D, A, B, T, C, E, D1, D2, D3, T1, T2, T3, X)
            optimizer.apply_gradients(zip(gradient, [D, A, B, T, C, E]))
            
        loss = loss_function(D, T, X, test_indices)
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
    
    return np.matmul(D, np.transpose(T))
    


def loss_function(D, T, X, test_indices):
    X_hat = np.matmul(D, np.transpose(T))
    X_hat[test_indices] = 0
    
    return get_static_value(0.5 * (norm(X - X_hat, 'fro', axis=[-2,-1]) ** 2))



def gradient1(D, T, X):
    grad_D = np.matmul(np.matmul(D, np.transpose(T)), T) - np.matmul(X, T)
    grad_T = np.matmul(np.matmul(T, np.transpose(D)), D) - np.matmul(np.transpose(X), D)

    return [grad_D, grad_T]



def gradient2(D, A, B, T, D1, D2, D3, X):
    Z1 = sp.linalg.khatri_rao(B, A)
    Z2 = sp.linalg.khatri_rao(B, D)
    Z3 = sp.linalg.khatri_rao(A, D)

    grad_D = ( np.matmul(np.matmul(D, np.transpose(Z1)), Z1) - np.matmul(D1, Z1) ) + \
             ( np.matmul(np.matmul(D, np.transpose(T)), T) - np.matmul(X, T) )
    grad_T = np.matmul(np.matmul(T, np.transpose(D)), D) - np.matmul(np.transpose(X), D)
    grad_A = np.matmul(np.matmul(A, np.transpose(Z2)), Z2) - np.matmul(D2, Z2)
    grad_B = np.matmul(np.matmul(B, np.transpose(Z3)), Z3) - np.matmul(D3, Z3)

    return [grad_D, grad_A, grad_B, grad_T]



def gradient3(D, T, C, E, T1, T2, T3, X):
    Z1 = sp.linalg.khatri_rao(E, C)
    Z2 = sp.linalg.khatri_rao(E, T)
    Z3 = sp.linalg.khatri_rao(C, T)

    grad_D = np.matmul(np.matmul(D, np.transpose(T)), T) - np.matmul(X, T)
    grad_T = ( np.matmul(np.matmul(T, np.transpose(D)), D) - np.matmul(np.transpose(X), D) ) + \
             ( np.matmul(np.matmul(T, np.transpose(Z1)), Z1) - np.matmul(T1, Z1) )
    grad_C = np.matmul(np.matmul(C, np.transpose(Z2)), Z2) - np.matmul(T2, Z2)
    grad_E = np.matmul(np.matmul(E, np.transpose(Z3)), Z3) - np.matmul(T3, Z3)

    return [grad_D, grad_T, grad_C, grad_E]



def gradient4(D, A, B, T, C, E, D1, D2, D3, T1, T2, T3, X):
    Z1 = sp.linalg.khatri_rao(B, A)
    Z2 = sp.linalg.khatri_rao(B, D)
    Z3 = sp.linalg.khatri_rao(A, D)
    Z4 = sp.linalg.khatri_rao(E, C)
    Z5 = sp.linalg.khatri_rao(E, T)
    Z6 = sp.linalg.khatri_rao(C, T)

    grad_D = ( np.matmul(np.matmul(D, np.transpose(Z1)), Z1) - np.matmul(D1, Z1) ) + \
             ( np.matmul(np.matmul(D, np.transpose(T)), T) - np.matmul(X, T) )
    grad_A = np.matmul(np.matmul(A, np.transpose(Z2)), Z2) - np.matmul(D2, Z2)
    grad_B = np.matmul(np.matmul(B, np.transpose(Z3)), Z3) - np.matmul(D3, Z3)
    grad_T = ( np.matmul(np.matmul(T, np.transpose(D)), D) - np.matmul(np.transpose(X), D) ) + \
             ( np.matmul(np.matmul(T, np.transpose(Z4)), Z4) - np.matmul(T1, Z4) )
    grad_C = np.matmul(np.matmul(C, np.transpose(Z5)), Z5) - np.matmul(T2, Z5)
    grad_E = np.matmul(np.matmul(E, np.transpose(Z6)), Z6) - np.matmul(T3, Z6)

    return [grad_D, grad_A, grad_B, grad_T, grad_C, grad_E]
    
