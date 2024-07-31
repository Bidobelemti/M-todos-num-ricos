import numpy as np
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

def gauss_seidel(A, b, error, max_iter):
    n = A.shape[0]
    x = np.zeros(n)
    iter = 0
    res = b - np.dot(A, x)
    
    while np.linalg.norm(res) > error and iter < max_iter:
        for k in range(n):
            x[k] = (b[k] - np.dot(A[k, :k], x[:k]) - np.dot(A[k, k+1:], x[k+1:])) / A[k, k]
        
        res = b - np.dot(A, x)
        iter += 1
        print(res, x)
        logging.info(f'\n{res}')
    
    return x, iter

def gauss_jacobi(A, b, error, max_iter):
    n = A.shape[0]
    x = np.zeros(n)
    iter = 0
    res = b - np.dot(A, x)
    
    while np.linalg.norm(res) > error and iter < max_iter:
        x = np.array([ (b[k] - np.dot(A[k, np.arange(n) != k], x[np.arange(n) != k])) / A[k, k] for k in np.arange(n) ])
        res = b - np.dot(A, x)
        iter += 1
        logging.info(f'\n{res}')

    return x, iter

