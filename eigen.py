import numpy as np
import scipy as sp

def power_iteration(A, e, n):
    iteration_count = 0
    rows, _ = a.shape
    b0 = np.random.rand(rows)
    while iteration_count < 1000:
        c = np.matmul(A, b0)
        c_norm = np.linalg.norm(c)
        b1 = c / c_norm
        b0 = b1
        iteration_count += 1
    print(f"Eigenvector is {b0}")
    b0 = np.array([b0]) # IMPORTANT: The lines are always vertical, meaning the the rows are actually the columns when counting!!!!
    print(f"Attempt multiplication: {np.dot(np.dot(b0, A), b0.transpose())}")
    print(f"Another multiplication: {np.dot(b0, b0.transpose())}")
    print(f"Actual rayleigh: {(np.dot(np.dot(b0, A), b0.transpose()) / np.dot(b0, b0.transpose()))[0][0]}")


def cheat(A):
    return sp.linalg.eig(A)


a = np.array([[-6, 3], [4, 5]])
print(a)
power_iteration(a, 0.001, 100)
print("Correct answers: ")
print(cheat(a))