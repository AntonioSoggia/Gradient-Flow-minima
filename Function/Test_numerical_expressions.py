import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from Sympy_functions import FunctionSimpy
from numpy.lib.stride_tricks import sliding_window_view
from functools import partial

def theoretical_variances(sigma2, E, D, use_1D):
    func = FunctionSimpy(use_Numeric=True, use_1D=use_1D, subs=0, sigma2_value=sigma2, w0=w0)
    if use_1D:
        params = tuple(E) + tuple(D)
    else:
        params = tuple(E.flatten()) + tuple(D.flatten())
    M = func.evaluate(params, "M")
    return M

def vectorized_pipeline(epsilon, E, D, pool_size, use_1D):
    if use_1D:
        m = len(E)
        m2 = len(D)
        windows = sliding_window_view(epsilon, window_shape=m, axis=1)
        O = np.dot(windows, E)
        Lp = O.shape[1] // pool_size
        A = 0.5 * (O[:, 0:2 * Lp:2] + O[:, 1:2 * Lp:2])
        U = np.zeros_like(O)
        U[:, 0:2 * Lp:2] = A
        # Second convolution: sliding window view with window size m2 on U
        windows_U = sliding_window_view(U, window_shape=m2, axis=1)
        F = np.dot(windows_U, D)
    else:
        windows_E = sliding_window_view(epsilon, window_shape=E.shape, axis=(1, 2))
        O = np.einsum('nxyij,ij->nxy', windows_E, E)
        n_trials, H_O, W_O = O.shape

        new_H = H_O // pool_size
        new_W = W_O // pool_size
        O_trimmed = O[:, :new_H * pool_size, :new_W * pool_size]
        A = O_trimmed.reshape(n_trials, new_H, pool_size, new_W, pool_size).mean(axis=(2, 4))

        U = np.zeros((n_trials, new_H * pool_size, new_W * pool_size))
        U[:, ::pool_size, ::pool_size] = A



        windows_D_full = sliding_window_view(U, window_shape=D.shape, axis=(1, 2))
        H_valid, W_valid = windows_D_full.shape[1], windows_D_full.shape[2]

        idx_i = np.arange(H_valid)
        idx_j = np.arange(W_valid)
        aligned_i = idx_i[(idx_i) % pool_size == 0]
        aligned_j = idx_j[(idx_j) % pool_size == 0]

        F_aligned = windows_D_full[:, aligned_i, :]
        F_aligned = F_aligned[:, :, aligned_j, :, :]
        F = np.einsum('nxyij,ij->nxy', F_aligned, D)
    return F


def vectorized_pipeline_trials(eps, E, D, pool_size, use_1D):
    n_trials = eps.shape[0]
    F_all = np.array([vectorized_pipeline(eps[i], E, D, pool_size, use_1D) for i in range(n_trials)])
    return F_all


def simulate_test(use_1D, L, test_idx):
    rng = np.random.default_rng()

    if use_1D == True:
        E = rng.uniform(-10, 12, size=3)
        D = rng.uniform(-10, 12, size=3)
        eps = rng.normal(0, np.sqrt(sigma2), size=(n_trials, L))
    else:
        E = rng.uniform(-10, 10, size=(3, 3))
        D = rng.uniform(-10, 10, size=(3, 3))
        eps = rng.normal(0, np.sqrt(sigma2), size=(n_trials, *L))
    M = theoretical_variances(sigma2, E, D, use_1D)


    F = vectorized_pipeline(eps, E, D, 2, use_1D)
    emp_var_F = 0
    if margin > 0:
        if use_1D:
            F = F[:, margin:-margin]
            emp_var_F = np.mean(np.var(F, axis=1))
        else:
            F = F[:, margin:-margin, margin:-margin]
            emp_var_F = np.mean(np.var(F, axis=(1, 2)))
    rel_error = np.abs(M - emp_var_F) / M

    return E, D, emp_var_F, rel_error, M


if __name__ == '__main__':
    w0 = 5
    sigma2 = 10
    n_trials = 1800
    n_tests = 50
    margin = 10
    use_1D = False
    if use_1D:
        Leng = 3000
    else:
        Leng = (100, 100)
    test_func = partial(simulate_test, use_1D, Leng)
    with mp.Pool() as pool:
        results = pool.map(test_func, range(n_tests))

    error_F_list = []
    filters_used = []
    for idx, res in enumerate(results):
        E, D, emp_var_F, rel_error, M = res
        filters_used.append((E, D))
        error_F_list.append(100 * rel_error)  # convert to percentage

    mean_error = np.mean(error_F_list)

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, n_tests + 1), error_F_list, marker='o', linestyle='-', label='Relative Error')
    plt.axhline(0, color='black', linewidth=1, linestyle='--', label='Zero Error')
    plt.axhline(mean_error, color='red', linewidth=1, linestyle='--', label=f'Mean Error = {mean_error:.4f}%')
    plt.xlabel("Test")
    plt.ylabel("Variance Relative Error (%)")
    plt.title("Error in Final Variance with Random Filter Pairs")
    plt.legend()
    plt.grid(True)
    plt.savefig("Error_E_D_2.pdf")
    plt.show()