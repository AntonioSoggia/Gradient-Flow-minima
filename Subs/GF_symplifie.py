from GF_symplifie_helper import *
from Function.Sympy_functions import FunctionSimpy
import sympy as sp
import mpmath
import numpy as np
import matplotlib.pyplot as plt

def main(D=True, noise=True, comb=True):
    w = 6
    w_odd = 6
    w_even = 5
    use_1D = not D

    subs_value = 0 if noise or D else 1

    func = FunctionSimpy(use_Numeric=False, use_1D=use_1D, subs=subs_value,
                         sigma2_value=sigma2, w0=w, combined=comb, w0_even = w_even, w0_odd=w_odd)

    f_expr = func.expr if noise else func.expr_L

    print("Configuration:")
    print("2D mode:", D, "(use_1D =", use_1D, ")")
    print("Noise mode:", noise, "(subs =", subs_value, ")")
    print("w0 =", w)
    print("\nSelected Expression:")
    print(f"\n{determine_subs_key(noise, D, w, comb)}")
    sp.pretty_print(f_expr)

    return f_expr, D, noise, w

def sympy_to_mpmath_matrix(sympy_mat, omicron_val):
    subs_expr = sympy_mat.subs(omicron, mpmath.mpf(omicron_val)).evalf()
    rows, cols = subs_expr.shape
    mp_mat = mpmath.matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            mp_mat[i, j] = mpmath.mpf(str(subs_expr[i, j]))
    return mp_mat

if __name__ == "__main__":
    mpmath.mp.dps = 50

    comb = True
    noise = True
    D = True
    f_expr, D, noise, w = main(D=D, noise=noise, comb=comb)

    neg_grad_expr = negative_gradient(f_expr, D, noise)
    print("\nNegative Gradient computed.")

    neg_grad_subs = [
        substitute_and_simplify(expr, D, noise, comb, w)
        for expr in neg_grad_expr
    ]

    neg_grad_lin = linearize_vector(neg_grad_subs, D, noise)
    print("Linearizing ok")

    A = jacobian_matrix(neg_grad_lin, D, noise)
    print("\nJacobian matrix A:")
    print("-" + sp.latex(A))
 #   _ = gershgorin(A)
  #  tildeA, Q1, C1, Q2, C2, S12 = compress_to_tildeA(A)
  #  print(sp.latex(tildeA))
    pos_roots = has_positive_eigenvalue_sturm(A)
    print(sp.latex(pos_roots))

 #   neg_vals = np.linspace(-100, -0.1, 101)
 #   pos_vals = np.linspace(0.1, 100, 101)
 #   omicron_vals = np.concatenate((neg_vals, pos_vals))
#
 #   max_eigs = []

#   for val in omicron_vals:
#       J_mp = sympy_to_mpmath_matrix(A, val)
#       eigs = mpmath.eig(J_mp, left=False, right=False)
#       eigs_real = [mpmath.re(e) for e in eigs]
#       max_eigs.append(max(eigs_real))

#   plt.figure()
#   plt.plot(omicron_vals, max_eigs)
#   plt.axhline(0, color='black', linestyle='--')
#   plt.xlabel("omicron")
#   plt.ylabel("Max eigenvalue")
#   plt.title("Max Eigenvalue with 50 precision digits")
#   plt.grid(True)
#   plt.show()
