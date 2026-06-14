from Subs import *
from functools import lru_cache
from collections import defaultdict
from sympy.polys.polytools import count_roots

def get_grad_vars(D, noise):
    if noise:
        if D:
            return E_symbols + D_symbols
        else:
            return E_sym + D_sym
    else:
        if D:
            return E_symbols + D_symbols
        else:
            return [a, b, c, d, e]

def get_x_vars(D, noise):
    all_vars = (x1, x2, x3, x4, x5, x6, x7, x8, x9,
                x10, x11, x12, x13, x14, x15, x16, x17, x18)
    if noise:
        return all_vars[:18] if D else all_vars[:6]
    else:
        return all_vars[:18] if D else all_vars[:5]

def determine_subs_key(noise, D, w, comb):
    if noise:
        if D:
            if comb:
                return "noise_D_combined"
            else:
                return "noise_D_odd" if w % 2 == 0 else "noise_D_even"
        else:
            if comb:
                return "noise_noD_mean"
            else:
                return "noise_noD_even" if w % 2 == 1 else "noise_noD_odd"
    else:
        if D:
            if comb:
                return "no_noise_D_combined"
            else:
                 return "no_noise_D_even" if w % 2 == 1 else "no_noise_D_odd"
        else:
            return "no_noise_noD_even" if w % 2 == 1 else "no_noise_noD_odd"


def get_subs(noise, D, w, comb):
    key = determine_subs_key(noise, D, w, comb)
    return subs_mapping[key]

def negative_gradient(f_expr, D, noise):
    vars_original = get_grad_vars(D, noise)
    grad = [sp.diff(f_expr, var) for var in vars_original]
    return [-g for g in grad]

@lru_cache(maxsize=None)
def substitute_and_simplify(expr, D, noise, comb, w):
    base_subs, opt_subs = get_subs(noise, D, w, comb)
    expr = expr.subs(base_subs).subs(opt_subs)
    expr = sp.cancel(expr)
    expr = sp.factor(expr)
    expr = sp.factor_terms(expr)
    expr = sp.collect(expr, omicron)
    return expr

def linearize(expr, D, noise):
    x_vars = get_x_vars(D, noise)
    grads_at_zero = [
        sp.diff(expr, xi).subs({xi: 0 for xi in x_vars})
        for xi in x_vars
    ]
    return sum(g0 * xi for g0, xi in zip(grads_at_zero, x_vars))

def linearize_vector(vec, D, noise):
    return [linearize(expr, D, noise) for expr in vec]


def jacobian_matrix(vec, D, noise):
    # build the raw Jacobian
    nonzero_vec  = [expr for expr in vec if expr != 0]
    x_vars       = list(get_x_vars(D, noise))
    active_vars  = [v for v in x_vars if any(v in e.free_symbols for e in nonzero_vec)]
    J            = sp.Matrix([[sp.diff(e, v) for v in active_vars] for e in nonzero_vec])

    # substitute x_i → 0
    J0 = J.subs({v: 0 for v in active_vars})

    # now clean up entry‑wise
    def clean(e):
        e = sp.cancel(e)
        e = sp.factor(e)
        e = sp.factor_terms(e)
        e = sp.collect(e, omicron)

        subs_map = {
            (alpha + delta + eta) * omicron: 1,
            omicron * (alpha + delta + eta): 1,
            -(alpha + delta + eta) * omicron: -1,
            -omicron * (alpha + delta + eta): -1,

            (alpha + delta + eta) ** 2: 1 / omicron ** 2,
            (alpha + delta + eta) ** 2 / 2: 1 / (2 * omicron ** 2),
        }
        e = e.subs(subs_map)

        return e
    if D and not noise:
        J0 = J0.applyfunc(clean)
    return J0

def gershgorin(A):
    M = -A
    n = A.rows
    for i in range(n):
        R = sum(abs(M[i,j]) for j in range(n) if j!=i)
        gap = sp.simplify(M[i,i] - R)
        print(f"Row {i}: Mii−Ri =", gap)
        # check gap>0

def extract_blocks_2x2(A: sp.Matrix, row_blocks=None, col_blocks=None):
    """
    Split A into B11, B12, B21, B22 via two index-lists each.
    Default: first/second half of rows and cols.
    """
    n = A.rows
    if row_blocks is None or col_blocks is None:
        half = n // 2
        row_blocks = [list(range(half)), list(range(half, n))]
        col_blocks = row_blocks
    B11 = A.extract(row_blocks[0], col_blocks[0])
    B12 = A.extract(row_blocks[0], col_blocks[1])
    B21 = A.extract(row_blocks[1], col_blocks[0])
    B22 = A.extract(row_blocks[1], col_blocks[1])
    return B11, B12, B21, B22

# --- 2) Equitable partition for a block ---
def find_rowcol_classes(A: sp.Matrix):
    n = A.rows
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    for i in range(n):
        for j in range(i+1, n):
            if (A.row(i) - A.row(j)).is_zero_matrix and (A.col(i) - A.col(j)).is_zero_matrix:
                union(i, j)
    groups = defaultdict(list)
    for i in range(n): groups[find(i)].append(i)
    return list(groups.values())


def equitable_quotient(A: sp.Matrix):
    classes = find_rowcol_classes(A)
    k = len(classes)
    Q = sp.zeros(k)
    for p, Cp in enumerate(classes):
        for q, Cq in enumerate(classes):
            Q[p, q] = sum(A[Cp[0], j] for j in Cq)
    return Q, classes

def compress_to_tildeA(A: sp.Matrix, row_blocks=None, col_blocks=None):
    """
    1) Extract B11, B12, B21, B22
    2) Equitably reduce B11 -> (Q1, C1) and B22 -> (Q2, C2)
    3) Compress off-diagonal once: S12, then S21 = S12.T
    4) Assemble tildeA = [[Q1, S12], [S12.T, Q2]]
    Returns tildeA, Q1, C1, Q2, C2, S12
    """
    B11, B12, _, B22 = extract_blocks_2x2(A, row_blocks, col_blocks)
    Q1, C1 = equitable_quotient(B11)
    Q2, C2 = equitable_quotient(B22)
    k1, k2 = len(C1), len(C2)
    S12 = sp.zeros(k1, k2)
    for i, Ci in enumerate(C1):
        for j, Cj in enumerate(C2):
            S12[i, j] = sum(B12[a, b] for a in Ci for b in Cj)
    tildeA = sp.BlockMatrix([[Q1, S12], [S12.T, Q2]]).as_explicit()
    return tildeA, Q1, C1, Q2, C2, S12


def sign_for_zero_negative(x):
    """
    Return +1 if x>0, −1 if x<0 or x==0.
    """
    s = sp.sign(x)
    return sp.Piecewise((1, x > 0), (-1, True))


def sign_variations_modified(values):
    """
    Count the number of sign changes in `values`, where each value v
    is first mapped to +1 if v>0, and −1 if v<=0.
    """
    signs = [sign_for_zero_negative(v) for v in values]
    # now count adjacent changes
    changes = 0
    for a, b in zip(signs, signs[1:]):
        # since these are ±1, a≠b exactly when they differ
        changes += sp.Abs(sp.simplify((a - b) / 2))  # (1 - (-1))/2 = 1, etc.
    return changes

lam = sp.symbols("lam")


def has_positive_eigenvalue_sturm(M):
    """
    Return True iff the real‐symmetric matrix M has at least one
    strictly positive eigenvalue, via an early‐exit Sturm‐sequence test.
    """
    lam = sp.symbols('lam', real=True)

    # 1) build and monic‐normalize the characteristic poly P0(λ)
    n = M.shape[0]
    P0 = sp.Poly(sp.expand(sp.det(lam*sp.eye(n) - M)), lam)
    # ensure leading coeff = 1
    lc = P0.LC()
    P0 = sp.Poly(P0.as_expr()/lc, lam)

    # 2) build the sequence P[0], P[1] = P0', then remainders
    seq = [P0, P0.diff(lam)]
    # track values at 0:
    def sign(x):
        return sp.sign(x).subs({sp.sign(0): -1})  # treat 0 as “non-positive”
    vals0 = [sign(seq[0].eval(lam, 0)), sign(seq[1].eval(lam, 0))]
    # count variations:
    def variations(xs):
        return sum(1 for i in range(len(xs)-1) if xs[i] != xs[i+1])
    V0 = variations(vals0)
    # at +∞, signs are just signs of leading coeffs (all 1 or negative if we normalized)
    signs_inf = [1, sp.sign(seq[1].all_coeffs()[0])]  # but P1’s leading coeff may be >0 or <0
    Vinf = variations(signs_inf)

    # early-exit: if V0>Vinf, there is already ≥1 positive root
    if V0 > Vinf:
        return True

    # otherwise continue building the sequence until it ends or we detect V0>Vinf
    while True:
        rem_poly = -sp.rem(seq[-2], seq[-1], lam)
        if rem_poly.is_zero:
            break
        Pi = sp.Poly(rem_poly.as_expr() / rem_poly.LC(), lam)
        seq.append(Pi)
        si = sign(Pi.eval(lam, 0))
        # update V0 incrementally
        last_sign = vals0[-1]
        if si != last_sign:
            V0 += 1
        vals0.append(si)
        # at +∞, Pi.leading coeff always +1 after normalization:
        # so Vinf stays the same if last sign was also +1
        # (we only normalized by positive LC)
        # thus no need to update Vinf
        if V0 > Vinf:
            return True

    # final check
    return (V0 > Vinf)