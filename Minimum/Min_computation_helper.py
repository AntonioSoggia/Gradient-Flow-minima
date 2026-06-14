from sympy import pretty_print
from Min_subs import *
from sympy.matrices.common import NonPositiveDefiniteMatrixError, NonInvertibleMatrixError
import sympy as sp

def treat_exp(expr):

    if isinstance(expr, sp.MatrixBase):
        return expr.applyfunc(treat_exp)
    expr = sp.together(expr)
    expr = sp.cancel(expr)
    expr = sp.factor_terms(expr, radical=True)
    expr = sp.factor(expr)
    expr = sp.powsimp(expr, force=True)
    expr = sp.simplify(expr)
    for sym in list(expr.free_symbols):
        expr = sp.collect(expr, sym)
    return expr



def solve_for_D(grad_vec, D_vars, free_values=None, default_free=0):
    if free_values is None:
        free_values = {}

    # pick out the D-equations
    eqs_D = list(grad_vec)[-len(D_vars):]
    M, v = sp.linear_eq_to_matrix(eqs_D, D_vars)
    n = len(D_vars)

    # build connectivity graph so we can solve by blocks
    var_graph = {i: set() for i in range(n)}
    for i in range(len(eqs_D)):
        nz = [j for j in range(n) if not M[i, j].equals(0)]
        for a in nz:
            for b in nz:
                var_graph[a].add(b)
                var_graph[b].add(a)

    visited = set()
    components = []
    for i in range(n):
        if i in visited:
            continue
        stack, comp = [i], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(var_graph[u] - comp)
        visited |= comp
        components.append(sorted(comp))

    sol = {}
    for comp in components:
        rows = [r for r in range(len(eqs_D))
                if any(not M[r, j].equals(0) for j in comp)]

        # no equations ⇒ everything free/default
        if not rows:
            for j in comp:
                sol[D_vars[j]] = free_values.get(D_vars[j], default_free)
            continue

        to_free  = [j for j in comp if D_vars[j] in free_values]
        to_solve = [j for j in comp if D_vars[j] not in free_values]

        M_u = M.extract(rows, to_solve)
        v_c = sp.Matrix([v[r, 0] for r in rows])

        # if there are free vars, build M_f and f_vec
        if to_free:
            M_f   = M.extract(rows, to_free)
            f_vec = sp.Matrix([free_values[D_vars[j]] for j in to_free])
            # correct RHS: M_u * u = v_c - M_f * f_vec
            rhs = v_c - M_f * f_vec
        else:
            # no free variables ⇒ M_f f = 0
            rhs = v_c

        # solve for the unknowns in this block
        if to_solve:
            try:
                u_sol = M_u.LUsolve(rhs)
            except NonInvertibleMatrixError:
                vars_u = [D_vars[j] for j in to_solve]
                sol_set = sp.linsolve((M_u, rhs), *vars_u)
                if not sol_set:
                    raise ValueError(f"Inconsistent block {vars_u}")
                u_sol = next(iter(sol_set))

            for idx, j in enumerate(to_solve):
                sol[D_vars[j]] = sp.simplify(u_sol[idx])

        # and finally plug in any truly free variables
        for j in to_free:
            sol[D_vars[j]] = free_values[D_vars[j]]

    return sol


def minimize(variables, *equations):
    solutions = sp.solve(equations, variables, dict=True)
    print("\nSolutions:")
    print(solutions)


def gershgorin_test(matrix):
    M = matrix
    n = M.rows
    checks = []
    for i in range(n):
        center = sp.simplify(M[i, i])
        radius = sp.simplify(sum(sp.Abs(M[i, j]) for j in range(n) if j != i))
        checks.append(sp.simplify(center - radius))
    return checks


def ldl_test(H):
    try:
        L, D = H.LDLdecomposition()
    except NonPositiveDefiniteMatrixError:
        print("Matrix is not positive-definite (LDL failed) => indefinite.")
        return "indefinite", None

    diags = [sp.simplify(D[i, i]) for i in range(D.rows)]

    # Display
    print("LDLᵀ diagonal entries:")
    for i, d in enumerate(diags):
        print(f"  D[{i},{i}] = {d}")

    # Check signs
    all_pos = all(d.is_positive for d in diags)
    all_nonneg = all(d.is_nonnegative for d in diags)

    if all_pos:
        status = "pos-definite"
    elif all_nonneg and not all_pos:
        status = "pos-semidef"
    else:
        status = "indefinite"

    print(f"\n=> Matrix is {status.replace('-', ' ')}.")
    return status, diags


    L, D = H.LDLdecomposition()

    diags = [sp.simplify(D[i, i]) for i in range(D.rows)]

    print("LDLᵀ diagonal entries:")
    for i, d in enumerate(diags):
        print(f"  D[{i},{i}] =", d)

    all_pos = all(d.is_positive for d in diags)
    all_nonneg = all(d.is_nonnegative for d in diags)

    if all_pos:
        status = "pos-definite"
    elif all_nonneg:
        status = "pos-semidef"
    else:
        status = "indefinite"

    print(f"\n=> Matrix is {status.replace('-', ' ')}.")
    return status, diags

def get_substitution_dict(tpe, func_instance, choose):
    if getattr(func_instance, "combined", False):
        if func_instance.use_1D:
            return get_combined_subs_1D(choose)
        else:
            return get_combined_subs_2D(func_instance.sigma2_value)

    if func_instance.use_1D:
        if tpe == "sum":
            return get_1D_sum_subs(func_instance.w0)
        else:
            return get_generic_subs(True, tpe, func_instance)
    else:
        if tpe == "sum":
            return get_2D_sum_subs(func_instance.w0)
        else:
            return get_generic_subs(False, tpe, func_instance)


def computation_helper(gradient, tpe, sub, func_instance, choose=None):
    if tpe == "combined":
        print("\nCombined mode: w0_even =", func_instance.w0_even, ", w0_odd =", func_instance.w0_odd, ", use_1D =", func_instance.use_1D)
        expr = sp.simplify((func_instance.compute_L() + func_instance.compute_M()))
        print("Function to compute gradient to is:", expr)
    elif tpe in ["L", "M", "sum"]:
        if tpe == "L":
            expr = func_instance.expr_L
        elif tpe == "M":
            expr = func_instance.expr_M
        else:
            expr = func_instance.expr
        pretty_print(expr)
    else:
        raise ValueError("Invalid type. Choose 'L', 'M', 'sum', or 'combined'.")

    mat = (func_instance.compute_gradient(expr, func_instance.opt_vars)
           if gradient else func_instance.compute_hessian(expr, func_instance.opt_vars))

    mat = treat_exp(mat)
    if sub:
        subs_dict = get_substitution_dict(tpe, func_instance, choose)
        mat = sp.simplify(mat.subs(subs_dict))

    return treat_exp(mat)


