#!/usr/bin/env python
import pandas as pd
import sympy as sp
from sympy.polys.polytools import groebner
from File_processing import load_csv_and_prune
from itertools import combinations_with_replacement

# === Rational reconstruction utilities ===

def gen_monomials(vars, max_deg):
    """
    Generate all monomials in 'vars' up to total degree max_deg.
    """
    monoms = []
    for deg in range(max_deg + 1):
        for combo in combinations_with_replacement(vars, deg):
            mon = 1
            for v in combo:
                mon *= v
            monoms.append(mon)
    return monoms

# Counter for coefficient naming
global_coeff_counter = 0

def _new_coeff(prefix="c"):
    """Generate a fresh Sympy Symbol for a coefficient."""
    global global_coeff_counter
    name = f"{prefix}{global_coeff_counter}"
    global_coeff_counter += 1
    return sp.Symbol(name)


def reconstruct_rational_expression(samples, input_vars, output_sym, num_degree, den_degree):
    """
    Fit a rational function R = P/Q where P, Q are polynomials in input_vars.
    P has total degree ≤ num_degree, Q has total degree ≤ den_degree (Q's constant=1).
    Returns the fitted sympy.Expr for R. Raises RuntimeError if fit fails.
    """
    # Build monomial bases
    num_monos = gen_monomials(input_vars, num_degree)
    den_monos = gen_monomials(input_vars, den_degree)
    # Remove constant term from denominator monomials
    if den_monos and den_monos[0] == 1:
        den_monos = den_monos[1:]

        # Create unknown coefficients
    a = [_new_coeff("a") for _ in num_monos]
    b = [_new_coeff("b") for _ in den_monos]

    # Build numerator and denominator and denominator
    P = sum(ci * mi for ci, mi in zip(a, num_monos))
    Q = 1 + sum(di * mj for di, mj in zip(b, den_monos))

    # Assemble linear equations: P(inputs) - y * Q(inputs) = 0
    eqs = []
    for samp in samples:
        # Map Symbol -> numeric value
        subs_map = {v: samp[v.name] for v in input_vars}
        y_val = samp[output_sym.name]
        p_val = P.subs(subs_map)
        q_val = Q.subs(subs_map)
        eqs.append(p_val - y_val * q_val)

    unknowns = a + b
    # Solve the linear system
    try:
        M, v = sp.linear_eq_to_matrix(eqs, unknowns)
    except Exception:
        raise RuntimeError(f"Cannot linearize fit for {output_sym.name}")

    sol_set = sp.linsolve((M, -v), *unknowns)
    sol_tuple = next(iter(sol_set), None)
    if sol_tuple is None:
        raise RuntimeError(f"No solution for rational fit of {output_sym.name}")

    # Construct fitted rational expression
    R = P/Q
    R_fitted = R.subs(dict(zip(unknowns, sol_tuple)))
    return sp.simplify(R_fitted)


def reconstruct_all_D(samples, input_vars, D_vars, num_degree, den_degree, known_relations=None):
    """
    Reconstruct each symbol in D_vars as a rational function of input_vars,
    optionally applying known linear relations to avoid redundant fits.

    Parameters
    ----------
    samples : list of dicts
        Numeric records mapping input_vars + D_vars to values.
    input_vars : list of sympy.Symbol
        Symbols used as inputs for rational model (E_i, S).
    D_vars : list of sympy.Symbol
        Symbols to reconstruct.
    num_degree : int
        Max total degree for numerator.
    den_degree : int
        Max total degree for denominator.
    known_relations : dict, optional
        Mapping from sympy.Symbol in D_vars to a Sympy expression in other D_vars
        (e.g. {D1: D4 - D7}); these entries are set by substitution instead of fitting.

    Returns
    -------
    dict
        Mapping each D_var to fitted sympy.Expr or None.
    """
    if known_relations is None:
        known_relations = {}

    results = {}
    # First reconstruct all D_vars that are NOT in known_relations
    for dv in D_vars:
        if dv in known_relations:
            continue
        try:
            expr = reconstruct_rational_expression(
                samples, input_vars, dv, num_degree, den_degree
            )
        except RuntimeError as e:
            print(f"Warning: {e}")
            expr = None
        results[dv] = expr

    # Now apply known relations
    for dv, rel_expr in known_relations.items():
        # Substitute previously computed results into the relation
        if all((sym in results and results[sym] is not None) for sym in rel_expr.free_symbols):
            results[dv] = sp.simplify(rel_expr.subs(results))
        else:
            # Cannot apply relation if dependencies missing
            results[dv] = None
    return results

# === Main pipeline ===

def analyse_dataset(csvfile: str,
                    cols_to_remove=None,
                    num_degree: int = 2,
                    den_degree: int = 1) -> dict:
    """
    Load CSV of numeric minima, then reconstruct each Dij as rational function
    of E_ variables (and optional S). Returns dict D_var -> expression.
    """
    if cols_to_remove is None:
        cols_to_remove = ["L", "M", "LM"]

    # Load and prune
    df = load_csv_and_prune(csvfile, cols_to_remove)
    samples = df.to_dict(orient="records")

    # Identify input and output symbols
    E_cols = [c for c in df.columns if c.startswith("E_")]
    D_cols = [c for c in df.columns if c.startswith("D_")]
    sigma_col = "S"

    input_vars = [sp.Symbol(c) for c in E_cols]
    if sigma_col in df.columns:
        input_vars.append(sp.Symbol(sigma_col))
    D_vars = [sp.Symbol(c) for c in D_cols]

    # Detect and apply exact linear relations among D_vars
    # Build all linear relations from numeric samples
    # relations: map dependent D_var -> expression in other D_vars
    def find_linear_relations(samples, D_vars, tol=1e-6):
        """
        Discover exact linear dependencies among D_vars from numeric samples.
        Returns dict mapping each dependent D_var (pivot) -> linear expression in others.
        """
        # Build numeric matrix of shape (n_samples, n_D)
        mat = sp.Matrix([[float(s[d.name]) for d in D_vars] for s in samples])
        # Nullspace gives basis vectors for c where mat * c = 0
        ns = mat.nullspace()
        rels = {}
        for vec in ns:
            # find a pivot index with a nonzero entry
            for i, coeff in enumerate(vec):
                if abs(float(coeff)) > tol:
                    pivot = i
                    break
            # build relation D_pivot + sum_{j!=pivot}(vec[j]/vec[pivot]*D_j) = 0
            expr = -sum((vec[j]/vec[pivot]) * D_vars[j] for j in range(len(D_vars)) if j != pivot)
            rels[D_vars[pivot]] = sp.simplify(expr)
        return rels

    known_rels = find_linear_relations(samples, D_vars)
    if known_rels:
        print("Detected linear relations among D:")
        for dv, expr in known_rels.items():
            print(f"  {dv} = {expr}")
    # Now reconstruct with known linear relations
    D_exprs = reconstruct_all_D(samples, input_vars, D_vars, num_degree, den_degree, known_relations=known_rels)(samples, input_vars, D_vars, num_degree, den_degree)
    return D_exprs

if __name__ == "__main__":
    D_exprs = analyse_dataset("sigma_mix.csv")
    print("Reconstructed D expressions:")
    for dv, expr in D_exprs.items():
        print(f"{dv} = {expr}")
