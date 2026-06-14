from Min_computation_helper import *
from Function.Symbols import *
from Function.Sympy_functions import FunctionSimpy

def main(var, hess):
    use_1D = False
    w0 = 5
    w0_even = 5
    w0_odd = 6
    tpe = "combined"
    choose = 1

    func_instance = FunctionSimpy(
        use_Numeric=False,
        use_1D=use_1D,
        subs=0,
        sigma2_value=sigma2,
        combined=True,
        w0_even=w0_even,
        w0_odd=w0_odd,
        w0=w0
    )


    grad_expr = computation_helper(gradient=True, tpe=tpe, sub=True, func_instance=func_instance, choose=choose)
    print("Gradient Expression (combined):")
    print(sp.latex(grad_expr))
    pretty_print(grad_expr)
    if var is not None:
        sol = solve_for_D(
            grad_expr,
            [D21, D22, D23],
        )
        for D in [D21, D22, D23]:
            print(D, "=", sp.latex(sol[D]))

    if hess:
        hessian_expr = computation_helper(gradient=False, tpe=tpe, sub=True, func_instance=func_instance)
        print("\nHessian Expression (combined):")
        print(sp.latex(hessian_expr))
        gershgorin = gershgorin_test(hessian_expr)
        status, diags = ldl_test(hessian_expr)
        print(gershgorin)
        print("\nLDL: ")
        print(status, diags)

if __name__ == "__main__":
    var_1D = [E1, E2, E3, D1, D2, D3]
    var_2D = [D21, D22, D23]
    var = var_2D
    hess = 1
    main(var, hess)
