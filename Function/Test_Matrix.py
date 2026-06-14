from Sympy_functions import FunctionSimpy
from Symbols import sigma2
import sympy as sp
w0 = 6
w1 = 5
w2 = 4
subs=1
L1 = FunctionSimpy(use_Numeric=False, use_1D=True, subs=subs, sigma2_value=sigma2, w0=w0).compute_L()
M1 = FunctionSimpy(use_Numeric=False, use_1D=True, subs=subs, sigma2_value=sigma2, w0=w0).compute_M()
S1 = L1 + M1
L2 = FunctionSimpy(use_Numeric=False, use_1D=True, subs=subs, sigma2_value=sigma2, w0=w1).evaluate(None, "grad_L")
M2 = FunctionSimpy(use_Numeric=False, use_1D=False, subs=subs, sigma2_value=sigma2, w0=w1).compute_M()
S2 = L2
fs = FunctionSimpy(
    use_Numeric=False,
    use_1D=True,
    subs=1,
    w0=3,
    sigma2_value=1.0
)

# this will print the symbolic gradient of L+M w.r.t. [E1,E2,E3,D1,D2,D3]
print(fs.evaluate(func_type='grad_L'))# sp.pretty_print(sp.simplify(1/2 * (S1 + S2)))


#L1 = FunctionSimpy(use_Numeric=False, use_1D=True, subs=subs, sigma2_value=sigma2, w0=w2).compute_L()
#print(L1)
#
#L1 = FunctionSimpy(use_Numeric=False, use_1D=False, subs=subs,  sigma2_value=sigma2, w0=w0).compute_L()
#
#print(L1)
#
#L1 = FunctionSimpy(use_Numeric=False, use_1D=False, subs=subs,  sigma2_value=sigma2, w0=w1).compute_L()
#
#print(L1)
#L1 = FunctionSimpy(use_Numeric=False, use_1D=False, subs=subs,  sigma2_value=sigma2, w0=w2).compute_L()
#
#print(L1)
#