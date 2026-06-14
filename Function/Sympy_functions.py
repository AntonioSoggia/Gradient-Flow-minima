import sympy as sp
import numpy as np
from Function.Symbols import E_symbols, D_symbols, E_sym, D_sym
from Function.Matrix_Op_tot import Mini_Unet


class FunctionSimpyBase:
    @staticmethod
    def compute_gradient(expr, variables):
        return sp.Matrix([sp.diff(expr, var) for var in variables])

    @staticmethod
    def compute_hessian(expr, variables):
        return sp.hessian(expr, variables)

    @staticmethod
    def sub(expr, sub_dict):
        return expr.subs(sub_dict)

    @staticmethod
    def autocorr_E(a, b, E):
        rows, cols = E.shape
        s = sp.Integer(0)
        for u in range(rows):
            for v in range(cols):
                u2 = u + a
                v2 = v + b
                if 0 <= u2 < rows and 0 <= v2 < cols:
                    s += E[u, v] * E[u2, v2]
        return sp.simplify(s)

    def lambdify_expr(self, expr, variables):
        return sp.lambdify(variables, expr, 'numpy')


class FunctionSimpy(FunctionSimpyBase):
    def __init__(self, use_Numeric, use_1D, subs, w0, sigma2_value, combined=False, w0_even=None, w0_odd=None):
        """
        If combined is True, two versions are computed:
          one with an even w₀ and one with an odd w₀.
        The loss is taken as the average of (L+M) from both versions.
        Additional keys in evaluate (grad_combined, hessian_combined, combined)
        refer to the averaged expressions.
        """
        self.combined = combined
        self.use_Numeric = use_Numeric
        self.use_1D = use_1D
        self.sigma2_value = sigma2_value
        self.subs = subs

        if not self.combined:
            self.w0 = w0
        else:
            self.w0_even = w0_even if w0_even is not None else (w0 if w0 % 2 == 1 else w0 - 1)
            self.w0_odd  = w0_odd  if w0_odd is not None  else (w0 if w0 % 2 == 0 else w0 + 1)

        if self.use_1D:
            self.E = E_sym
            self.D = D_sym
            self.opt_vars = list(self.E) + list(self.D)
            self.size = 11
        else:
            self.E_symbols = E_symbols
            self.D_symbols = D_symbols
            self.E = sp.Matrix(3, 3, self.E_symbols)
            self.D = sp.Matrix(3, 3, self.D_symbols)
            self.opt_vars = list(self.E_symbols) + list(self.D_symbols)
            self.size = (11, 11)

        # Compute expressions.
        if not self.combined:
            self.expr_L, self.expr_M, self.expr = self._compute_exprs(self.w0)
        else:
            L_even, M_even, total_even = self._compute_exprs(self.w0_even)
            L_odd, M_odd, total_odd = self._compute_exprs(self.w0_odd)
            self.expr_L = sp.simplify((L_even + L_odd) / 2)
            self.expr_M = sp.simplify((M_even + M_odd) / 2)
            self.expr   = sp.simplify((total_even + total_odd) / 2)

        # Compute gradients and Hessians.
        self.sym_grad = self.compute_gradient(self.expr, self.opt_vars)
        self.sym_hessian = self.compute_hessian(self.expr, self.opt_vars)
        # hessian_sum could be defined as the Hessian of (L+M) which is self.expr.
        self.hessian_sum = self.compute_hessian(self.expr, self.opt_vars)

        # Lambdify for numeric evaluations.
        if self.use_Numeric:
            self.lambdify_all()
        else:
            self._f_numeric = None
            self._grad_numeric = None
            self._hessian_numeric = None

    def _compute_exprs(self, w0):
        """
        Computes the L and M expressions (and their sum) for a given w₀.
        """
        conv_chain = Mini_Unet(self.E, self.D, self.size, w0, 2, 2, None, use_1D=self.use_1D)
        Y = conv_chain.compute_Y(subs=self.subs, same_size=True)
        target = conv_chain.ones()
        L = conv_chain.sum_of_squares(Y, row_indices=[1, 2], target=target)
        L = sp.simplify(L)
        sigma2_val = self.sigma2_value
        if self.use_1D:
            E1, E2, E3 = self.E
            D1, D2, D3 = self.D
            M = sigma2_val / 4 * (
                (D1**2 + D2**2 + D3**2) *
                (E1**2 + E2**2 + E3**2 + E1*E2 + E2*E3) +
                D1 * D3 * (2*E1*E3 + E1*E2 + E2*E3)
            )
        else:
            R00 = self.autocorr_E(0, 0, self.E)
            R01 = self.autocorr_E(0, 1, self.E)
            R10 = self.autocorr_E(1, 0, self.E)
            R11 = self.autocorr_E(1, 1, self.E)
            R1m1 = self.autocorr_E(1, -1, self.E)
            Q = sp.simplify(sigma2_val / 16 * (4*R00 + 4*(R01+R10) + 2*(R11+R1m1)))
            cov_h = sp.simplify(sigma2_val / 16 * sum(
                self.autocorr_E(i-p, j-q-2, self.E)
                for i in range(2) for j in range(2)
                for p in range(2) for q in range(2)
            ))
            cov_v = sp.simplify(sigma2_val / 16 * sum(
                self.autocorr_E(i-p-2, j-q, self.E)
                for i in range(2) for j in range(2)
                for p in range(2) for q in range(2)
            ))
            cov_d1 = sp.simplify(sigma2_val / 16 * sum(
                self.autocorr_E(i-p-2, j-q-2, self.E)
                for i in range(2) for j in range(2)
                for p in range(2) for q in range(2)
            ))
            cov_d2 = sp.simplify(sigma2_val / 16 * sum(
                self.autocorr_E(i-p-2, j-q+2, self.E)
                for i in range(2) for j in range(2)
                for p in range(2) for q in range(2)
            ))
            M = sp.simplify(
                (self.D[0,0]**2 + self.D[0,2]**2 + self.D[2,0]**2 + self.D[2,2]**2)*Q +
                2*(self.D[0,0]*self.D[0,2] + self.D[2,0]*self.D[2,2])*cov_h +
                2*(self.D[0,0]*self.D[2,0] + self.D[0,2]*self.D[2,2])*cov_v +
                2*self.D[0,0]*self.D[2,2]*cov_d1 +
                2*self.D[0,2]*self.D[2,0]*cov_d2
            )
        M = sp.simplify(M)
        return L, M, sp.simplify(L + M)

    def lambdify_all(self):
        self._f_numeric = self.lambdify_expr(self.expr, self.opt_vars)
        self._grad_numeric = self.lambdify_expr(self.compute_gradient(self.expr, self.opt_vars), self.opt_vars)
        self._hessian_numeric = self.lambdify_expr(self.compute_hessian(self.expr, self.opt_vars), self.opt_vars)
        self._f_L_numeric = self.lambdify_expr(self.expr_L, self.opt_vars)
        self._f_M_numeric = self.lambdify_expr(self.expr_M, self.opt_vars)
        self._grad_L_numeric = self.lambdify_expr(self.compute_gradient(self.expr_L, self.opt_vars), self.opt_vars)
        self._grad_M_numeric = self.lambdify_expr(self.compute_gradient(self.expr_M, self.opt_vars), self.opt_vars)
        self._grad_sum_numeric = self.lambdify_expr(self.compute_gradient(self.expr, self.opt_vars), self.opt_vars)

    def compute_L(self):
        return self.expr_L

    def compute_M(self):
        return self.expr_M

    def compute_hessian_sum(self):
        return self.compute_hessian(self.expr, self.opt_vars)

    def evaluate(self, vals=None, func_type='all'):
        if self.use_Numeric:
            if vals is None:
                raise ValueError("Numeric mode requires a values tuple.")
            if func_type == 'L':
                return self._f_L_numeric(*vals)
            elif func_type == 'M':
                return self._f_M_numeric(*vals)
            elif func_type == 'grad_L':
                return np.array(self._grad_L_numeric(*vals)).flatten()
            elif func_type == 'grad_M':
                return np.array(self._grad_M_numeric(*vals)).flatten()
            elif func_type in ['grad_sum', 'grad_combined']:
                return np.array(self._grad_numeric(*vals)).flatten()
            elif func_type in ['hessian', 'hessian_combined']:
                return self._hessian_numeric(*vals)
            elif func_type in ['combined', 'all']:
                return self._f_numeric(*vals)
            else:
                raise ValueError("Unknown function type")
        else:
            if vals is None:
                if func_type == 'L':
                    return self.compute_L()
                elif func_type == 'M':
                    return self.compute_M()
                elif func_type == 'grad_L':
                    return self.compute_gradient(self.expr_L, self.opt_vars)
                elif func_type == 'grad_M':
                    return self.compute_gradient(self.expr_M, self.opt_vars)
                elif func_type in ['grad_sum', 'grad_combined']:
                    return self.compute_gradient(self.expr, self.opt_vars)
                elif func_type in ['hessian', 'hessian_combined']:
                    return self.compute_hessian(self.expr, self.opt_vars)
                elif func_type in ['combined', 'all']:
                    return self.expr
                else:
                    raise ValueError("Unknown function type")
            else:
                subs_dict = {var: val for var, val in zip(self.opt_vars, vals)}
                if func_type == 'L':
                    return self.compute_L().subs(subs_dict)
                elif func_type == 'M':
                    return self.compute_M().subs(subs_dict)
                elif func_type == 'grad_L':
                    return self.compute_gradient(self.expr_L, self.opt_vars).subs(subs_dict)
                elif func_type == 'grad_M':
                    return self.compute_gradient(self.expr_M, self.opt_vars).subs(subs_dict)
                elif func_type in ['grad_sum', 'grad_combined']:
                    return self.compute_gradient(self.expr, self.opt_vars).subs(subs_dict)
                elif func_type in ['hessian', 'hessian_combined']:
                    return self.compute_hessian(self.expr, self.opt_vars).subs(subs_dict)
                elif func_type in ['combined', 'all']:
                    return (self.compute_L() + self.compute_M()).subs(subs_dict)
                else:
                    raise ValueError("Unknown function type")

