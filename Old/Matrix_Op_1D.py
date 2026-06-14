import sympy as sp
from Function.Symbols import E_sym, D_sym


class SymbolicConvChain1D:
    def __init__(self, E_sym, D_sym, L, w0, pool_size, factor, pad):
        """
        Initialize the 1D convolution chain.

        Parameters:
            E_symbols (list): A list of 3 symbolic entries for filter E.
            D_symbols (list): A list of 3 symbolic entries for filter D.
            L (int): Length of the input vector.
            w0 (int): Index (0-indexed) where the one is placed in the input vector.
            pool_size (int): Pooling window size (default 2).
            factor (int): Upsampling factor for scatter pooling (default 2).
            pad (int or None): Padding width. If None, it will be set automatically.
        """
        self.E_symbols = E_sym
        self.D_symbols = D_sym
        # Represent the 1D filters as 1x3 matrices (row vectors).
        self.E = sp.Matrix([self.E_symbols])
        self.D = sp.Matrix([self.D_symbols])
        self.L = L
        self.w0 = w0
        self.pool_size = pool_size
        self.factor = factor
        self.pad = pad  # May be None; if so, will be computed automatically.

    def conv_valid_sym(self, vector, filt):
        """
        Perform a valid (no-padding) convolution of a 1D symbolic row vector with a filter.
        """
        n = vector.cols  # vector is 1 x n.
        k = filt.cols  # filt is 1 x k.
        out_length = n - k + 1
        out = sp.Matrix(1, out_length, lambda i, j: 0)
        for j in range(out_length):
            s = 0
            for b in range(k):
                s += filt[0, b] * vector[0, j + b]
            out[0, j] = sp.simplify(s)
        return out

    def pool_vector(self, vector):
        """
        Perform average pooling on a 1D row vector with a window of size pool_size.
        """
        n = vector.cols
        out_length = n // self.pool_size
        A = sp.zeros(1, out_length)
        for j in range(out_length):
            s = sp.S(0)
            for b in range(self.pool_size):
                s += vector[0, self.pool_size * j + b]
            A[0, j] = sp.simplify(s / self.pool_size)
        return A

    def scatter_pool(self, A):
        """
        Upsample the 1D vector A by scattering its elements into a larger vector.
        Each element of A is placed at an index that is a multiple of factor,
        with zeros inserted in between.
        """
        m = A.cols
        out_length = m * self.factor
        U = sp.zeros(1, out_length)
        for j in range(m):
            U[0, self.factor * j] = A[0, j]
        return U

    def pad_vector(self, vector, pad):
        """
        Pad a 1D vector with zeros on both sides.
        If pad is None, use self.pad.
        """
        if pad is None:
            pad = self.pad
        n = vector.cols
        new_n = n + 2 * pad
        padded = sp.zeros(1, new_n)
        for j in range(n):
            padded[0, j + pad] = vector[0, j]
        return sp.simplify(padded)

    def vector_ones(self):
        """
        Create a 1 x L row vector with a one at index w0 and zeros elsewhere.
        """
        M = sp.zeros(1, self.L)
        M[0, self.w0] = 1
        return M

    def compute_Y(self, subs=False, same_size=False):
        xi = self.vector_ones()  # Step 1.
        C = self.conv_valid_sym(xi, self.E)  # Step 2.
        A = self.pool_vector(C)  # Step 3.
        if subs:
            # Choose grouping for filter E based on w0.
            if self.w0 % 2 == 0:
                # For odd w0, let: a = E₁ and b = E₂ + E₃.
                a_expr = self.E[0] / 2
                b_expr = (self.E[1] + self.E[2]) / 2
            else:
                a_expr = (self.E[2]) / 2
                b_expr = (self.E[0] + self.E[1]) / 2

            a_sym, b_sym = sp.symbols('a b')
            A = sp.simplify(A.subs({a_expr: a_sym, b_expr: b_sym}))
        U = self.scatter_pool(A)

        if same_size:
            total_pad = (self.L + 2) - U.cols
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            pad = pad_left
            U_padded = self.pad_vector(U, pad)
        else:
            U_padded = self.pad_vector(U, self.pad)

        Y = self.conv_valid_sym(U_padded, self.D)  # Step 6

        if subs:
            c_sym, d_sym, e_sym = sp.symbols('c d e')
            subs_dict = {
                self.D[0]: c_sym,
                self.D[1]: d_sym,
                self.D[2]: e_sym
            }
            Y = Y.applyfunc(lambda expr: sp.collect(expr, a_sym))
            Y = Y.applyfunc(lambda expr: sp.collect(expr, b_sym))
            Y = sp.simplify(Y.subs(subs_dict))

        return Y

    def sum_of_squares(self, Y, target=None):
        """
        Compute the sum of squares of differences between the output Y and a target vector.
        Both Y and target are assumed to be 1 x n row vectors.
        If target is None, compute the sum of squares of Y.
        """
        total = sp.S(0)
        if target is not None:
            for j in range(Y.cols):
                total += sp.simplify((Y[0, j] - target[0, j]) ** 2)
        else:
            for j in range(Y.cols):
                total += sp.simplify(Y[0, j] ** 2)
        return sp.simplify(total)

if __name__ == "__main__":
    L = 100
    w0 = 33
    conv_chain = SymbolicConvChain1D(E_sym, D_sym, L, w0, pool_size=2, factor=2, pad=None)

    Y = conv_chain.compute_Y(subs=0, same_size=True)
    target = conv_chain.vector_ones()

    sum = conv_chain.sum_of_squares(Y, target)
    sp.pretty_print(Y)
    sp.pretty_print(sum)