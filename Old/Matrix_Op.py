import sympy as sp
from Function.Symbols import E_symbols, D_symbols

class SymbolicConvChain:
    def __init__(self, E_symbols, D_symbols, H, W, w0, pool_size, factor, pad):
        """
        Initialize the convolution chain with common parameters.

        Parameters:
            E (sp.Matrix): Filter matrix E.
            D (sp.Matrix): Filter matrix D.
            H (int): Height of the input matrix.
            W (int): Width of the input matrix.
            w0 (int): Column index where ones will be placed.
            pool_size (int): Pooling window size (default 2).
            factor (int): Upsampling factor for scatter pooling (default 2).
            pad (int or None): Padding width. If None, then pad will be set automatically
                               in compute_Y to ensure the output has the same size as the input.
        """
        self.E_symbols = E_symbols
        self.D_symbols = D_symbols
        self.E = sp.Matrix(3, 3, self.E_symbols)
        self.D = sp.Matrix(3, 3, self.D_symbols)
        self.H = H
        self.W = W
        self.w0 = w0
        self.pool_size = pool_size
        self.factor = factor
        self.pad = pad  # May be None; if so, we'll compute it automatically.


    def conv_valid_sym(self, matrix, filt):
        """
        Perform a valid (no-padding) convolution of a symbolic matrix with a filter.
        """
        m, n = matrix.shape
        fm, fn = filt.shape
        out_rows = m - fm + 1
        out_cols = n - fn + 1
        out = sp.Matrix(out_rows, out_cols, lambda i, j: 0)
        for i in range(out_rows):
            for j in range(out_cols):
                s = 0
                for a in range(fm):
                    for b in range(fn):
                        s += filt[a, b] * matrix[i + a, j + b]
                out[i, j] = sp.simplify(s)
        return out

    def pool_matrix(self, O):
        """
        Perform average pooling on matrix O with a square window of size pool_size.
        """
        m, n = O.shape
        out_rows = m // self.pool_size
        out_cols = n // self.pool_size
        A = sp.zeros(out_rows, out_cols)
        for i in range(out_rows):
            for j in range(out_cols):
                s = sp.S(0)
                for a in range(self.pool_size):
                    for b in range(self.pool_size):
                        s += O[self.pool_size * i + a, self.pool_size * j + b]
                A[i, j] = sp.simplify(s / (self.pool_size ** 2))
        return A

    def scatter_pool(self, A):
        """
        Upsample matrix A by scattering its elements into a larger matrix.
        """
        U = sp.zeros(A.rows * self.factor, A.cols * self.factor)
        for i in range(A.rows):
            for j in range(A.cols):
                U[self.factor * i, self.factor * j] = A[i, j]
        return U

    def pad_matrix(self, matrix, pad):
        """
        Pad a matrix with zeros on all sides with a padding width 'pad'.
        If pad is None, use self.pad.
        """
        if pad is None:
            pad = self.pad
        m, n = matrix.shape
        new_m = m + 2 * pad
        new_n = n + 2 * pad
        padded = sp.zeros(new_m, new_n)
        for i in range(m):
            for j in range(n):
                padded[i + pad, j + pad] = matrix[i, j]
        return sp.simplify(padded)

    def matrix_ones(self):
        """
        Create an H x W matrix with ones in the specified column (w0) and zeros elsewhere.
        """
        M = sp.zeros(self.H, self.W)
        for i in range(self.H):
            M[i, self.w0] = 1
        return M

    def compute_Y(self, subs, same_size):
        """
        Compute the output Y via the chain:
          1. xi = matrix_ones(H, W, w0)
          2. C = conv_valid_sym(xi, E)
          3. A_mat = pool_matrix(C, pool_size)
          4. U = scatter_pool(A_mat, factor)
          5. U_padded = pad_matrix(U, pad)
          6. Y = conv_valid_sym(U_padded, D)

        Optionally perform substitutions based on the value of w0.

        If same_size is True, the padding is automatically chosen so that the final output Y
        has the same dimensions as the input matrix.
        Returns Y (after applying substitutions if subs is truthy).
        """
        xi = self.matrix_ones()
        C = self.conv_valid_sym(xi, self.E)
        A_mat = self.pool_matrix(C)

        if subs:
            # Use self.w0 rather than an undefined variable w0.
            if self.w0 % 2 == 1:
                # For odd w0, use one grouping of filter E entries.
                a_expr = (self.E[0, 2] + self.E[1, 2] + self.E[2, 2]) / 2
                b_expr = (self.E[0, 0] + self.E[1, 0] + self.E[2, 0] +
                          self.E[0, 1] + self.E[1, 1] + self.E[2, 1]) / 2
            else:
                # For even w0, use a different grouping.
                a_expr = (self.E[0, 1] + self.E[1, 1] + self.E[2, 1] +
                          self.E[0, 2] + self.E[1, 2] + self.E[2, 2]) / 2
                b_expr = (self.E[0, 0] + self.E[1, 0] + self.E[2, 0]) / 2
            a_sym, b_sym = sp.symbols('a b')
            A_mat = sp.simplify(A_mat.subs({a_expr: a_sym, b_expr: b_sym}))

            # For filter D, perform similar substitutions.
            c_expr = self.D[0, 0] + self.D[2, 0]
            d_expr = self.D[0, 2] + self.D[2, 2]
            e_expr = self.D[0, 1] + self.D[2, 1]
            f_expr = self.D[1, 0]
            g_expr = self.D[1, 2]
            h_expr = self.D[1, 1]
            c_sym, d_sym, e_sym, f_sym, g_sym, h_sym = sp.symbols('c d e f g h')

        U = self.scatter_pool(A_mat)

        if same_size:
            required_pad = (self.H - U.rows + 2) // 2
            U_padded = self.pad_matrix(U, pad=required_pad)
        else:
            U_padded = self.pad_matrix(U, pad=self.pad)

        Y = self.conv_valid_sym(U_padded, self.D)

        if subs:
            Y_collected = Y.applyfunc(lambda expr: sp.collect(expr, a_sym))
            Y_collected = Y_collected.applyfunc(lambda expr: sp.collect(expr, b_sym))
            Y_subs = Y_collected.subs({
                c_expr: c_sym,
                d_expr: d_sym,
                e_expr: e_sym,
                f_expr: f_sym,
                g_expr: g_sym,
                h_expr: h_sym
            })
            Y = sp.simplify(Y_subs)

        return Y

    def sum_of_squares_of_rows(self, Y, row_indices, target=None):

        total = sp.S(0)
        print(target.row(2))
        for i in row_indices:
            row = Y.row(i)
            if target is not None:
                targ_row = target.row(i)
                total += sum(sp.simplify((row[j] - targ_row[j]) ** 2) for j in range(Y.cols))
            else:
                total += sum(sp.simplify(row[j] ** 2) for j in range(Y.cols))
        return sp.simplify(total)


# Example usage:
if __name__ == "__main__":
    # Define symbolic filters for E and D (assumed to be 3x3).


    # Set dimensions of the input matrix.
    H, W = 13, 13
    w0 = 5
    conv_chain = SymbolicConvChain(E_symbols, D_symbols, H, W, w0, pool_size=2, factor=2, pad=None)

    Y = conv_chain.compute_Y(subs=0, same_size=True)
    target = conv_chain.matrix_ones()

    sum = conv_chain.sum_of_squares_of_rows(Y, [2, 3], target)
    sp.pretty_print(Y)
    sp.pretty_print(sum)