import sympy as sp
from Function.Symbols import E_sym, D_sym, E_symbols, D_symbols, a, b, c, d, e, f, g, h

class Mini_Unet:
    def __init__(self, E, D, size, w0, pool_size, factor, pad, use_1D=False):
        self.use_1D = use_1D
        self.E_symbols = E
        self.D_symbols = D
        if use_1D:
            self.L = size
            self.E = sp.Matrix([self.E_symbols])
            self.D = sp.Matrix([self.D_symbols])
        else:
            self.H, self.W = size
            self.E = sp.Matrix(3, 3, self.E_symbols)
            self.D = sp.Matrix(3, 3, self.D_symbols)
        self.w0 = w0
        self.pool_size = pool_size
        self.factor = factor
        self.pad = pad

    def conv_valid_sym(self, matrix, filt):
        if self.use_1D:
            n = matrix.cols
            k = filt.cols
            out_length = n - k + 1
            out = sp.Matrix(1, out_length, lambda i, j: 0)
            for j in range(out_length):
                s = 0
                for b in range(k):
                    s += filt[0, b] * matrix[0, j + b]
                out[0, j] = sp.simplify(s)
            return out
        else:
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

    def pool(self, X):
        if self.use_1D:
            n = X.cols
            out_length = n // self.pool_size
            A = sp.zeros(1, out_length)
            for j in range(out_length):
                s = sp.S(0)
                for b in range(self.pool_size):
                    s += X[0, self.pool_size * j + b]
                A[0, j] = sp.simplify(s / self.pool_size)
            return A
        else:
            m, n = X.shape
            out_rows = m // self.pool_size
            out_cols = n // self.pool_size
            A = sp.zeros(out_rows, out_cols)
            for i in range(out_rows):
                for j in range(out_cols):
                    s = sp.S(0)
                    for a in range(self.pool_size):
                        for b in range(self.pool_size):
                            s += X[self.pool_size * i + a, self.pool_size * j + b]
                    A[i, j] = sp.simplify(s / (self.pool_size ** 2))
            return A

    def scatter_pool(self, A):
        if self.use_1D:
            m = A.cols
            out_length = m * self.factor
            U = sp.zeros(1, out_length)
            for j in range(m):
                U[0, self.factor * j] = A[0, j]
            return U
        else:
            U = sp.zeros(A.rows * self.factor, A.cols * self.factor)
            for i in range(A.rows):
                for j in range(A.cols):
                    U[self.factor * i, self.factor * j] = A[i, j]
            return U

    def pad_matrix(self, matrix, pad):
        if pad is None:
            pad = self.pad
        if isinstance(pad, tuple):
            pad_top, pad_bottom, pad_left, pad_right = pad
        else:
            pad_top = pad_bottom = pad_left = pad_right = pad
        m, n = matrix.shape
        new_m = m + pad_top + pad_bottom
        new_n = n + pad_left + pad_right
        padded = sp.zeros(new_m, new_n)
        for i in range(m):
            for j in range(n):
                padded[i + pad_top, j + pad_left] = matrix[i, j]
        return sp.simplify(padded)
    def ones(self):
        if self.use_1D:
            M = sp.zeros(1, self.L)
            M[0, self.w0] = 1
            return M
        else:
            M = sp.zeros(self.H, self.W)
            for i in range(self.H):
                M[i, self.w0] = 1
            return M

    def pad_vector(self, vector, pad):
        if pad is None:
            pad = self.pad
        if isinstance(pad, tuple):
            pad_left, pad_right = pad
        else:
            pad_left = pad_right = pad
        n = vector.cols
        new_n = n + pad_left + pad_right
        padded = sp.zeros(1, new_n)
        for j in range(n):
            padded[0, j + pad_left] = vector[0, j]
        return sp.simplify(padded)

    def compute_Y(self, subs, same_size):
        xi = self.ones()
        C = self.conv_valid_sym(xi, self.E)
        A = self.pool(C)
        if self.use_1D:
            if subs:
                if self.w0 % 2 == 0:
                    a_expr = self.E[0] / 2
                    b_expr = (self.E[1] + self.E[2]) / 2
                else:
                    a_expr = self.E[2] / 2
                    b_expr = (self.E[0] + self.E[1]) / 2
                a_sym, b_sym = sp.symbols('a b')
                A = sp.simplify(A.subs({a_expr: a_sym, b_expr: b_sym}))
            U = self.scatter_pool(A)
            if same_size:
                total_pad = (self.L + 2) - U.cols
                pad_left = total_pad // 2
                pad_right = total_pad - pad_left
                U_padded = self.pad_vector(U, (pad_left, pad_right))
            else:
                U_padded = self.pad_vector(U, self.pad)
            Y = self.conv_valid_sym(U_padded, self.D)
            if subs:
                c_sym, d_sym, e_sym = sp.symbols('c d e')
                subs_dict = {self.D[0]: c_sym, self.D[1]: d_sym, self.D[2]: e_sym}
                Y = Y.applyfunc(lambda expr: sp.collect(expr, a_sym))
                Y = Y.applyfunc(lambda expr: sp.collect(expr, b_sym))
                Y = sp.simplify(Y.subs(subs_dict))
            return Y
        else:
            if subs:
                if self.w0 % 2 == 1:
                    a_expr = (self.E[0, 2] + self.E[1, 2] + self.E[2, 2]) / 2
                    b_expr = (self.E[0, 0] + self.E[1, 0] + self.E[2, 0] + self.E[0, 1] + self.E[1, 1] + self.E[2, 1]) / 2
                else:
                    a_expr = (self.E[0, 1] + self.E[1, 1] + self.E[2, 1] + self.E[0, 2] + self.E[1, 2] + self.E[2, 2]) / 2
                    b_expr = (self.E[0, 0] + self.E[1, 0] + self.E[2, 0]) / 2
                a_sym, b_sym = a, b
                A = sp.simplify(A.subs({a_expr: a_sym, b_expr: b_sym}))
                c_expr = self.D[0, 0] + self.D[2, 0]
                d_expr = self.D[0, 2] + self.D[2, 2]
                e_expr = self.D[0, 1] + self.D[2, 1]
                f_expr = self.D[1, 0]
                g_expr = self.D[1, 2]
                h_expr = self.D[1, 1]
                c_sym, d_sym, e_sym, f_sym, g_sym, h_sym = c, d, e, f, g, h
            U = self.scatter_pool(A)
            if same_size:
                total_pad_rows = (self.H + 2) - U.rows
                top_pad = total_pad_rows // 2
                bottom_pad = total_pad_rows - top_pad
                total_pad_cols = (self.W + 2) - U.cols
                left_pad = total_pad_cols // 2
                right_pad = total_pad_cols - left_pad
                U_padded = self.pad_matrix(U, (top_pad, bottom_pad, left_pad, right_pad))
            else:
                U_padded = self.pad_matrix(U, self.pad)
            Y = self.conv_valid_sym(U_padded, self.D)
            if subs:
                Y_collected = Y.applyfunc(lambda expr: sp.collect(expr, a_sym))
                Y_collected = Y_collected.applyfunc(lambda expr: sp.collect(expr, b_sym))
                Y_subs = Y_collected.subs({c_expr: c_sym, d_expr: d_sym, e_expr: e_sym, f_expr: f_sym, g_expr: g_sym, h_expr: h_sym})
                Y = sp.simplify(Y_subs)
            return Y

    def sum_of_squares(self, Y, row_indices=None, target=None):
        total = sp.S(0)
        if self.use_1D:
            if target is not None:
                for j in range(Y.cols):
                    total += sp.simplify((Y[0, j] - target[0, j])**2)
            else:
                for j in range(Y.cols):
                    total += sp.simplify(Y[0, j]**2)
        else:
            if row_indices is None:
                row_indices = range(Y.rows)
            for i in row_indices:
                row = Y.row(i)
                if target is not None:
                    targ_row = target.row(i)
                    total += sum(sp.simplify((row[j] - targ_row[j])**2) for j in range(Y.cols))
                else:
                    total += sum(sp.simplify(row[j]**2) for j in range(Y.cols))
        return sp.simplify(total)

if __name__ == "__main__":
    use_1D = False
    if use_1D:
        size=15
        E, D = E_sym, D_sym
    else:
        size = (15, 15)
        E, D = E_symbols, D_symbols

    conv_chain = Mini_Unet(E, D, size, 10, 2, 2, 1, use_1D=use_1D)
    conv_chain_1 = Mini_Unet(E, D, size, 11, 2, 2, 1, use_1D=use_1D)
    Y = conv_chain.compute_Y(subs=0, same_size=True)
    Y_1 = conv_chain_1.compute_Y(subs=0, same_size=True)

    target = conv_chain.ones()
    target_1 = conv_chain_1.ones()

    s = conv_chain.sum_of_squares(Y, row_indices=[1,2], target=target)
    s_1 = conv_chain_1.sum_of_squares(Y_1, row_indices=[1,2], target=target_1)
    print(Y.shape)
    sp.pretty_print(Y)
    print(sp.latex((1/2) * (s+s_1)))
