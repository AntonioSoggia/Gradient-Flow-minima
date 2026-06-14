import sympy as sp

E_symbols = sp.symbols('E11 E12 E13 E21 E22 E23 E31 E32 E33', real=True)
D_symbols = sp.symbols('D11 D12 D13 D21 D22 D23 D31 D32 D33', real=True)
sigma2 = sp.symbols('sigma2', positive=True)
E_sym = sp.symbols('E1 E2 E3', real=True)
D_sym = sp.symbols('D1 D2 D3', real=True)
x_vars1 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18')
E11, E12, E13, E21, E22, E23, E31, E32, E33 = E_symbols
D11, D12, D13, D21, D22, D23, D31, D32, D33 = D_symbols

E1, E2, E3 = E_sym
D1, D2, D3 = D_sym

x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18 = x_vars1

alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa, lambd, mu, nu, xi, omicron, pi_, rho, tau, Sigma  = sp.symbols(
    'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho tau Sigma', real = "True"
)

a, b, c, d, e, f, g, h = sp.symbols('a b c d e f g h')

S1, S2, S3, S4, T1, T2, T3 = sp.symbols('S1 S2 S3 S4 T1 T2 T3')
