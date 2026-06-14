from Function.Symbols import *

# First subs
subs_noise_D = {
    E11: x1 + alpha, E12: x2 + beta, E13: x3 + gamma,
    E21: x4 + delta, E22: x5 + epsilon, E23: x6 + zeta,
    E31: x7 + eta, E32: x8 + theta, E33: x9 + iota,
    D11: x10 + kappa, D12: x11 + lambd, D13: x12 + mu,
    D21: x13 + nu, D22: x14 + xi, D23: x15 + omicron,
    D31: x16 + pi_, D32: x17 + rho, D33: x18 + tau,
}

subs_no_noise_D = {
    a: x1 + alpha, b: x2 + beta, c: x3 + gamma,
    d: x4 + delta, e: x5 + epsilon, f: x6 + zeta,
    g: x7 + eta, h: x8 + theta,
}

subs_noise_noD = {
    E1: x1 + alpha, E2: x2 + beta, E3: x3 + gamma,
    D1: x4 + delta, D2: x5 + epsilon, D3: x6 + zeta,
}

subs_no_noise_noD = {a: x1 + alpha, b: x2 + beta, c: x3 + gamma, d: x4 + delta, e: x5 + epsilon}



# Other subs

subs_opt_no_noise_noD_odd = {beta: 0, gamma: 0, delta: 0, epsilon: 1 / alpha}

subs_opt_no_noise_noD_even = {alpha: 0, gamma: 0, delta: 1 / beta, epsilon: 0}



subs_opt_no_noise_D_odd = {alpha: 0, gamma: 0, delta: 1 / beta,
                             epsilon: 0, zeta: 0,
                             eta: 1 / beta, theta: 0}

subs_opt_no_noise_D_even= {alpha: 0, gamma: 0, delta: 0,
                             epsilon: 1 / beta, zeta: 0,
                             eta: 0, theta: 1 / beta}

subs_opt_noise_noD_mean = {
    alpha: 1 / (epsilon * (sigma2 + 1)),
    beta: 0,
    gamma: 0,
    delta: epsilon,
    zeta: epsilon,
    sigma2: 1,
}


subs_opt_noise_noD_mean = {
    alpha: 0,
    beta: 1 / (epsilon * (sigma2 + 1)),
    gamma: 0,
    delta: epsilon,
    zeta: 0,
    sigma2: 1,
}

subs_opt_no_noise_D_combined = {
    beta:    -(epsilon + theta),
    gamma:   -zeta - iota,
    kappa:   -pi_,

    (alpha + delta + eta)**(-1): omicron,

    xi:     omicron,
    lambd:  omicron - rho,
    mu:     omicron - tau,
    nu:     0,
}

subs_opt_noise_noD_even = {
    delta: 0,
    zeta: 0,
    sigma2: 1,
    alpha: 1 / (epsilon * (sigma2 + 1)),
    beta: 2 / (epsilon * (sigma2 + 2)),
    gamma: -sigma2 / (epsilon * (sigma2 + 1) * (sigma2 + 2)),
}

subs_opt_noise_D_even = {
    alpha: -(eta+delta),
    beta: -(gamma+epsilon+zeta+theta+iota),
kappa: 0,
pi_ : 0,
mu : 0,
tau : 0,
    rho:-lambd,
}

subs_opt_noise_D_odd = {
    alpha: -(eta+delta),
    beta: -(gamma+epsilon+zeta+theta+iota),
    kappa: 0,
    pi_ : 0,
    mu : 0,
    tau : 0,
    nu: 0,
    omicron: 0,
    rho:-lambd,
}
subs_opt_noise_D_mean = {
    kappa:0,
    mu:0,
    pi_:0,
    tau: 0,
    sigma2: 1,
}
subs = {
    D11: 0,
    D13: 0,
    D31: 0,
    D33: 0,
}
subs_mapping = {
    # For noise = True, D = True
    "noise_D_even": (subs_noise_D, subs_opt_noise_D_even),
    "noise_D_odd": (subs_noise_D, subs_opt_noise_D_odd),
    "noise_D_combined": (subs_noise_D, subs_opt_noise_D_mean),
    # For noise = True, D = False
    "noise_noD_even": (subs_noise_noD, subs_opt_noise_noD_even),
    "noise_noD_mean": (subs_noise_noD, subs_opt_noise_noD_mean),

    # For noise = False, D = True
    "no_noise_D_odd": (subs_no_noise_D, subs_opt_no_noise_D_odd),
    "no_noise_D_even": (subs_no_noise_D, subs_opt_no_noise_D_even),
    "no_noise_D_combined": (subs_noise_D, subs_opt_no_noise_D_combined),

    # For noise = False, D = False
    "no_noise_noD_odd": (subs_no_noise_noD, subs_opt_no_noise_noD_odd),
    "no_noise_noD_even": (subs_no_noise_noD, subs_opt_no_noise_noD_even),
}