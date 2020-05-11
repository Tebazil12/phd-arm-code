def max_log_like(hyper_pars, data):
    [sigma_f, l, sigma_n] = hyper_pars
    [y, x_matrix] = data
    print(".", end='')
    return (sigma_f * l / sigma_n)
