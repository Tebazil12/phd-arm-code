import scipy.optimize
import numpy as np
import gp

class GPLVM:
    sigma_n_y = 1.14  #TODO this is for the normal tactip, needs setting for others!

    def __init__(self, x, y, sigma_f=None, l=None):
        """
        Take in x and y as lists
        """

        x = np.array(x)
        print(x.shape)
        self.x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        y = np.array(y)
        self.y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])

        print(x.shape)
        print(y.shape)
        if sigma_f is None or l is None:
            #optmise
            self.optim_hyperpars()
        else:
            # assuming hyperpars already optimised
            self.sigma_f = sigma_f
            self.l = l

    def max_log_like(self, hyper_pars, data):
        sigma_f, l_disp, l_mu = hyper_pars
        x, y = data

        # print(y.shape)
        d_cap, n_cap = y.shape

        s_cap = (1 / d_cap) * (y @ y.conj().T)

        ls = np.array([l_disp, l_mu]) # TODO confirm 2 ls works with calc_K

        k_cap = gp.calc_K(x, sigma_f, ls, self.sigma_n_y)

        r_cap = np.linalg.cholesky(k_cap)
        sign, logdet_K = np.linalg.slogdet(r_cap)

        part_1 = - (d_cap * n_cap) * 0.5 * np.log(2 * np.pi)
        part_2 = - d_cap * 0.5 * logdet_K
        part_3 = - d_cap * 0.5 * np.trace(np.linalg.inv(k_cap) * s_cap)

        neg_val = part_1 + part_2 + part_3
        return -neg_val  # because trying to find max with a min search

    def optim_hyperpars(self, x=None, y=None, start_hyperpars=None, update_data=False):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        if start_hyperpars is None:
            start_hyperpars = np.array([1, 300, 5]) # sigma_f , L_disp, L_mu respectively

        data = [x, y]
        # minimizer_kwargs = {"args": data}
        result = scipy.optimize.minimize(self.max_log_like, start_hyperpars,
                                         args=data, method='BFGS')
        # print(result)

        [sigma_f, l_disp, l_mu] = result.x
        print(result)
        raise (stop)