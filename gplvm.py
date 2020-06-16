import scipy.optimize
import numpy as np
import gp


class GPLVM:
    sigma_n_y = 1.14  # TODO this is for the normal tactip, needs setting for others!

    def __init__(self, x, y, sigma_f=None, ls=None):
        """
        Take in x and y as np.arrays of the correct size and shape to be used
        """
        self.x = x
        self.y = y

        # print(y.shape)
        # print(x.shape)
        if sigma_f is None or ls is None:
            # optmise
            self.optim_hyperpars()
        else:
            # assuming hyperpars already optimised
            self.sigma_f = sigma_f
            self.ls = ls

    def max_log_like(self, hyper_pars, data):
        sigma_f, l_disp, l_mu = hyper_pars
        x, y = data

        # print(y.shape)
        d_cap, n_cap = y.shape

        s_cap = (1 / d_cap) * (y @ y.conj().T)

        # if not np.isscalar(s_cap):
        #     raise NameError("s_cap is not scalar!")

        ls = np.array([l_disp, l_mu])  # TODO confirm 2 ls works with calc_K

        k_cap = gp.calc_K(x, sigma_f, ls, self.sigma_n_y)

        r_cap = np.linalg.cholesky(k_cap)
        sign, logdet_K = np.linalg.slogdet(r_cap)

        part_1 = -(d_cap * n_cap) * 0.5 * np.log(2 * np.pi)
        part_2 = -d_cap * 0.5 * logdet_K

        # print("Here")
        # print(d_cap)
        # print(s_cap)
        # print(np.trace(np.linalg.inv(k_cap)))
        part_3 = -d_cap * 0.5 * np.trace(np.linalg.inv(k_cap) @ s_cap)

        neg_val = part_1 + part_2 + part_3
        # print(neg_val)
        return -neg_val  # because trying to find max with a min search

    def optim_hyperpars(self, x=None, y=None, start_hyperpars=None, update_data=False):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        if start_hyperpars is None:
            start_hyperpars = np.array(
                [1, 300, 5]
            )  # sigma_f , L_disp, L_mu respectively

        data = [x, y]
        # minimizer_kwargs = {"args": data}
        result = scipy.optimize.minimize(
            self.max_log_like,
            start_hyperpars,
            args=data,
            method="BFGS",
            options={"gtol": 0.01, "maxiter": 300},  # is this the best number?
        )
        # print(result)

        [sigma_f, l_disp, l_mu] = result.x
        self.sigma_f = sigma_f
        self.ls = [l_disp, l_mu]
        # print(result)
