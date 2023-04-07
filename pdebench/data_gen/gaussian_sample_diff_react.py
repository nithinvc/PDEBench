import numpy as np
import random

SEED = 3


def gen_params(mean, stddev, count):
    params = []
    while len(params) < count:
        param = np.random.normal(mean, stddev)
        while param <= 0:
            param = np.random.normal(mean, stddev)
        params.append(param)
    return params


def main():
    num_runs = 4
    num_params = 3
    Dv_mean = 5e-4
    Dv_sigma = 1e-2

    Du_mean = Dv_mean
    Du_sigma = Dv_sigma

    k_mean = Dv_mean
    k_sigma = Dv_sigma

    for run_id in range(num_runs):
        dv_run = 'sim.Dv='
        du_run = 'sim.Du='
        k_run = 'sim.k='

        dv_params = gen_params(Dv_mean, Dv_sigma, num_params)
        du_params = gen_params(Du_mean, Du_sigma, num_params)
        k_params = gen_params(k_mean, k_sigma, num_params)
        for du, dv, k in zip(du_params, dv_params, k_params):
            dv_run += f'{dv:.6e},'
            du_run += f'{du:.6e},'
            k_run += f'{k:.6e},'
        print(run_id)
        print(f'{dv_run[:-1]} {du_run[:-1]} {k_run[:-1]}')


if __name__ == "__main__":
    main()
