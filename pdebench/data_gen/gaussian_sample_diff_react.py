import numpy as np
import random

SEED = 3
random.seed(SEED)
np.random.seed(SEED)


def gen_params(mean, stddev, count):
    params = []
    while len(params) < count:
        param = np.random.normal(mean, stddev)
        while param <= 0:
            param = np.random.normal(mean, stddev)
        params.append(param)
    return params


def gen_params_uniform(min, max, count, global_params):
    params = []
    while len(params) < count:
        param = random.uniform(min, max)
        while param <= 0 and param not in params and param not in global_params:
            param = random.uniform(min, max)
        params.append(param)
        global_params.append(param)
    return params


def main():
    num_runs = 4
    num_params = 7
    k_min = .001
    k_max = .1

    Dv_min = .001
    Dv_max = .5

    Du_min = .0001
    Du_max = .5
    # Dv_mean = 5e-4
    # Dv_sigma = 1e-2

    # Du_mean = Dv_mean
    # Du_sigma = Dv_sigma

    # k_mean = Dv_mean
    # k_sigma = Dv_sigma
    k_global_params = []
    du_global_params = []
    dv_global_params = []
    for run_id in range(num_runs):
        dv_run = 'sim.Dv='
        du_run = 'sim.Du='
        k_run = 'sim.k='

        # dv_params = gen_params(Dv_mean, Dv_sigma, num_params)
        # du_params = gen_params(Du_mean, Du_sigma, num_params)
        # k_params = gen_params(k_mean, k_sigma, num_params)
        dv_params = gen_params_uniform(
            Dv_min, Dv_max, num_params, k_global_params)
        du_params = gen_params_uniform(
            Du_min, Du_max, num_params, du_global_params)
        k_params = gen_params_uniform(
            k_min, k_max, num_params, dv_global_params)

        for du, dv, k in zip(du_params, dv_params, k_params):
            dv_run += f'{dv:.6e},'
            du_run += f'{du:.6e},'
            k_run += f'{k:.6e},'
        print(run_id)
        print(f'{dv_run[:-1]} {du_run[:-1]} {k_run[:-1]}')


if __name__ == "__main__":
    main()
