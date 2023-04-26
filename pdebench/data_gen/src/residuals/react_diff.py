import numpy as np
import h5py as h
import time


def reaction_diff_2d_residual(output_path, residual_path, sim_config):
    dataset = h.File(output_path, mode='r')
    residual_dataset = None
    while residual_dataset is None:
        try:
            residual_dataset = h.File(residual_path, mode='a')
        except IOError:
            residual_dataset = None
            time.sleep(.1)

    for seed in dataset.keys():
        data = np.asarray(dataset[seed]['data'])
        x = np.asarray(dataset[seed]['grid/x'])
        y = np.asarray(dataset[seed]['grid/y'])
        t = np.asarray(dataset[seed]['grid/t'])
        residuals = reaction_diff_2d_residual_compute(
            data[:, :, :, 0], data[:, :, :, 1], x, y, t, sim_config.k, sim_config.Du, sim_config.Dv)
        key = f'k={sim_config.k}_Du={sim_config.Du}_Dv={sim_config.Dv}/{seed}'
        residual_dataset.create_dataset(
            key, data=residuals, dtype='float32', compression='lzf')

    dataset.close()
    residual_dataset.close()


def partials(data, x, y, t):
    y_axis = 2
    x_axis = 1
    t_axis = 0
    data_x = np.gradient(data, x, axis=x_axis)
    data_xx = np.gradient(data_x, x, axis=x_axis)
    data_y = np.gradient(data, y, axis=y_axis)
    data_yy = np.gradient(data_y, y, axis=y_axis)
    data_t = np.gradient(data, t, axis=t_axis)
    return data_x, data_y, data_xx, data_yy, data_t


def reaction_diff_2d_residual_compute(u, v, x, y, t, k, dv, du):
    u_x, u_y, u_xx, u_yy, u_t = partials(u, x, y, t)
    v_x, v_y, v_xx, v_yy, v_t = partials(v, x, y, t)
    ru = u - (u ** 3) - k - v
    rv = u - v
    eqn1 = du * u_xx + du * u_yy + ru - u_t
    eqn2 = dv * v_xx + dv * v_yy + rv - v_t
    return eqn1 + eqn2
