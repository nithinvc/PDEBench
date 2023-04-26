import h5py as h
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def visualize_2d_reaction_diffusion(output_path, vis_path, sim_config):
    dataset = h.File(output_path, mode='r')
    for seed in dataset.keys():
        data = np.asarray(dataset[seed]['data'])
        t = np.asarray(dataset[seed]['grid/t'])
        current_vis_path = f'{vis_path}/{seed}'
        title = f'Du = {sim_config.Du:.4e}, Dv = {sim_config.Dv:.4e}, k = {sim_config.k:.4e}, Seed = {seed}'
        visualize_2d_reaction_diffusion_single(
            data, t, title, current_vis_path)


def visualize_2d_reaction_diffusion_single(data, t, title, vis_path):
    # Initialize plot
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)

    # Store the plot handle at each time step in the 'ims' list
    ims = []
    for i in range(data.shape[0]):
        time_title_u = ax[0].text(0.5, 1.07, f"u, t = {t[i]:.2f}", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                                  transform=ax[0].transAxes, ha="center")
        time_title_v = ax[1].text(0.5, 1.07, f"v, t = {t[i]:.2f}", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                                  transform=ax[1].transAxes, ha="center")
        im1 = ax[0].imshow(data[i, ..., 0].squeeze(), animated=True)
        im2 = ax[1].imshow(data[i, ..., 1].squeeze(), animated=True)
        if i == 0:
            # show an initial one first
            ax[0].imshow(data[0, ..., 0].squeeze())
            # show an initial one first
            ax[1].imshow(data[0, ..., 1].squeeze())
        ims.append([im1, im2, time_title_u, time_title_v])

    # Animate the plot
    ani = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save(f"{vis_path}.gif", writer=writer)
