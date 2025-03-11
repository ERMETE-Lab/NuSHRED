import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, patches
import seaborn as sns
import pandas as pd

import imageio.v2 as imageio
import os
import cv2

def extract_dynasty_coordinates():
    data = {
        "CV_code": range(1, 123),  # Example CV codes
        "Component": [
            *["GV1"]*2, *["Cooler"]*(26-5), *["GV2"]*(61-26), *["GO1"]*(92-61), *["GV1"]*(125-92)
        ]
    }
    df = pd.DataFrame(data)

    # Loop dimensions
    loop_dimension = 3.05
    half_dim = loop_dimension / 2

    # Segment lengths based on the number of control volumes
    num_gv1 = sum(df['Component'] == 'GV1')
    num_cooler = sum(df['Component'] == 'Cooler')
    num_gv2 = sum(df['Component'] == 'GV2')
    num_go1 = sum(df['Component'] == 'GO1')

    # Calculate x, y coordinates for each segment
    coords = []

    # GV1 (left leg) goes from bottom (-half_dim, -half_dim) to top (-half_dim, +half_dim)
    y_gv1 = np.linspace(-half_dim, half_dim, num_gv1)
    x_gv1 = np.full(num_gv1, -half_dim)
    coords.extend(zip(x_gv1, y_gv1))

    # Cooler (top leg) goes from left (-half_dim, +half_dim) to right (+half_dim, +half_dim)
    x_cooler = np.linspace(-half_dim, half_dim, num_cooler)
    y_cooler = np.full(num_cooler, half_dim)
    coords.extend(zip(x_cooler, y_cooler))

    # GV2 (right leg) goes from top (+half_dim, +half_dim) to bottom (+half_dim, -half_dim)
    y_gv2 = np.linspace(half_dim, -half_dim, num_gv2)
    x_gv2 = np.full(num_gv2, half_dim)
    coords.extend(zip(x_gv2, y_gv2))

    # GO1 (bottom leg) goes from right (+half_dim, -half_dim) to left (-half_dim, -half_dim)
    x_go1 = np.linspace(half_dim, -half_dim, num_go1)
    y_go1 = np.full(num_go1, -half_dim)
    coords.extend(zip(x_go1, y_go1))

    # Rearrange the coordinates to start from the second-to-last point of GV1
    coords = coords[(125-92):] + coords[:(125-92)]  # Move last two points of GV1 to the start

    # Assign coordinates to DataFrame
    df['x'], df['y'] = zip(*coords)

    return np.asarray(df['x'].to_numpy()), np.asarray(df['y'].to_numpy())

def add_tubes(ax, _s, facecolors = ['none']*2):
    outer_rect = patches.Rectangle((-3.05/2 - _s, -3.05/2 - _s), 3.05 + 2 * _s, 3.05 + 2 * _s, linewidth=1, edgecolor='k', facecolor=facecolors[0])
    inner_rect = patches.Rectangle((-3.05/2 + _s, -3.05/2 + _s), 3.05 - 2 * _s, 3.05 - 2 * _s, linewidth=1, edgecolor='k', facecolor=facecolors[1])
    
    ax.add_patch(outer_rect)
    ax.add_patch(inner_rect)

def loop_heatmap(ax, snap, cmap=cm.jet, s=100,
                 _s_coeff = None, show_ticks=False, vmin=None, vmax=None):

    # Create and add the rectangles
    if _s_coeff is not None:
        _s = s / _s_coeff
        add_tubes(ax, _s)
    
    # Plot the loop
    coords = extract_dynasty_coordinates()
    sc = ax.scatter(*coords, c=snap, cmap=cmap, s=s, marker='s', vmin=vmin, vmax=vmax)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    return sc 

def plot_FOM_vs_Recon(x, t, fom: np.ndarray, recons: dict, std_recons: dict = None,
                      cmap = sns.color_palette('icefire', as_cmap=True), cmap_res = cm.hot,
                      nlevels = 30, spatial_idx = [0.3, 0.6, 0.9],
                      ylabel = None, filename = None, figsize=[6,5],
                      fontsize=15, format = 'svg',
                      cut_train = None,
                      box = None):
    
    assert len(x) == fom.shape[0]
    assert len(t) == fom.shape[1]
    
    nrows = 3
    
    keys = list(recons.keys())
    ncols = len(keys) + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
    
    # FOM
    levels = np.linspace(fom.min(), fom.max(), nlevels)
    contour_plot(axs[1, 0], x, t, fom.T, title=r'FOM', levels=levels, cmap=cmap, labels=[fontsize, fontsize])
    
    # Reconstructions and Residuals
    for key_i in range(len(keys)):
        contour_plot(axs[1, key_i+1], x, t, recons[keys[key_i]].T, title=keys[key_i], levels=levels, cmap=cmap, labels=[fontsize, fontsize])
        contour_plot(axs[0, key_i+1], x, t, np.abs(fom - recons[keys[key_i]]).T, title='Residual - '+keys[key_i], levels=nlevels, cmap=cmap_res, labels=[fontsize, fontsize])
    
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    axs[0,0].grid(False)
    axs[0, 0].spines['top'].set_visible(False)
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].spines['left'].set_visible(False)
    axs[0, 0].spines['bottom'].set_visible(False)
    
    ## Line Plots
    assert len(spatial_idx) == ncols
    
    idx_to_plot = [int(ii) for ii in spatial_idx]
    
    colors = cmap(np.linspace(0.25,0.75,len(keys)+1))
    
    for kk in range(len(idx_to_plot)):
        
        # FOM
        axs[2, kk].plot(t, fom[idx_to_plot[kk]], color=colors[0], label='FOM')
        
        # Reconstruction
        for key_i in range(len(keys)):
            axs[2, kk].plot(t, recons[keys[key_i]][idx_to_plot[kk]], '--', color = colors[key_i+1], label=keys[key_i])

        # Standard deviation
        if std_recons is not None:
            for key_i in range(len(keys)):
                if std_recons[keys[key_i]] is not None:
                    axs[2, kk].fill_between(t, recons[keys[key_i]][idx_to_plot[kk]] - 1.96 * std_recons[keys[key_i]][idx_to_plot[kk]],
                                            recons[keys[key_i]][idx_to_plot[kk]] +  1.96 * std_recons[keys[key_i]][idx_to_plot[kk]],
                                            color=colors[key_i+1], alpha=0.3)

        axs[2, kk].set_title(r'Space $x_{idx}='+str(x[idx_to_plot[kk]])+'$', fontsize=fontsize)
        axs[2, kk].grid()
        axs[2, kk].legend(framealpha=1, fontsize=fontsize)
        axs[2, kk].set_xlabel(r'Time $t$ (s)', fontsize=fontsize)
        axs[2, kk].set_xlim(0, max(t))

        if cut_train is not None:
            axs[2, kk].axvline(x=cut_train, color='k', linestyle='--', label='Training cut')
        
    if ylabel is not None:
        axs[2,0].set_ylabel(ylabel, fontsize=fontsize)

    if box is not None:
        axs[0, 0].annotate(box, xy=(0.35, 0.3), xycoords='axes fraction', fontsize=fontsize,
                   bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.5'),
                   ha='center', va='center')


    plt.tight_layout()
    
    if filename is not None:
        plt.savefig(filename+'.'+format, format=format, dpi=250, bbox_inches='tight')
    else:
        plt.show()

def contour_plot(ax, x, t, umatrix, cmap=cm.jet, title=None, labels = [None, None], **kwargs):
    tgrid, xgrid = np.meshgrid(t, x)
    
    cont = ax.contourf(tgrid, xgrid, umatrix.T, cmap=cmap, **kwargs)
    
    if labels[1] is not None:
        ax.set_ylabel(r'Space $x$', fontsize=labels[0])
    if labels[0] is not None:
        ax.set_xlabel(r'Time $t$', fontsize=labels[1])
    
    ax.set_ylim(x[0], x[-1])
    ax.set_xlim(t[0], t[-1])
    
    if title is not None:
        ax.set_title(title)
    plt.colorbar(cont, ax=ax)
    
def plot_shred_error_bars(relative_test_errors: list[float]):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(25,5))

    n_configurations = len(relative_test_errors)

    # Bars
    axs[0].bar(np.arange(1, n_configurations+1, 1), relative_test_errors, 
            color = cm.RdYlBu(np.linspace(0,1,len(relative_test_errors))),
            edgecolor='k')

    axs[0].set_xticks(np.arange(1, n_configurations+1, 1))
    axs[0].set_xlabel(r'Configurations', fontsize=20)
    axs[0].set_ylabel(r'Relative Test Error $\varepsilon_{2}$', fontsize=20)

    # Histogram
    axs[1].hist(relative_test_errors, edgecolor='k', density=True)
    axs[1].set_xlabel(r'Relative Test Error $\varepsilon_{2}$', fontsize=20)

    [ax.tick_params(axis='both', labelsize=15) for ax in axs]

    plt.tight_layout()
    
    plt.show()

def add_exp_position(ax, exp_idx, show_labels=False, fontsize=15, textcolor = 'black', **kwargs):
    dyn_coord = extract_dynasty_coordinates()
    ax.scatter(dyn_coord[0][exp_idx], dyn_coord[1][exp_idx], **kwargs)

    if textcolor == 'black':
        textcolor = ['black'] * len(exp_idx)

    if show_labels:
        labels = ['TC1', 'TC2', 'TC3', 'TC4']
        for i, label in enumerate(labels):
            ax.annotate(label, 
                        (dyn_coord[0][exp_idx[i]], dyn_coord[1][exp_idx[i]]),  # Position of annotation
                        textcoords="offset points",  # Set relative position
                        xytext=(10, 5),  # Offset by 5 points in both x and y directions
                        ha='center',  # Horizontal alignment
                        fontsize=fontsize,   # Font size for annotation text
                        color=textcolor[i])  # Color of the annotation text

def plot_loop_FOM_vs_SHRED(fom, shred, times, time_idx, lags,
                           exp_idx, std_shred = None, exp_data = None,
                           cut_train = None, lims = None, sampling_exp = 20, loc_legend = None,
                           cmap = cm.jet, sens_color = None, filename=None):

    if lims is None:
        lims = [fom.min(), fom.max()]

    ncols = 3
    nrows = 2
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(6 * ncols, 5 * nrows))

    fom_plot = loop_heatmap(axs[0,0], fom[:, time_idx],
                            s = 250, _s_coeff = 2500, cmap = cmap,
                            vmin = lims[0], vmax=lims[1])

    shred_plot = loop_heatmap(axs[1,0], shred[:, time_idx - lags],
                            s = 250, _s_coeff = 2500, cmap = cmap,
                            vmin = lims[0], vmax=lims[1])

    # Add title
    axs[0,0].set_title('FOM - Time = {:.2f} min'.format(times[time_idx] / 60), fontsize=25)
    axs[1,0].set_title('SHRED - Time = {:.2f} min'.format(times[time_idx] / 60), fontsize=25)

    # Add exp position
    if sens_color is None:
        sens_color = ['black'] * len(exp_idx)
    add_exp_position(axs[0,0], exp_idx, show_labels=True, marker='*', s=100, color='black', textcolor=sens_color)
    add_exp_position(axs[1,0], exp_idx, show_labels=True, marker='*', s=100, color='black', textcolor=sens_color)

    # Colorbars
    fig.colorbar(fom_plot, ax=axs[0,0]).set_label('Temperature (-)', fontsize=20)
    fig.colorbar(shred_plot, ax=axs[1,0]).set_label('Temperature (-)', fontsize=20)

    # TC plot over time
    tc_axs = axs[:, 1:].flatten()
    if loc_legend is None:
        loc_legend = ['lower right', 'upper left', 'upper left', 'upper left']
    for ii in range(len(exp_idx)):
        if exp_data is not None:
            time_exp_idx = np.where(exp_data['Time'] <= times[time_idx])[0]
            # tc_axs[ii].plot(exp_data['Time'][time_exp_idx][::sampling_exp] / 60, exp_data['TC'][ii, time_exp_idx][::sampling_exp], 'o', label='Exp', color='green', alpha=0.5)
            tc_axs[ii].errorbar(exp_data['Time'][time_exp_idx][::sampling_exp] / 60, exp_data['TC'][ii, time_exp_idx][::sampling_exp], yerr=2.15 / (335 -295), alpha=0.35, color='green',
            errorevery=sampling_exp*2, fmt='o', label='Exp')

        tc_axs[ii].plot(times[:time_idx] / 60, fom[exp_idx[ii],   :time_idx], label='FOM', color='red')
        tc_axs[ii].plot(times[lags:time_idx] / 60, shred[exp_idx[ii], :time_idx - lags], label='SHRED', color='blue')
        
        if std_shred is not None:
            tc_axs[ii].fill_between(times[lags:time_idx] / 60,
                                    shred[exp_idx[ii], :time_idx - lags] - std_shred[exp_idx[ii], :time_idx - lags],
                                    shred[exp_idx[ii], :time_idx - lags] + std_shred[exp_idx[ii], :time_idx - lags],
                                    color='blue', alpha=0.2, label='95% CI')

        if cut_train is not None:
            tc_axs[ii].axvline(x=times[cut_train]/60, color='k', linestyle='--', label='Training cut')
        tc_axs[ii].set_title(f'TC{ii+1}', fontsize=35)
        tc_axs[ii].grid()
        tc_axs[ii].legend(framealpha=1, fontsize=16, ncols=2, loc=loc_legend[ii])
        tc_axs[ii].tick_params(axis='both', labelsize=15)
        tc_axs[ii].set_xlabel('Time (min)', fontsize=20)
        tc_axs[ii].scatter(times[time_idx]/60, fom[exp_idx[ii], time_idx], color='red', marker='*', s=100)

        tc_axs[ii].set_xlim(min(times)/60, max(times)/60)
        tc_axs[ii].set_ylim(fom.min(), fom.max())

    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename, format='png', dpi=250, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig

def make_gif(path, gif_path, duration = 250):
    # Sort image files numerically based on the extracted time value
    image_files = [img for img in os.listdir(path) if img.endswith('.png')]
    image_files = sorted(image_files, key=lambda x: float(x.split('time')[1].split('.png')[0]))

    # Create the GIF by loading each image and adding it to the array
    with imageio.get_writer(gif_path, mode='I', duration=duration, loop=0) as writer:
        for img_file in image_files:
            img_path = os.path.join(path, img_file)
            image = imageio.imread(img_path)
            writer.append_data(image)


def make_mp4(path, video_path, fps=30):
    # Sort image files numerically based on the extracted time value
    image_files = [img for img in os.listdir(path) if img.endswith('.png')]
    image_files = sorted(image_files, key=lambda x: float(x.split('time')[1].split('.png')[0]))

    # Create the MP4 video by loading each image and adding it to the array
    with imageio.get_writer(video_path, fps=fps, codec="libx264") as writer:
        for img_file in image_files:
            img_path = os.path.join(path, img_file)
            image = imageio.imread(img_path)
            writer.append_data(image)

def make_mp4_opencv(path, video_path, fps=30):
    # Sort image files numerically based on the extracted time value
    image_files = [img for img in os.listdir(path) if img.endswith('.png')]
    image_files = sorted(image_files, key=lambda x: float(x.split('time')[1].split('.png')[0]))

    # Read the first image to get frame dimensions
    first_image = cv2.imread(os.path.join(path, image_files[0]))
    height, width, _ = first_image.shape

    # Define the codec and create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write each image to the video
    for img_file in image_files:
        img_path = os.path.join(path, img_file)
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()
