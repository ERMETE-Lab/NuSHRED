import matplotlib.pyplot as plt
from matplotlib import patches, cm
import numpy as np
from scipy.interpolate import griddata


def get_msfr_geometry(ax, show_ticks = False):
    # Defining the rectangles for the blue zones
    rect_core = patches.Rectangle((0, -2.26/2), 2.05, 2.26, linewidth=1, facecolor='gray')
    rect_reflector_top = patches.Rectangle((0, 2.26/2), 2.25, 0.20, linewidth=1, facecolor='black')
    rect_reflector_bot = patches.Rectangle((0, -2.26/2), 2.25, -0.20, linewidth=1, facecolor='black')
    rect_reflector_lateral = patches.Rectangle((2.05, -2.26/2), 0.2, 2.26, linewidth=1, facecolor='black')
    rect_blanket = patches.Rectangle((1.13, -1.88/2), 0.7, 1.88, linewidth=1, facecolor='white')

    ax.add_patch(rect_core)
    ax.add_patch(rect_reflector_top)
    ax.add_patch(rect_reflector_bot)
    ax.add_patch(rect_reflector_lateral)
    ax.add_patch(rect_blanket)

    # Setting the limits and aspect
    ax.set_xlim(0, 2.25)
    ax.set_ylim(-2.66/2, 2.66/2)
    ax.set_aspect('equal')

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

def get_core_msfr_geometry(ax, show_ticks = False):

    rect_blanket = patches.Rectangle((1.13, -1.88/2), 0.7, 1.88, linewidth=1, facecolor='white')
    ax.add_patch(rect_blanket)

    # Setting the limits and aspect
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-2.26/2, 2.26/2)
    ax.set_aspect('equal')

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

def plot_contour(ax, domain, _snap, 
                 vec_mode_to_plot = None, 
                 cmap = cm.jet, levels = 20,
                 streamline_plot = False, density = 2, linewidth=0.75):
    
    if vec_mode_to_plot is not None:
        if vec_mode_to_plot == 'x':
            snap = _snap[:,::3]
        elif vec_mode_to_plot == 'y':
            snap = _snap[:,1::3]
        elif vec_mode_to_plot == 'z':
            snap = _snap[:,2::3]
        else:
            snap = np.linalg.norm(_snap.reshape(-1, 3), axis=1)
    else:
        snap = _snap

    cont = ax.tricontourf(*domain.T, snap, cmap = cmap, levels = levels)

    # Streamlines
    if streamline_plot:
        
        strm = ax.streamplot(*create_streamlines(domain, _snap), 
                            color='k', density=density, linewidth=linewidth, arrowstyle='->')

    # Add Blanket
    rect_blanket = patches.Rectangle((1.13, -1.88/2), 0.7, 1.88, linewidth=1, facecolor='white',
                                     edgecolor='black',
                                    zorder=3)

    ax.add_patch(rect_blanket)

    # Setting the limits and aspect
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-2.26/2, 2.26/2)
    ax.set_aspect('equal')

    ax.set_xticks([])
    ax.set_yticks([])

    return cont

def create_streamlines(domain, velocity):

    x_grid = np.linspace(0, 2.05, 50)
    y_grid = np.linspace(-2.26 / 2, 2.26 / 2, 50)
    X, Y = np.meshgrid(x_grid, y_grid)

    vel_2D = velocity.reshape(-1, 3)

    U_interp = griddata(domain, vel_2D[:,0], (X, Y), method='linear')
    V_interp = griddata(domain, vel_2D[:,2], (X, Y), method='linear')

    return X, Y, U_interp, V_interp

def plot_trajectory(axs, domain, vel_mag, particle_traj, time_idx,
                    plot_full_traj = True, cmap = cm.RdYlBu_r,
                    streamline_plot = None, levels=20,
                    density = 2, linewidth=0.75):

    rect_blanket = patches.Rectangle((1.13, -1.88/2), 0.7, 1.88, linewidth=1, facecolor='white',
                                    zorder=3)

    # Setting the limits and aspect
    axs.set_xlim(0, 2.05)
    axs.set_ylim(-2.26/2, 2.26/2)
    axs.set_aspect('equal')

    axs.set_xticks([])
    axs.set_yticks([])

    # Add velocity streamlines
    if streamline_plot is not None:
        
        strm = axs.streamplot(*streamline_plot, 
                            color='k', density=density, linewidth=linewidth, arrowstyle='->')

    cont = axs.tricontourf(*domain.T, vel_mag, levels = levels, cmap=cmap)
    
    if plot_full_traj:

        num_particle = int(particle_traj.shape[1] / 2)

        for kk in range(num_particle):
            axs.plot(particle_traj[:time_idx+1, kk * 2], particle_traj[:time_idx+1, kk * 2 + 1], '--', linewidth=2, color='purple', label="Trajectory", zorder=3)
            axs.scatter(particle_traj[0, kk * 2], particle_traj[0, kk * 2 + 1], color='purple', s=50, label="Start", zorder=3)
            axs.scatter(particle_traj[time_idx, kk * 2], particle_traj[time_idx, kk * 2 + 1], color='magenta', s=50, label="Current", zorder=3)
            axs.text(particle_traj[0, kk * 2], particle_traj[0, kk * 2 + 1], str(kk+1), color='black', fontsize=20, ha='left', va='bottom')
    else:
        axs.scatter(particle_traj[time_idx, 0], particle_traj[time_idx, 1], color='purple', s=50, label="Start")

    axs.add_patch(rect_blanket)
    plt.colorbar(cont)