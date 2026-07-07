from scipy.interpolate import griddata
import numpy as np

class Plotter():
    def __init__(self, nodes: np.ndarray):
        self.nodes = nodes
        self.N = int(np.sqrt(nodes.shape[0]))  # Assuming square grid

    def plot(self, ax, snap: np.ndarray, levels = 20, cmap = 'jet', 
             streamlines = True, streamline_color = 'white'):

        if snap.shape[0] == self.nodes.shape[0]:
            vec = None
        elif snap.shape[0] == self.nodes.shape[0] * 2:
            vec = np.zeros((self.nodes.shape[0], 2))
            vec[:, 0] = snap[::2]
            vec[:, 1] = snap[1::2]
            snap = np.linalg.norm(vec, axis=-1)
        else:
            raise ValueError("Snapshot shape does not match the number of nodes or twice the number of nodes.")
        
        c = ax.tricontourf(self.nodes[:, 0], self.nodes[:, 1], snap, levels=levels, cmap=cmap)

        if streamlines and vec is not None:
            X, Y = np.meshgrid(np.linspace(0, 1, self.N), np.linspace(0, 1, self.N))
            ux = griddata(self.nodes[:, :2], vec[:, 0], (X, Y), method='linear', fill_value=0)
            uy = griddata(self.nodes[:, :2], vec[:, 1], (X, Y), method='linear', fill_value=0)
            ax.streamplot(X, Y, ux, uy,color=streamline_color, density=2, linewidth=1, arrowsize=1)
        
        ax.set_xticks([])
        ax.set_yticks([])

        return c