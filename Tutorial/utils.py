import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

def solve_kolmogorov_2d(viscosity: float, N: int = 64, dt: float = None, t_max: float = 15.0, n_force: int = 4, num_snapshots: int = 50, cfl: float = 0.5):
    """
    Solves the 2D Kolmogorov flow and returns the vorticity history.
    
    Parameters:
    viscosity     : Kinematic viscosity (nu)
    N             : Grid resolution (N x N points)
    dt            : Time step size. If None, it is automatically computed based on the CFL condition.
    t_max         : Total simulation time
    n_force       : Wavenumber of the sinusoidal forcing
    num_snapshots : Number of frames to save evenly across the simulation duration
    cfl           : Courant number used for the automatic dt calculation (default 0.5)
    
    Returns:
    times         : 1D array of time values
    omega_hist    : 2D array of flattened vorticity fields (N*N, T_saved)
    Re_hist       : 1D array of the dynamic Reynolds number at each saved snapshot
    """
    
    # 1. Domain Setup [0, 2pi] x [0, 2pi]
    L = 2 * np.pi
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # --- Sanity Check & Automatic dt Calculation ---
    # The laminar peak velocity for Kolmogorov flow is U_max = 1 / (nu * n^2)
    # We add a safety multiplier (2.0) to account for turbulent bursts
    u_max_est = max(1.0, (1.0 / (viscosity * n_force**2)) * 2.0)
    dt_safe = cfl * dx / u_max_est

    if dt is None:
        dt = dt_safe
    elif dt > dt_safe:
        warnings.warn(f"Provided dt ({dt}) exceeds the estimated safe dt ({dt_safe:.6f}). The simulation may become numerically unstable.")

    num_steps = int(t_max / dt)
    
    # --- Snapshot Calculation ---
    save_every_n = max(1, num_steps // num_snapshots)

    # 2. Spectral Wavenumbers
    kx = np.fft.fftfreq(N, d=L/(2*np.pi*N))
    ky = np.fft.fftfreq(N, d=L/(2*np.pi*N))
    KX, KY = np.meshgrid(kx, ky)
    
    K2 = KX**2 + KY**2
    K2[0, 0] = 1e-10 

    # 3. Dealiasing Mask (Orszag's 2/3 Rule)
    kmax = N / 3
    dealias_mask = (np.abs(KX) < kmax) & (np.abs(KY) < kmax)

    # 4. Kolmogorov Forcing setup
    f_omega = -n_force * np.cos(n_force * Y)
    f_omega_hat = np.fft.fft2(f_omega) * dealias_mask

    # 5. Initial Conditions (Small random noise)
    np.random.seed(42)
    omega = np.random.randn(N, N) * 0.1
    omega_hat = np.fft.fft2(omega) * dealias_mask

    exp_visc = np.exp(-viscosity * K2 * dt)

    # --- History Storage Initialization ---
    times = []
    omega_hist = []
    Re_hist = []

    # 6. Main Integration Loop
    for step in tqdm(range(1, num_steps + 1), desc="Solving for nu={:.4e}".format(viscosity)):
        
        # A. Streamfunction
        psi_hat = omega_hat / K2
        psi_hat[0, 0] = 0

        # B. Velocities
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))

        # C. Spatial derivatives
        domega_dx_hat = 1j * KX * omega_hat
        domega_dy_hat = 1j * KY * omega_hat
        domega_dx = np.real(np.fft.ifft2(domega_dx_hat))
        domega_dy = np.real(np.fft.ifft2(domega_dy_hat))

        # D. Nonlinear advection
        nl = -(u * domega_dx + v * domega_dy)
        nl_hat = np.fft.fft2(nl) * dealias_mask

        # E. Time Step Update
        omega_hat = (omega_hat + dt * (nl_hat + f_omega_hat)) * exp_visc

        # F. Save State
        if step % save_every_n == 0:
            omega_current = np.real(np.fft.ifft2(omega_hat))
            if np.isfinite(omega_current).all():
                # Save vorticity and time
                omega_hist.append(omega_current.copy().flatten())
                times.append(step * dt)
                
                # Calculate and save Reynolds number
                U_rms = np.sqrt(np.mean(u**2 + v**2))
                current_Re = U_rms / (viscosity * n_force)
                Re_hist.append(current_Re)
            else:
                raise ValueError("Non-finite values detected in vorticity field at time {:.3f}".format(step * dt))

    return np.array(times), np.array(omega_hist).T, np.array(Re_hist)