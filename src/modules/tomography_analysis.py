import numpy as np
import plotly.graph_objects as go
import argparse

def anisotropic_diffusion_3d(volume, niter=10, kappa=50, gamma=0.1, step=(1.,1.,1.), option=1):
    """
    Apply 3D anisotropic diffusion (Perona-Malik filtering) to a volume.
    """
    vol = volume.astype(np.float32)
    vol_out = vol.copy()
    # Initialize gradient difference arrays
    deltaD = np.zeros_like(vol_out)
    deltaS = np.zeros_like(vol_out)
    deltaE = np.zeros_like(vol_out)
    # Initialize flux arrays
    NS = np.zeros_like(vol_out)
    EW = np.zeros_like(vol_out)
    UD = np.zeros_like(vol_out)
    for _ in range(niter):
        # Compute differences along each axis
        deltaD[:-1, :, :] = np.diff(vol_out, axis=0);   deltaD[-1, :, :] = 0
        deltaS[:, :-1, :] = np.diff(vol_out, axis=1);   deltaS[:, -1, :] = 0
        deltaE[:, :, :-1] = np.diff(vol_out, axis=2);   deltaE[:, :, -1] = 0
        # Compute conductance (edge stopping) coefficients
        if option == 1:
            cD = np.exp(-(deltaD / kappa)**2) / step[0]
            cS = np.exp(-(deltaS / kappa)**2) / step[1]
            cE = np.exp(-(deltaE / kappa)**2) / step[2]
        else:  # option == 2
            cD = 1.0 / (1.0 + (deltaD / kappa)**2) / step[0]
            cS = 1.0 / (1.0 + (deltaS / kappa)**2) / step[1]
            cE = 1.0 / (1.0 + (deltaE / kappa)**2) / step[2]
        # Flux in each direction
        D = cD * deltaD
        S = cS * deltaS
        E = cE * deltaE
        # Net flux (divergence)
        UD[:] = D;   UD[1:, :, :] -= D[:-1, :, :]
        NS[:] = S;   NS[:, 1:, :] -= S[:, :-1, :]
        EW[:] = E;   EW[:, :, 1:] -= E[:, :, :-1]
        # Update volume
        vol_out += gamma * (UD + NS + EW)
    return vol_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D tomographic data processing with anisotropic diffusion and quantile thresholding")
    parser.add_argument("input_file", help="Path to input 3D .npy file")
    parser.add_argument("--iterations", "-n", type=int, default=10, help="Number of diffusion iterations")
    parser.add_argument("--kappa", "-k", type=float, default=30.0, help="Conduction coefficient for anisotropic diffusion")
    parser.add_argument("--gamma", "-g", type=float, default=0.1, help="Diffusion speed (time step)")
    args = parser.parse_args()

    # Load data
    input_file = "data\intermdata1\tomo_Grafene_24h.npy"
    data = np.load(args.input_file)
    print(f"Loaded volume of shape {data.shape}, dtype {data.dtype}")
    print(f"Mean={data.mean():.4f}, Std={data.std():.4f}, Min={data.min():.4f}, Max={data.max():.4f}")
    q95 = np.quantile(data, 0.95);  q99 = np.quantile(data, 0.99)
    print(f"95th percentile={q95:.4f}, 99th percentile={q99:.4f}")

    # Anisotropic diffusion filtering
    print(f"Applying anisotropic diffusion: niter={args.iterations}, kappa={args.kappa}, gamma={args.gamma}")
    filtered = anisotropic_diffusion_3d(data, niter=args.iterations, kappa=args.kappa, gamma=args.gamma, option=1)
    # Compute thresholds on filtered data
    thr95 = np.quantile(filtered, 0.95)
    thr99 = np.quantile(filtered, 0.99)
    print(f"Thresholding at 95% = {thr95:.4f}, 99% = {thr99:.4f}")

    # Prepare volume for visualization (clip values below 95th percentile)
    vol_display = filtered.copy()
    vol_display[vol_display < thr95] = thr95

    # Create Plotly volume plot
    fig = go.Figure(data=go.Volume(
        x=np.arange(filtered.shape[2]),
        y=np.arange(filtered.shape[1]),
        z=np.arange(filtered.shape[0]),
        value=vol_display,
        isomin=thr95,
        isomax=float(filtered.max()),
        opacity=0.1,
        surface_count=3,
        colorscale="Viridis",
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    fig.update_layout(scene=dict(aspectmode='data'),
                      title="3D Volume rendering (thresholded at 95th percentile)")
    # Show figure (in a script, this will open a browser window)
    fig.show()
