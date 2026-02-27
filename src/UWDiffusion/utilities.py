import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import underworld3 as uw
from matplotlib.patches import Circle
from matplotlib.path import Path
from scipy.spatial import ConvexHull


lambda_235 = 9.8485e-10  # Decay constant for U-235 in yr^-1
lambda_238 = 1.55125e-10  # Decay constant for U-238 in yr^-1
current_ratio_U238_to_U235 = 137.818  # Present-day ratio of 238 to 235


# Calculate Concordia points
def concordia_curve(t, lambda_235, lambda_238):
    e_lambda_235t = np.exp(lambda_235 * t)
    e_lambda_238t = np.exp(lambda_238 * t)
    Pb206_U238 = e_lambda_238t - 1
    Pb207_U235 = e_lambda_235t - 1
    Pb207_Pb206 = (Pb207_U235 / Pb206_U238) * (1 / current_ratio_U238_to_U235)
    return Pb207_Pb206, Pb206_U238

def plot_terra_wasserburg_plot(
    start_time, end_time, marker_spacing, 
    ax=None, figsize=(10, 6), xlabel=None, ylabel=None, title=None, isotopic_system="U-Pb"
):
    """
    Generates a Terra-Wasserburg plot for a given isotopic system.

    Parameters:
    - start_time: float
        The starting time (in years) for the Concordia curve.
    - end_time: float
        The ending time (in years) for the Concordia curve.
    - marker_spacing: float
        Spacing (in years) between markers on the Concordia curve.
    - lambda_235: float
        Decay constant for the parent isotope 235 (e.g., U-235).
    - lambda_238: float
        Decay constant for the parent isotope 238 (e.g., U-238).
    - current_ratio_U238_to_U235: float
        Present-day ratio of 238 to 235 (e.g., 137.818 for the U-Pb system).
    - ax: matplotlib.axes.Axes, optional
        A pre-existing Matplotlib axis object. If None, a new figure and axis are created.
    - figsize: tuple, optional
        Size of the figure (default: (10, 6)).
    - xlabel: str, optional
        Label for the x-axis (default: "$^{238}U/^{206}Pb$").
    - ylabel: str, optional
        Label for the y-axis (default: "$^{207}Pb/^{206}Pb$").
    - title: str, optional
        Title for the plot (default: "Terra-Wasserburg").
    - isotopic_system: str, optional
        Isotopic system being used (default: "U-Pb").

    Returns:
    - ax: matplotlib.axes.Axes
        The axis object used for the plot, allowing further customization.
    """
    # Default axis labels and title
    if xlabel is None:
        xlabel = r"$^{238}\mathrm{U}/^{206}\mathrm{Pb}$"
    if ylabel is None:
        ylabel = r"$^{207}\mathrm{Pb}/^{206}\mathrm{Pb}$"
    # if title is None:
    #     title = f"Terra-Wasserburg {isotopic_system} Concordia Plot"



    # Generate the time intervals
    time_concordia = np.linspace(end_time, start_time, 1000)  # High-resolution Concordia curve
    time_markers = np.arange(end_time, start_time + marker_spacing, marker_spacing)  # Marker intervals


    
    pb207_pb206, pb206_u238 = concordia_curve(time_concordia, lambda_235, lambda_238)
    pb207_pb206_markers, pb206_u238_markers = concordia_curve(time_markers, lambda_235, lambda_238)

    # Create the axis if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the Concordia curve and markers
    ax.plot(1 / pb206_u238, pb207_pb206, label="Concordia Curve", color="black", linewidth=2)
    ax.scatter(1 / pb206_u238_markers, pb207_pb206_markers, color="green", edgecolors="black", s=80, zorder=5)

    # Annotate markers with the corresponding time
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_offset = 0.002 * (xlim[1] - xlim[0])  # 2% of plot width
    y_offset = 0.02 * (ylim[1] - ylim[0])  # 2% of plot height

    for i, (x, y) in enumerate(zip(1 / pb206_u238_markers, pb207_pb206_markers)):
        ax.text(x + x_offset, y + y_offset, f"{int(time_markers[i] / 1e6)} Ma", fontsize=10, ha="left", va="center", clip_on=True)

    # Add labels, title, grid, and legend
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    # ax.set_title(title, fontsize=14, pad=15)
    # ax.set_xlim(1, 150)
    # ax.set_ylim(0.01, 0.08)
    ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10, loc="best", frameon=True)

    return ax


def is_circle_inside_hex(center, radius, hex_vertices, n_points=32):
    # Generate points around the circle
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    circle_points = np.array([
        center + radius * np.array([np.cos(a), np.sin(a)])
        for a in angles
    ])
    hex_path = Path(hex_vertices)
    
    return np.all(hex_path.contains_points(circle_points))


def sample_spot(coords, datasets, center, radius):
    """
    Sample mesh data within a circular spot.

    Parameters:
    - coords: (N, 2) array of mesh coordinates (x, y)
    - datasets: array or list of arrays, each of shape (N,)
    - center: tuple (x, y) for spot center
    - radius: float, spot radius

    Returns:
    - mask: boolean array of shape (N,) for points inside the spot
    - sampled: list of arrays, each containing values inside the spot
    """

    # coords = is_circle_inside_hex(center, radius, coords)

    center = uw.scaling.non_dimensionalise(np.asarray(center))
    radius = uw.scaling.non_dimensionalise(radius)

    distances = np.linalg.norm(coords - center, axis=1)
    mask = distances <= radius
   
    # Support single or multiple datasets
    if isinstance(datasets, (list, tuple)):
        sampled = [np.asarray(data)[mask] for data in datasets]
    else:
        sampled = [np.asarray(datasets)[mask]]
    return sampled


def plot_mesh_data(coords, data, ax=None, levels=14, cmap='viridis', figsize=(8, 6), vmin=None, vmax=None):
    """
    Plots mesh data over the mesh coordinates using tricontourf.

    Parameters:
    - coords: (N, 2) array of mesh coordinates (x, y)
    - data: (N,) array of data values at each coordinate
    - ax: matplotlib.axes.Axes, optional
        A pre-existing Matplotlib axis object. If None, a new figure and axis are created.
    - levels: int, number of contour levels (default: 14)
    - cmap: str, colormap to use (default: 'viridis')
    - figsize: tuple, size of the figure (default: (8, 6))

    Returns:
    - fig, ax: Matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    contour = ax.tricontourf(coords[:, 0], coords[:, 1], data, levels=np.linspace(vmin, vmax, levels), cmap=cmap, extend='both')
    if ax is None:
        return fig, ax, contour
    else:
        return contour


### create a function to plot where the spot sample is
def plot_spot_sample(center, radius, ax=None, figsize=(8, 6), colour='k', linestyle='--', linewidth=2, alpha=1.0):
    """
    Plots the mesh coordinates and highlights the spot sample area.

    Parameters:
    - center: tuple (x, y) for spot center
    - radius: float, spot radius
    - ax: matplotlib.axes.Axes, optional
        A pre-existing Matplotlib axis object. If None, a new figure and axis are created.
    - figsize: tuple, size of the figure (default: (8, 6))

    Returns:
    - fig, ax: Matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # ax.scatter(coords[:, 0], coords[:, 1], s=5, color='blue', label='Mesh Points')


    # Default: black dashed line, no fill, label 'Spot Sample Area'
    circle = Circle(
        center, radius,
        edgecolor=colour, facecolor='none', linestyle=linestyle, linewidth=linewidth, label='Spot Sample Area', alpha=alpha
    )
    ax.add_patch(circle)

    if ax is None:
        return fig, ax


def _adams_moulton_flux(flux, flux_history, order=None, dt_current=None, dt_history=None):
    # Adams-Moulton coefficients for up to third order.
    # For variable timesteps, order=2 uses variable-step coefficients.
    # For order=3 with strongly variable timesteps, this falls back to variable-step order=2.
    if order is None:
        order = 1

    if dt_history is None:
        dt_history = []

    def _to_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    with sp.core.evaluate(False):
        if order == 1:
            return (flux + flux_history[0]) / 2

        if order == 2:
            dt_n = _to_float(dt_current)
            dt_nm1 = _to_float(dt_history[0]) if len(dt_history) > 0 else None

            if dt_n is None or dt_nm1 is None or dt_nm1 <= 0:
                return (5 * flux + 8 * flux_history[0] - flux_history[1]) / 12

            r = dt_n / dt_nm1
            if abs(r - 1.0) < 1.0e-12:
                return (5 * flux + 8 * flux_history[0] - flux_history[1]) / 12

            w0 = (2 * r + 3) / (6 * (1 + r))
            w1 = (3 + r) / 6
            w2 = -(r**2) / (6 * (1 + r))
            return w0 * flux + w1 * flux_history[0] + w2 * flux_history[1]

        if order == 3:
            dt_n = _to_float(dt_current)
            dt_nm1 = _to_float(dt_history[0]) if len(dt_history) > 0 else None
            dt_nm2 = _to_float(dt_history[1]) if len(dt_history) > 1 else None

            if (
                dt_n is not None
                and dt_nm1 is not None
                and dt_nm2 is not None
                and dt_nm1 > 0
                and dt_nm2 > 0
            ):
                r1 = dt_n / dt_nm1
                r2 = dt_nm1 / dt_nm2
                if abs(r1 - 1.0) < 0.05 and abs(r2 - 1.0) < 0.05:
                    return (9 * flux + 19 * flux_history[0] - 5 * flux_history[1] + flux_history[2]) / 24

                return _adams_moulton_flux(
                    flux,
                    flux_history,
                    order=2,
                    dt_current=dt_current,
                    dt_history=dt_history,
                )

            return (9 * flux + 19 * flux_history[0] - 5 * flux_history[1] + flux_history[2]) / 24

        raise NotImplementedError("Order > 3 not implemented for Adams-Moulton flux")