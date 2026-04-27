import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from pathlib import Path
from shapely.geometry import box


def FireForestViz(fire_id: int) -> None:
    """
    Visualizes the temporal progression of a specific wildfire on a map.

    This function loads spatial data, filters it by fire ID, and generates
    a grid of maps showing the evolution of the burn area step by step
    overlaid on an OpenStreetMap basemap.

    Args:
        fire_id (int): The unique identifier of the wildfire to visualize.

    Returns:
        None: Displays a matplotlib figure.
    """

    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / "data" / "backtest" / "hist_data.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"File not found at: {data_path}")

    df = gpd.read_parquet(data_path)
    wildfire_data = (
        df[df.wildfire_id == fire_id].sort_values("prop_step").to_crs(epsg=3857)
    )

    steps = wildfire_data.prop_step.values
    dates = wildfire_data.date.values

    n_plots = len(steps)
    if n_plots == 0:
        return

    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    minx, miny, maxx, maxy = wildfire_data.total_bounds
    margin = 400

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, n_rows * 5), sharex=True, sharey=True
    )

    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (step, dt) in enumerate(zip(steps, dates)):
        ax = axes[i]

        wildfire_data.plot(ax=ax, color="lightgrey", edgecolor="none", alpha=0.3)

        day_data = wildfire_data[wildfire_data["prop_step"] == step]
        day_data.plot(ax=ax, color="red", edgecolor="black", linewidth=0.5)

        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)

        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)

        formatted_date = pd.to_datetime(dt).strftime("%Y-%m-%d : %H:%M")
        ax.set_title(f"Step: {step} \n {formatted_date}")
        ax.set_axis_off()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def generate_matrix_for_one_step(wildfire_id: int, step: int, bounds, resolution, gdf):
    """
    Generates a rasterized grid representing the burned area for a specific wildfire time step.

    The function creates a matrix where each cell value represents the fraction of
    the cell's area that has been burned, normalized between 0.0 and 1.0.

    Args:
        wildfire_id (int): Unique identifier of the wildfire.
        step (int): The specific propagation step to process.
        bounds (list): List of coordinates [minx, miny, maxx, maxy] defining the grid extent.
        resolution (float): The size of each grid cell in map units.
        gdf (GeoDataFrame): Spatial data containing wildfire geometries and attributes.

    Returns:
        np.ndarray: A 2D numpy array representing the burned intensity grid.
    """

    minx, miny, maxx, maxy = bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    step_gdf = gdf[(gdf.wildfire_id == wildfire_id) & (gdf.prop_step == step)]

    matrix = np.zeros((height, width))
    cell_area = resolution**2

    x_coords = np.arange(minx, maxx, resolution)
    y_coords = np.arange(maxy, miny, -resolution)

    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            cell = box(x, y - resolution, x + resolution, y)
            inter = step_gdf.intersection(cell)
            if not inter.is_empty.any():
                area_burned = inter.area.sum()
                matrix[i, j] = min(area_burned / cell_area, 1.0)

    return matrix


def generate_wildfire_propagation_grids(wildfire_id, margin, resolution, gdf):
    """
    Computes a sequence of propagation matrices for a given wildfire across all time steps.

    This function determines the global bounding box of the wildfire (including a margin),
    identifies all unique propagation steps, and generates a matrix for each step.

    Args:
        wildfire_id (int): Unique identifier of the wildfire.
        margin (float): Extra space added around the wildfire's total bounds.
        resolution (float): The size of each grid cell in map units.
        gdf (GeoDataFrame): Spatial data containing wildfire geometries.

    Returns:
        list[np.ndarray]: A list of 2D numpy arrays, one for each propagation step.
    """
    minx = gdf[gdf.wildfire_id == wildfire_id].total_bounds[0] - margin
    miny = gdf[gdf.wildfire_id == wildfire_id].total_bounds[1] - margin
    maxx = gdf[gdf.wildfire_id == wildfire_id].total_bounds[2] + margin
    maxy = gdf[gdf.wildfire_id == wildfire_id].total_bounds[3] + margin
    steps_to_check = gdf[gdf.wildfire_id == wildfire_id].prop_step.unique()
    bounds = [minx, miny, maxx, maxy]
    matrices = []

    for step in steps_to_check:
        matrices.append(
            generate_matrix_for_one_step(wildfire_id, step, bounds, resolution, gdf)
        )
    return matrices


def plot_matrix_wildfire_propagation(liste_matrices, n_cols=3):
    """
    Visualizes the sequence of wildfire propagation matrices in a grid of subplots.

    Each matrix is displayed using a color map (viridis) where values range from
    0 (not burned) to 1 (fully burned).

    Args:
        liste_matrices (list[np.ndarray]): List of 2D numpy arrays to visualize.
        n_cols (int, optional): Number of columns in the plot grid. Defaults to 3.

    Returns:
        None: Displays the plot using matplotlib.
    """
    n_images = len(liste_matrices)
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes_flat = axes
    else:
        axes_flat = axes.flatten()

    im = None
    for i in range(len(axes_flat)):
        if i < n_images:
            im = axes_flat[i].imshow(liste_matrices[i], cmap="viridis", vmin=0, vmax=1)
            axes_flat[i].set_title(f"Matrice {i}")
        else:
            axes_flat[i].axis("off")
    fig.colorbar(im, ax=axes_flat.tolist(), shrink=0.6, label="Échelle [0, 1]")
    plt.show()


def FireForestVizMatrix(wildfire_id: int, margin: int, resolution: int, gdf) -> None:
    """
    Orchestrates the full pipeline to generate and visualize wildfire propagation grids.

    This high-level function fetches the burned area data for a specific fire,
    converts the spatial geometries into a sequence of intensity matrices based
    on the provided resolution, and generates a comparative visualization.

    Args:
        wildfire_id (int): Unique identifier of the wildfire to visualize.
        margin (int): Buffer distance added around the wildfire extent to provide context.
        resolution (int): The spatial resolution (cell size) for the output matrices.

    Returns:
        None: This function outputs a Matplotlib plot directly.
    """

    matrices = generate_wildfire_propagation_grids(wildfire_id, margin, resolution, gdf)
    plot_matrix_wildfire_propagation(matrices)


if __name__ == "__main__":
    FireForestViz(5)
