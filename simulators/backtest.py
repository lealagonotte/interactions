import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from pathlib import Path


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


if __name__ == "__main__":
    FireForestViz(5)
