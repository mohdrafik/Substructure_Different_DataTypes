import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from listspecificfiles import ReadListFiles  # custom module to list files

class DataPlotter:
    """
    Class to visualize 1D .npy data arrays as linearly spaced scatter plots.

    Args:
        data_dir (str or Path): Path to the directory containing .npy files.
        base_dir (str or Path, optional): Root directory for results. If None, auto-detected from data_dir.
        save_results (bool): Whether to save plots to disk. Default is True.
        save_dir (str or Path, optional): Custom directory to save results. If None, defaults to 'results/rawData_XYPlot' inside base_dir.

    Methods:
        plot_simple(file, save_name=None): Generates a basic scatter plot.
        plot_complex(file, save_name=None): Generates a colored scatter plot with min/mean/max lines and auto-adjusted labels.
        run_all(complex_plot=False): Plots all .npy files in the folder using simple or complex mode.
    """

    def __init__(self, data_dir, base_dir=None, save_results=True, save_dir=None):
        self.data_dir = Path(data_dir)
        self.base_dir = Path(base_dir) if base_dir else self.data_dir.parent.parent
        self.save_results = save_results

        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = self.base_dir / "results" / "rawData_XYPlot"

        if self.save_results:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Use ReadListFiles to list .npy files with full paths
        self.files = ReadListFiles(data_path=self.data_dir.relative_to(self.base_dir), keyword=".npy")

    def _generate_save_path(self, filename_stem, suffix=".png"):
        return self.save_dir / f"{filename_stem}{suffix}"

    def plot_simple(self, file, save_name=None):
        data = np.load(file).flatten().reshape(-1, 1)
        x_data = np.linspace(data.min(), data.max(), len(data))

        plt.plot(x_data, data, '.', color='blue', markersize=0.4)
        plt.title(f"{file.stem} Linearly Spaced Vector")
        plt.xlabel('linearly spaced vector from data itself')
        plt.ylabel('original Values')
        plt.grid(True)

        if self.save_results:
            save_path = self._generate_save_path(save_name or file.stem)
            plt.savefig(save_path, dpi=300)

        plt.close()

    def plot_complex(self, file, save_name=None):
        data = np.load(file).flatten()
        x_data = np.linspace(data.min(), data.max(), len(data))

        hist_vals, bin_edges = np.histogram(data, bins=10000)
        dominant_bin_index = np.argmax(hist_vals)
        bin_start, bin_end = bin_edges[dominant_bin_index], bin_edges[dominant_bin_index + 1]
        flat_band = data[(data >= bin_start) & (data < bin_end)]

        if flat_band.size == 0:
            print(f"No flat band detected in file: {file.name}")
            return

        flat_min, flat_max, flat_mean = flat_band.min(), flat_band.max(), flat_band.mean()

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(x_data, data, c=data, cmap='viridis', s=0.5)

        flat_y_vals = [("Min", flat_min, 'red'), ("Mean", flat_mean, 'orange'), ("Max", flat_max, 'green')]
        used_y = []
        for label, y_val, color in flat_y_vals:
            plt.axhline(y=y_val, color=color, linestyle='--', linewidth=1)

            offset = 0
            while any(abs((y_val + offset) - y) < 0.002 * (flat_max - flat_min) for y in used_y):
                offset += 0.002 * (flat_max - flat_min)
            y_text = y_val + offset
            used_y.append(y_text)

            plt.text(x_data[0], y_text, f'{label}: {y_val:.8f}', color=color, fontsize=8, verticalalignment='bottom')

        plt.title(f"{file.stem} Linearly Spaced Vector")
        plt.xlabel('Linearly spaced vector from data')
        plt.ylabel('Original Values')
        plt.grid(True)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Intensity')

        if self.save_results:
            save_path = self._generate_save_path(save_name or file.stem + '_marked')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)

        plt.close()

    def run_all(self, complex_plot=False):
        print(f"Found {len(self.files)} .npy files.")
        t_start = time.time()

        for file in self.files:
            if complex_plot:
                self.plot_complex(file)
            else:
                self.plot_simple(file)

        print(f"All plots completed in {time.time() - t_start:.2f} seconds.")

if __name__ == "__main__":

    # from plot_dataModule import DataPlotter

    plotter = DataPlotter(
        # __init__(self, data_dir, base_dir=None, save_results=True, save_dir=None)
        data_dir="data/raw_npyData",  # relative to BASE_DIR
        base_dir="E:/Projects/substructure_3d_data/Substructure_Different_DataTypes"
    )
    plotter.run_all(complex_plot=True)