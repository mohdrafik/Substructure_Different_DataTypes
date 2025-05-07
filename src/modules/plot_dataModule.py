import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from matplotlib.patches import Patch

# from path_manager import addpath
# addpath()  # custom module to add paths this is not needed in this case because this module is in the same directory as the plot_dataModule.py script

from listspecificfiles import readlistFiles  # custom module to list files

class DataPlotter:
    """
    Class to visualize .npy data and significant digit distributions.

    Args:
        data_dir (str or Path): Path to directory with .npy files.
        base_dir (str or Path, optional): Root directory for results. Auto-detected from data_dir if None.
        save_results (bool): Whether to save plots to disk. Default is True.
        save_dir (str or Path, optional): Custom save directory. Defaults to 'results/rawData_XYPlot' inside base_dir.

    Methods:
        plot_simple(file, save_name=None): Basic scatter plot.
        plot_complex(file, save_name=None): Colored scatter plot with min/mean/max lines.
        plot_histogram_with_annotate_counts(counts, values, ...): Colored 1-height bar plot with counts annotated.
        plot_horizontal_split_histogram(sig_digit_data, ...): Horizontal bar plot split by frequency category.
        run_all(complex_plot=False): Process and plot all files.
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

        self.files = readlistFiles(data_path=self.data_dir.relative_to(self.base_dir), keyword=".npy")

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

    @staticmethod
    def plot_histogram_with_annotate_counts(counts, values, title=None, xlabel=None, ylabel=None, saveplot=False, filename=None, save_dir=None, dpi=600):
        
        """
        Plot a clean, publication-ready histogram with counts mapped to color and bold annotations.
        Necessity:
        Data counts are differ in Counts from very high to very low. So I make all counts to 1 coorresponding to each unique value and annotate the original counts on top of each bar.
        This is useful for visualizing the distribution of significant digits in the data.
        Plot histogram with original counts annotated on top of each bar.

        Args:
            Counts_OFUniqueValues (list or array): Count values corresponding to unique values.
            UniqueValues (list or array): List of unique values (e.g., significant digits).
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            saveplot (bool): If True, saves the figure.
            filename (str): Base filename (no extension).
            save_dir (str): Directory to save the plot.
            dpi (int): Dots per inch for saving resolution.
        """
        x = np.array(values)
        counts = np.array(counts)
        y = np.ones_like(counts)

        norm = plt.Normalize(counts.min(), counts.max())
        cmap = plt.cm.viridis
        colors = cmap(norm(counts))

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(x, y, width=0.9, color=colors, edgecolor='black', linewidth=1.2)

        for xi, count in zip(x, counts):
            ax.text(xi, 1.008, f"{count}", ha='center', va='bottom', fontsize=4, fontweight='bold')

        ax.set_xlabel(xlabel if xlabel else "Significant Digits", fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel if ylabel else "Normalized Count (1 per bin)", fontsize=11, fontweight='bold')
        ax.set_title(title if title else "Histogram with Annotated Counts", fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_yticks([])
        ax.tick_params(axis='x', length=5, width=1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Original Count Value", fontsize=11, fontweight='bold')

        plt.tight_layout()

        if saveplot:
            if filename and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{filename}_histogram.png")
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                print(f" Plot saved at: {save_path}")
            else:
                print(" Cannot save: filename and save_dir must be specified.")
        else:
            print("saveplot is False; plot will not be saved.")

        plt.show()

    @staticmethod
    def plot_horizontal_split_histogram(significant_digit_data, title=None, xlabel=None, ylabel=None, saveplot=False, filename=None, save_dir=None, dpi=600):
        """
        here significant_digit_data is a list of significant digits in my data values, this is counted and plot as horizontal split histogram 
        Plot a horizontal histogram with bars color-coded by frequency level:
        very low (highlighted), low, moderate, and high.

        Args:
            significant_digit_data (list): Input digit list.
            title (str): Plot title. If None, default will be used.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            saveplot (bool): Whether to save the plot.
            filename (str): Filename without extension.
            save_dir (str): Directory to save the plot.
            dpi (int): Resolution for saved plot.
        """

        counter = Counter(significant_digit_data)
        digits = np.array(sorted(counter.keys()))
        counts = np.array([counter[d] for d in digits])

        very_low_thresh = np.percentile(counts, 10)
        low_thresh = np.percentile(counts, 33)
        high_thresh = np.percentile(counts, 66)

        colors = []
        for c in counts:
            if c <= very_low_thresh:
                colors.append("darkred")
            elif c < low_thresh:
                colors.append("skyblue")
            elif c < high_thresh:
                colors.append("gold")
            else:
                colors.append("salmon")

        sns.set(style="whitegrid", font_scale=1.1, rc={
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'font.family': 'serif'
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(digits, counts, color=colors, edgecolor='black', height=0.7)

        for d, c in zip(digits, counts):
            ax.text(c + max(counts)*0.01, d, f"{c}", va='center', fontsize=9)

        ax.set_title(title if title else "Horizontal Histogram", pad=10)
        ax.set_xlabel(xlabel if xlabel else "Count", labelpad=6)
        ax.set_ylabel(ylabel if ylabel else "Significant Digit", labelpad=6)
        ax.set_yticks(digits)

        legend_elements = [
            Patch(facecolor='darkred', edgecolor='black', label='Very Low'),
            Patch(facecolor='skyblue', edgecolor='black', label='Low'),
            Patch(facecolor='gold', edgecolor='black', label='Moderate'),
            Patch(facecolor='salmon', edgecolor='black', label='High')
        ]
        ax.legend(handles=legend_elements, title="Frequency Level", loc='best')

        plt.tight_layout()

        if saveplot:
            if filename and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{filename}_horizontal_histogram.png")
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                print(f" Plot saved at: {save_path}")
            else:
                print(" Cannot save: provide both filename and save_dir.")
        else:
            print("Plot not saved (saveplot=False).")

        plt.show()

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

    from plot_dataModule import DataPlotter
    # Example data usage
    plotter = DataPlotter(
        # __init__(self, data_dir, base_dir=None, save_results=True, save_dir=None)
        data_dir="data/raw_npyData",  # relative to BASE_DIR
        base_dir="E:/Projects/substructure_3d_data/Substructure_Different_DataTypes"
    )
    plotter.run_all(complex_plot=True)



    from plot_dataModule import DataPlotter

    # Example data usage
    from plot_dataModule import DataPlotter
    counts = [1, 10, 50, 200, 800, 3000]
    values = [1, 2, 3, 4, 5, 6]
    significant_digits = [1, 2, 2, 3, 3, 3, 4, 5, 6]

    # Call directly on the class (no instance required)
    DataPlotter.plot_histogram_with_annotate_counts(
        counts, values,
        title="Annotated Histogram",
        saveplot=False
    )

    DataPlotter.plot_horizontal_split_histogram(
        significant_digits,
        title="Horizontal Split Histogram",
        saveplot=False
    )

# counts = np.array([      1,      20,     253,    2466,   25466,  257152, 2461352,  5670279, 1041250,   97875,   10690,     838])
# values = [val for val in range(1, 13)]
# significant_digit_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
