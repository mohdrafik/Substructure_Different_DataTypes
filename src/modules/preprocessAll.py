import numpy as np
import os
from skimage.filters import threshold_otsu
import matplotlib.pyplot
from scipy.stats import norm
from itertools import cycle
from path_manager import addpath
addpath()


class DataPreprocessor:
    """
    A class for preprocessing data, including methods for analyzing and extracting features.
    
    A class for preprocessing data with support for methods like Otsu segmentation,
    peak detection using Freedman-Diaconis rule, quantile filtering, etc.
    Easily extendable for future preprocessing techniques.
    
    """

    def __init__(self, data, metadata=None):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy ndarray")
        self.data = data
        self.results = {}
        self.metadata = metadata or {}

    def apply_otsu_segmentation(self, save_masks_to=None):
        """
        Apply Otsu's thresholding method to segment foreground from background.
        Optionally save foreground and background masks as .npy files.
        """
        flat_data = self.data[self.data > 0].flatten()
        threshold = threshold_otsu(flat_data)
        fg_mask = self.data > threshold
        bg_mask = ~fg_mask

        if save_masks_to:
            os.makedirs(save_masks_to, exist_ok=True)
            np.save(os.path.join(save_masks_to, "fg_mask.npy"), fg_mask)
            np.save(os.path.join(save_masks_to, "bg_mask.npy"), bg_mask)

        self.results['otsu'] = {
            'threshold': threshold,
            'fg_mask': fg_mask,
            'bg_mask': bg_mask
        }
        return self.results['otsu']

    def apply_quantile_thresholding(self, lower_q=0.05, upper_q=0.95):
        """
        Apply quantile-based thresholding. Returns filtered data.
        """
        lower = np.quantile(self.data, lower_q)
        upper = np.quantile(self.data, upper_q)
        mask = (self.data >= lower) & (self.data <= upper)
        filtered_data = self.data[mask]

        self.results['quantile'] = {
            'lower_bound': lower,
            'upper_bound': upper,
            'filtered_data': filtered_data,
            'mask': mask
        }
        return self.results['quantile']

    @staticmethod
    def find_peak_bin(data, threshold_ratio=None):
        """
        Determine the highest peak in a dataset using a histogram with Freedman-Diaconis rule for bin width.

        Supports 1D arrays of scalar values or 3D point clouds/volumes with an associated scalar value.

        Parameters:
        -----------
        data : array-like
            Input data containing the scalar values. This can be:
             - A 1D array or list of numeric values.
             - A multi-dimensional array (e.g. 3D volume of values), which will be flattened.
             - A 2D array of shape (N, M) representing a point cloud. If M == 4, the last column is taken as the scalar values.
               If M == 3 (just coordinates with no associated scalar), a ValueError is raised.
        threshold_ratio : float, optional
            Fraction of the peak height to define an extended range around the peak. 
            - If None or 0 (default), returns only the exact peak bin interval.
            - If provided (e.g. 0.5 for 50%), includes all contiguous bins around the peak with counts >= threshold_ratio * peak_count.

        Returns:
        --------
        result : dict
            Dictionary with:
            - 'peak_value_range': tuple (min_val, max_val) of the selected peak bin or extended range.
            - 'selected_indices': indices of the original data values falling in the selected range. For multi-dimensional input, these are indices in the flattened array.
            - 'peak_bin_index': index of the peak bin in the histogram.

        Raises:
        -------
        ValueError:
            If data is empty.
            If data is a (N,3) array of coordinates with no scalar values (ambiguous input).
        """
        # Convert input to numpy array for uniform processing
        arr = np.asarray(data)
        if arr.size == 0:
            raise ValueError("Input data is empty.")
        
        # Extract a 1D array of scalar values from the input
        if arr.ndim == 1:
            values = arr
        elif arr.ndim == 2:
            if arr.shape[1] == 1:
                values = arr.ravel()
            elif arr.shape[1] == 4:
                values = arr[:, -1]
            elif arr.shape[1] == 3:
                raise ValueError("Data appears to be 3D coordinates with no scalar values provided.")
            else:
                values = arr.ravel()
        else:
            values = arr.ravel()
        
        n = values.size
        if n == 0:
            raise ValueError("Input data contains no values.")
        
        # Compute the interquartile range (IQR)
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        IQR = q75 - q25
        
        # Determine the number of bins using Freedman-Diaconis or a fallback
        if IQR == 0:
            min_val = float(values.min())
            max_val = float(values.max())
            if min_val == max_val:
                return {
                    'peak_value_range': (min_val, max_val),
                    'selected_indices': np.arange(n),
                    'peak_bin_index': 0
                }
            bin_count = int(np.ceil(np.log2(n) + 1))
            if bin_count < 1:
                bin_count = 1
        else:
            bin_width = 2 * IQR / (n ** (1/3))
            if bin_width <= 0:
                bin_count = int(np.ceil(np.log2(n) + 1))
            else:
                data_range = float(values.max() - values.min())
                bin_count = int(np.ceil(data_range / bin_width))
                if bin_count < 1:
                    bin_count = 1
        
        # Compute the histogram with the calculated number of bins
        hist, edges = np.histogram(values, bins=bin_count)
        peak_bin_idx = int(np.argmax(hist))
        
        # Determine the value range of the peak and (optionally) extended range
        if threshold_ratio is not None and threshold_ratio > 0:
            peak_count = hist[peak_bin_idx]
            threshold_count = peak_count * threshold_ratio
            start_idx = peak_bin_idx
            while start_idx > 0 and hist[start_idx - 1] >= threshold_count:
                start_idx -= 1
            end_idx = peak_bin_idx
            while end_idx < len(hist) - 1 and hist[end_idx + 1] >= threshold_count:
                end_idx += 1
            range_min_val = edges[start_idx]
            range_max_val = edges[end_idx + 1]
            peak_range = (float(range_min_val), float(range_max_val))
            if end_idx == len(hist) - 1:
                mask = (values >= range_min_val) & (values <= range_max_val)
            else:
                mask = (values >= range_min_val) & (values < range_max_val)
            selected_idx = np.nonzero(mask)[0]
        else:
            range_min_val = edges[peak_bin_idx]
            range_max_val = edges[peak_bin_idx + 1]
            peak_range = (float(range_min_val), float(range_max_val))
            if peak_bin_idx == len(hist) - 1:
                mask = (values >= range_min_val) & (values <= range_max_val)
            else:
                mask = (values >= range_min_val) & (values < range_max_val)
            selected_idx = np.nonzero(mask)[0]
        
        result = {
            'peak_value_range': peak_range,
            'selected_indices': selected_idx,
            'peak_bin_index': peak_bin_idx
        }
        
###############################################################################################
    @staticmethod
    def plot_histograms_from_directory(path, bins=800):
        import  matplotlib.pyplot as plt
        from pathlib import Path
        """
            Processes all .mat and .npy files in the specified directory.
            For each file, this method:
            - Loads and flattens the data
            - Filters out zero values
            - Plots histogram with envelope (frequency)
            - Fits and plots a Gaussian distribution over the data histogram

            Saves both plots as PNG files into a results subdirectory.

            Args:
                path (str or Path): Directory path containing the .mat or .npy files.
                bins (int): Number of bins to use for the histogram plots.
        """

        path = Path(path)
        results_dir = path / "results" / "histogramrawData"
        results_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in os.listdir(path) if f.endswith('.mat') or f.endswith('.npy')]
        print(f"Number of data files found: {len(files)}")

        for count, filename in enumerate(files, 1):
            print(f"Processing file {count}: {filename}")
            filepath = path / filename

            if filename.endswith('.mat'):
                datastruct = sio.loadmat(filepath)
                choose_keyVal = int(input(f"File {filename}: Enter 1 for 'elem_double' or 0 for 'mat_temp': "))
                data = datastruct['elem_double'] if choose_keyVal == 1 else datastruct['mat_temp']
            else:
                data = np.load(filepath)

            data = np.array(data).flatten()
            data = data[data != 0]

            clean_filename = filename.replace('.mat', '').replace('.npy', '')

            # Plot histogram with envelope
            plt.figure(figsize=(6, 5))
            counts, edges, _ = plt.hist(data, bins=bins, color='white', alpha=0.7, edgecolor='white')
            plt.fill_between(edges[:-1], counts, color='None', edgecolor='blue', alpha=0.7)
            plt.title('Envelope of data distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.tight_layout()
            envelope_path = results_dir / f"{clean_filename}_hist_envelope.png"
            plt.savefig(envelope_path, dpi=400, bbox_inches='tight')
            plt.close()
            print(f"Saved histogram and envelope: {envelope_path}")

            # Gaussian fit
            counts, bin_edges = np.histogram(data, bins=bins)
            mu, std = norm.fit(data)
            plt.figure(figsize=(6, 5))
            plt.hist(data, bins=bins, density=True, alpha=0.6, color='g', edgecolor='black')
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 1000)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            plt.title(f"Gaussian Fit: μ={mu:.4f}, σ={std:.4f}")
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.tight_layout()
            peak_path = results_dir / f"{clean_filename}_gaussian_peak.png"
            plt.savefig(peak_path, dpi=400, bbox_inches='tight')
            plt.close()
            print(f"Saved Gaussian peak fit: {peak_path}")



###############################################################################################
    @staticmethod
    def quantile_based_auto_threshold_plot(data, n_peaks=4, colors=None, show_subplots=False, save_dir=None):
        import matplotlib.pyplot as plt

        """
        Generates histograms based on quantile thresholds.

        Args:
            data (np.ndarray): Input data.
            n_peaks (int): Number of quantile thresholds to use.
            colors (list[str], optional): List of colors. Auto-generated if None.
            show_subplots (bool): If True, show subplots; otherwise, show a single overlay plot.
            save_dir (str or Path, optional): If provided, saves plots to this directory.

        Returns:
            list[dict]: List of computed quantile information (quantile, threshold, filtered_data, color).
        """
        def generate_colors(n):
            default_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'magenta', 'olive']
            return default_colors[:n] if n <= len(default_colors) else [default_colors[i % len(default_colors)] for i in range(n)]

        import os
        from pathlib import Path
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        qvalues = []
        colors = colors or generate_colors(n_peaks)

        for i in range(1, n_peaks + 1):
            qi = 1 / i
            threshold = np.quantile(data, qi)
            filtered_data = data[data > threshold]
            qvalues.append({
                "quantile": qi,
                "threshold": threshold,
                "filtered_data": filtered_data,
                "color": colors[i - 1]
            })

        if not show_subplots:
            plt.figure(figsize=(10, 6))
            plt.hist(data, bins='fd', alpha=0.6, label='Original Data', color='gray', edgecolor='black')
            for q in qvalues:
                plt.hist(q["filtered_data"], bins='fd', alpha=0.3, color=q["color"], label=f'> Q_{q["quantile"]:.2f}')
                plt.axvline(q["threshold"], linestyle='--', color=q["color"], linewidth=1.5,
                            label=f'Q_{q["quantile"]:.2f} = {q["threshold"]:.2f}')
            plt.title("Histogram with Quantile Thresholds")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / "quantile_overlaid_histogram.png", dpi=300)
            plt.show()
        else:
            n = len(qvalues) + 1
            cols = 2
            rows = (n + 1) // cols
            fig, axs = plt.subplots(rows, cols, figsize=(12, rows * 4))
            axs = axs.flatten()

            axs[0].hist(data, bins='fd', alpha=0.6, color='gray', label='Original Data', edgecolor='black')
            for q in qvalues:
                axs[0].hist(q["filtered_data"], bins='fd', alpha=0.3, color=q["color"], label=f'> Q_{q["quantile"]:.2f}')
                axs[0].axvline(q["threshold"], linestyle='--', color=q["color"], linewidth=1.5,
                               label=f'Q_{q["quantile"]:.2f} = {q["threshold"]:.2f}')
            axs[0].legend()
            axs[0].grid(True)
            axs[0].set_title("Histogram with Quantile Overlays")

            for idx, q in enumerate(qvalues, 1):
                axs[idx].hist(q["filtered_data"], bins='fd', alpha=0.6, color=q["color"], edgecolor='black')
                axs[idx].axvline(q["threshold"], linestyle='--', color='black', linewidth=1.2,
                                 label=f'Threshold = {q["threshold"]:.2f}')
                axs[idx].legend()
                axs[idx].grid(True)
                axs[idx].set_title(f"Filtered Data > Q_{q['quantile']:.2f}")

            for ax in axs[n:]:
                ax.set_visible(False)

            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / "quantile_subplots.png", dpi=300)
            plt.show()

        return qvalues
    # example usage 
    # qvalues = DataPreprocessor.quantile_based_auto_threshold_plot(
    #     data=my_array,
    #     n_peaks=4,
    #     show_subplots=True,
    #     save_dir="results/quantile_plots"
    # )




        
   


