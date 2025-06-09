# use it, for not loosing the function name and it's doc string.
from functools import wraps
import scipy.io as sio
import numpy as np
import os
from scipy.signal import find_peaks
from scipy.stats import norm, gaussian_kde
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from itertools import cycle
from pathlib import Path
from scipy.io import savemat

from path_manager import addpath
addpath()


# @staticmethod  # on top if we are defining do't need @staticmethod.
def logfunction(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n ---------------------> /// Implementing method: {func.__name__} \\\ <------------------------------------------------------- \n")
        results = func(*args, **kwargs)
        # print(f"\n ---------------------> /// Finished executing method: {func.__name__} \\\ <--------------------------------------------------\n")
        return results
    # print(f"\n ---------------------> /// Finished executing method:  \\\ <--------------------------------------------------\n")
    return wrapper



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



    @staticmethod
    @logfunction
    def DataMasker(data, maskValue = None, masked_WithZero=False, masked_ANDRemoved=False):
        """
        A class method to create a mask for the data based on specified conditions.
        
        Parameters:
        -----------
        data : np.ndarray
            The input data array.
        masked_WithZero : bool, optional
            If True, masks the data with zeros where conditions are met.
        masked_ANDRemoved : bool, optional
            If True, removes the masked values from the data.
        
        Returns:
        --------
        np.ndarray
            The masked or modified data array.
        """

        if maskValue is not None:   # traget value to mask 1.334 
            Masked_data = data.copy()
            mask = data == maskValue  # mask will be the boolean array of TRue and False. 
            maskedValues_coordsOnly = np.argwhere(mask)  # coordinates of the masked values
            UnMasked_coords = np.argwhere(~mask)  # coordinates of the NON masked values
            maskedValuesExtractedbg = Masked_data*mask  # return bakground array of size of data,with maskValues as it is and other are zeros.
            # print(f"Masking values equal to {maskValue} at coordinates: {coords}")
            if masked_WithZero:
                Masked_data[mask] = 0  # set the masked values to zero
                filtered_Data_WithoutZero = Masked_data[Masked_data > 0] 
            elif masked_ANDRemoved:
                Masked_data = Masked_data[~mask]

            else:
                raise ValueError("Please specify either masked_WithZero or masked_ANDRemoved as True.") 

            # print(f" Masked data (returns the array of data size) where masked values is - {maskValue} and made them all zeros and left other as it is, or just removed them if masked_ANDRemoved = True.:{Masked_data} \n ")
            # print(f" Masked values coordinates only where masked value - {maskValue} :\n  {maskedValues_coordsOnly} \n ")
            # print(f" filtered data (return a list of nonzero values from the data) after masking with zero(making them 0) for given mask value - {maskValue} and removing those zeros. :\n {filtered_Data_WithoutZero} \n ")
            # print(f" coordinates of the unmasked values only:\n, it returns list of list (nx3 dimension), n is unmasked values.  {UnMasked_coords} \n ") 
            # print(f" mask will be the boolean array of size of data exactly with TRUE and False \n {mask} \n ") 
            # print(f"Masked values extracted background only, it is list of msked values, size of list is equal to the number  of maskValue in the data. :\n {maskedValuesExtractedbg} \n ")

        return Masked_data, maskedValues_coordsOnly, filtered_Data_WithoutZero, UnMasked_coords, mask , maskedValuesExtractedbg


    
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


# ---------------------------------DECORATOR DEFINED under the @staticmethos -------------------------

# @staticmethod
# def logfunction(func):
#     @wraps(func)
#     def wrapper(self, *args, **kwargs):
#         print(f"\n ---------------------> /// Implementing method: {func.__name__} \\\ <------------------------------------------------------- \n")
#         results = func(self, *args, **kwargs)
#         print(f"\n ---------------------> /// Finished executing method: {func.__name__} \\\ <--------------------------------------------------\n")
#         return results
#     return wrapper



########## --Binwidth explorer and finding the peaks automatically from the data, this can be generalize ##########

class BinWidthExplorer(DataPreprocessor):
    def __init__(self, data, metadata=None):
        super().__init__(data, metadata)
        self.data = self.data.flatten() if self.data.ndim == 3 else self.data

    def evaluate_binwidth_range(self, decimal_place, bw_listLength=None,  plot=False, save_dir=None, target_peak=None, tolerance=None):
        min_bw = 10 ** -(decimal_place + 1)
        max_bw = 10 ** -(decimal_place)
        if bw_listLength is None:
            bw_list = np.linspace(min_bw, max_bw, 10)
        else:
            bw_list = np.linspace(min_bw, max_bw, bw_listLength)

        # bw_list = range(min_bw, max_bw, step=10**-(decimal_place+2))  # Adjust step size as needed

        best_match = {
            'binwidth': None,
            'peak_count': -1,
            'peak_value': None,
            'peak_bin_edges': None,
            'hist': None,
            'edges': None
        }
        countIteration = 0
        for i, bw in enumerate(bw_list):
            countIteration += 1
            print(f"count iteration: {countIteration} with bw: {bw:.1e}")
            # Calculate the number of bins based on the current binwidth
            data_min, data_max = self.data.min(), self.data.max()
            num_bins = int((data_max - data_min) / bw) + 1

            counts, edges = np.histogram(self.data, bins=num_bins)
            # centers(represents the approximate values) of each bin size equal to the counts.
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            peak_idx = np.argmax(counts)
            peak_val = bin_centers[peak_idx]

            # This given if statement with target_peak: ensures that only those binwidths are considered whose resulting peak is close to your target value, defined by:
            # target_peak: manually provided known peak (most frequent value observed in your analysis).
            # bw: binwidth used at that iteration from the --> bw_list. % bw_list = np.linspace(min_bw, max_bw, 20)

            if tolerance is not None and target_peak is not None:
                # if we manually want to manage the iteration with the help of tolerance values.the use this by assigning some value to tolerance..
                print(f" picking the tolerance ----> ")
                if np.abs(target_peak - peak_val) < tolerance:
                    continue
            if target_peak is not None and tolerance is None:
                # if we want to use the target_peak value only, then use this.
                print(
                    f" Target_Peak is : {target_peak} gradually reaching to traget with bw step:{bw} and corresponding paek_val:{peak_val} ---> ")
                if target_peak is not None and not (target_peak - bw <= peak_val <= target_peak + bw):
                    continue

            if counts[peak_idx] > best_match['peak_count']:
                best_match.update({
                    'binwidth': bw,
                    'peak_count': counts[peak_idx],
                    'peak_value': peak_val,
                    'peak_bin_edges': (edges[peak_idx], edges[peak_idx+1]),
                    'hist': counts,
                    'edges': edges
                })

            if plot:
                plt.figure(figsize=(6, 3))
                plt.hist(self.data, bins=num_bins, alpha=0.6,
                        color='skyblue', edgecolor='black', density=True)
                plt.bar(bin_centers, counts, width=bw,
                        alpha=0.6, edgecolor='black')
                plt.axvline(peak_val, color='red', linestyle='--',
                            label=f"Peak: {peak_val:.6f}")
                plt.title(f"Binwidth  = {bw:.1e} | Peak = {peak_val:.6f}")
                plt.xlabel("Value")
                plt.ylabel("Count")
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.legend()
                plt.tight_layout()
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(
                        f"{save_dir}/hist_bw_{i:02d}_{bw:.1e}.png", dpi=300)
                # plt.show()
                plt.close()

        return best_match
    @logfunction
    def fit_gaussian_to_peak(self, peak_range, binwidth, key, save_dir=None):
        """ 
        peak_range = binwidth range.i.e. --> 
        explorer = BinWidthExplorer(data, metadata=None)
        results = explorer.evaluate_binwidth_range(decimal_place, plot=False, save_dir=None, target_peak=None, tolerance=None)
        peak_range = results['peak_bin_edges']
        mu, std, peak_data = explorer.fit_gaussian_to_peak(results['peak_bin_edges'],save_dir = None)
        kde_vals, kde_x = explorer.plot_kde_comparison()

        """
        mask = (self.data >= peak_range[0]) & (self.data <= peak_range[1])
        peak_data = self.data[mask]
        mu, std = norm.fit(peak_data)
        number_of_values = len(peak_data)
        print(
            f"Number of values in peak range {peak_range}: {number_of_values}")

        x = np.linspace(peak_range[0], peak_range[1],
                        number_of_values)  # 10000
        
        p = norm.pdf(x, mu, std)
        nbins = int((peak_range[1] - peak_range[0])/binwidth) + 1
        nbins = 1000
        plt.figure(figsize=(6, 3))
        plt.hist(peak_data, bins=nbins, density=True,
                 alpha=0.6, color='skyblue', edgecolor='black')
        plt.plot(x, p, 'r--', label=f'Gaussian Fit\nμ={mu:.5f}, σ={std:.2e}')
        plt.legend()
        plt.title(f"Gaussian Fit to Peak {key} ")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', alpha=0.5)
        if save_dir:
            plt.savefig(os.path.join(
                save_dir, "gaussian_fit_peak.png"), dpi=300)
            np.save(os.path.join(save_dir, "gaussian_peak_data.npy"), peak_data)
        plt.show()
        plt.close()

        return {'mu': mu, 'std': std, 'data': peak_data}

    @logfunction
    def plot_kde_comparison(self, peak_range, binwidth, key, save_dir=None):

        mask = (self.data >= peak_range[0]) & (self.data <= peak_range[1])
        peak_data = self.data[mask]
        number_of_values = len(peak_data)
        print(
            f"Number of values in peak range {peak_range}: {number_of_values}")
        mu = np.mean(peak_data)
        std = np.std(peak_data)
        kde = gaussian_kde(peak_data)
        x = np.linspace(self.data.min(), self.data.max(),
                        1000)  # 1000 points for smoothness
        # x = np.linspace(self.data.min(), self.data.max(),
        #                 number_of_values)  # 1000 points for smoothness
        y = kde.evaluate(x)
        kde_vals = y
        kde_x = x
        nbins = int((peak_range[1] - peak_range[0])/binwidth) + 1

        plt.figure(figsize=(6, 3))
        plt.plot(x, y, label=f'kde\nμ={mu:.5f}, σ={std:.2e} ', color='green')
        plt.hist(self.data, bins=nbins, density=True,
                 alpha=0.6, color='gray', label='Histogram')
        plt.legend()
        plt.title(f"KDE vs Histogram {key}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', alpha=0.5)
        if save_dir:
            plt.savefig(os.path.join(save_dir, "kde_comparison.png"), dpi=300)
        plt.show()
        plt.close()

        return kde_vals, kde_x

    # def fit_gmm_and_save(self, n_components=2, save_dir=None):
    #     data_reshaped = self.data.reshape(-1, 1)
    #     gmm = GaussianMixture(n_components=n_components, random_state=0).fit(data_reshaped)
    #     labels = gmm.predict(data_reshaped)
    #     for i in range(n_components):
    #         cluster_data = self.data[labels == i]
    #         np.save(os.path.join(save_dir, f"gmm_cluster_{i}_data.npy"), cluster_data)

    #     return gmm, labels
    @logfunction
    def fit_gmm_and_save(self, peak_range, key, n_components=2, ForAllData=None, save_dir=None, save_mat=False, plot=False):
        os.makedirs(save_dir, exist_ok=True)

        if ForAllData is not None:
            data_reshaped = self.data.reshape(-1, 1)
        else:
            mask = (self.data >= peak_range[0]) & (self.data <= peak_range[1])
            peak_data = self.data[mask]
            data_reshaped = peak_data.reshape(-1,1)
            number_of_values = len(peak_data)

        gmm = GaussianMixture(n_components=n_components,
                              random_state=0).fit(data_reshaped)
        labels = gmm.predict(data_reshaped)
        volume_shape = self.data.shape

        all_coords = np.array(np.unravel_index(
            np.arange(data_reshaped.shape[0]), volume_shape)).T
        cluster_matrix = []

        for i in range(n_components):
            cluster_mask = labels == i
            cluster_data = data_reshaped[cluster_mask].flatten()
            cluster_coords = all_coords[cluster_mask]

            np.save(os.path.join(
                save_dir, f"gmm_cluster_{i}_data.npy"), cluster_data)
            np.save(os.path.join(
                save_dir, f"gmm_cluster_{i}_coords.npy"), cluster_coords)

            labeled_coords = np.hstack(
                (cluster_coords, np.full((cluster_coords.shape[0], 1), i)))
            cluster_matrix.append(labeled_coords)

        if save_mat:
            all_labeled_data = np.vstack(cluster_matrix)
            mat_save_path = os.path.join(save_dir, 'cluster_coords_labels.mat')
            savemat(mat_save_path, {'cluster_coords_labels': all_labeled_data})
            print(f" Saved .mat file for all clusters: {mat_save_path}")

        if plot:
            x_vals = np.linspace(data_reshaped.min(),
                                 data_reshaped.max(), 1000).reshape(-1, 1)
            y_vals = np.exp(gmm.score_samples(x_vals))

            plt.figure(figsize=(8, 5))
            plt.hist(data_reshaped, bins=300, density=True,
                     alpha=0.3, color='gray', label='Data Histogram')

            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_).flatten()
            weights = gmm.weights_

            plt.figure(figsize=(6, 3))
            for i in range(n_components):
                component = weights[i] * (1 / (stds[i] * np.sqrt(2 * np.pi))) * np.exp(
                    -0.5 * ((x_vals.flatten() - means[i]) / stds[i]) ** 2)
                plt.plot(x_vals, component, label=f'Component {i}', alpha=0.7)

            plt.plot(x_vals, y_vals, 'k--', label='Total GMM Fit')
            plt.title(f'Gaussian Mixture Model Fit {key}')
            plt.xlabel('Data Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)

            plot_path = os.path.join(save_dir, 'gmm_fit_plot.png')
            plt.savefig(plot_path)
            print(f"GMM plot saved to: {plot_path}")
            plt.show()
            plt.close()

        print(f"GMM clusters saved in: {save_dir}")
        self.results['gmm'] = {
            'gmm_model': gmm,
            'labels': labels,
            'components': n_components
        }

        return gmm, labels

    # usage:
    # data = np.load("your_data.npy")
    # explorer = BinWidthExplorer(data, save_dir="results")
    # result = explorer.explore_binwidth_and_detect_peak(decimal_place=3)
    # mu, std, peak_data = explorer.fit_gaussian_to_peak(result["peak_edges"])
    # kde_vals, kde_x = explorer.plot_kde_comparison()
    # gmm, labels, sil_score, db_score = explorer.fit_gmm_and_save(n_components=2)
    # best_method, scores = explorer.compare_methods(peak_data, kde_vals, kde_x, gmm, labels, auto_select=True)

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
                raise ValueError(
                    "Data appears to be 3D coordinates with no scalar values provided.")
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

        files = [f for f in os.listdir(path) if f.endswith(
            '.mat') or f.endswith('.npy')]
        print(f"Number of data files found: {len(files)}")

        for count, filename in enumerate(files, 1):
            print(f"Processing file {count}: {filename}")
            filepath = path / filename

            if filename.endswith('.mat'):
                datastruct = sio.loadmat(filepath)
                choose_keyVal = int(
                    input(f"File {filename}: Enter 1 for 'elem_double' or 0 for 'mat_temp': "))
                data = datastruct['elem_double'] if choose_keyVal == 1 else datastruct['mat_temp']
            else:
                data = np.load(filepath)

            data = np.array(data).flatten()
            data = data[data != 0]

            clean_filename = filename.replace('.mat', '').replace('.npy', '')

            # Plot histogram with envelope
            plt.figure(figsize=(6, 5))
            counts, edges, _ = plt.hist(
                data, bins=bins, color='white', alpha=0.7, edgecolor='white')
            plt.fill_between(edges[:-1], counts,
                             color='None', edgecolor='blue', alpha=0.7)
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
            plt.hist(data, bins=bins, density=True,
                     alpha=0.6, color='g', edgecolor='black')
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
            default_colors = ['red', 'blue', 'green', 'purple',
                              'orange', 'cyan', 'brown', 'magenta', 'olive']
            return default_colors[:n] if n <= len(default_colors) else [default_colors[i % len(default_colors)] for i in range(n)]

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
            plt.hist(data, bins='fd', alpha=0.6, label='Original Data',
                     color='gray', edgecolor='black')
            for q in qvalues:
                plt.hist(q["filtered_data"], bins='fd', alpha=0.3,
                         color=q["color"], label=f'> Q_{q["quantile"]:.2f}')
                plt.axvline(q["threshold"], linestyle='--', color=q["color"], linewidth=1.5,
                            label=f'Q_{q["quantile"]:.2f} = {q["threshold"]:.2f}')
            plt.title("Histogram with Quantile Thresholds")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_dir:
                plt.savefig(
                    save_dir / "quantile_overlaid_histogram.png", dpi=300)
            plt.show()
        else:
            n = len(qvalues) + 1
            cols = 2
            rows = (n + 1) // cols
            fig, axs = plt.subplots(rows, cols, figsize=(12, rows * 4))
            axs = axs.flatten()

            axs[0].hist(data, bins='fd', alpha=0.6, color='gray',
                        label='Original Data', edgecolor='black')
            for q in qvalues:
                axs[0].hist(q["filtered_data"], bins='fd', alpha=0.3,
                            color=q["color"], label=f'> Q_{q["quantile"]:.2f}')
                axs[0].axvline(q["threshold"], linestyle='--', color=q["color"], linewidth=1.5,
                               label=f'Q_{q["quantile"]:.2f} = {q["threshold"]:.2f}')
            axs[0].legend()
            axs[0].grid(True)
            axs[0].set_title("Histogram with Quantile Overlays")

            for idx, q in enumerate(qvalues, 1):
                axs[idx].hist(q["filtered_data"], bins='fd',
                              alpha=0.6, color=q["color"], edgecolor='black')
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

    


    @staticmethod
    def WinwidthExplorer_PicbwFindPeak(data, d=3, plothist=False, TargetPeak=None):
        """ 
        TargetPeak: this value is extracted by the significant analysis.
        From my significant decimal digit analysis, i already found the most occured values with particular decimal digits(d: i.e. 3)
        I will use this (d) information to group all such value in a group (i.e. bin) by using the b.w. range 10^-(d+1) =< b.w.<10^-d .
        then I will find peak in that bin. 
        then --> which bw value is giving me the best peak -> that means close to the known my mode/median values from data. 
        here just to test the histogram with the winwidth variation 
        """
        if data.ndim >= 2:
            data = data.flatten()
        # else:
        #   # data.ndim == 1:
        #     data = data
        # binwidth = (max(data) - min(data)) / nbins

        binwidth = np.linspace(10**-(d+1), 10**(-d), 10**-(d+2))
        nbins = (np.max(data) - np.min(data)) / binwidth
        nbins = np.ceil(np.abs(nbins))

        if TargetPeak is not None:
            def Best_bw_explorer():
                counts, bin_edges = np.histogram(data, nbis=nbins)
            # counts: array of how many data points fall in each bin
            # bin_edges: the boundaries of the bins

                PeakArgument = np.argwhere(max(counts))
                peakValue = bin_edges[PeakArgument] + bin_edges[PeakArgument + 1]/2

        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=nbins)
        # plt.show()
        plt.title(f'hist for nbins:{nbins},{binwidth}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.ylim([0, 10000])
        plt.tight_layout()

        return nbins
