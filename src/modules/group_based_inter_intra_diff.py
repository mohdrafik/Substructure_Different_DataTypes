import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.signal import savgol_filter

from pathlib import Path

sys.path.append(
    r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\src"
)

from path_manager import addpath
addpath()
from listspecificfiles import readlistFiles

class GroupingBasedDistance:
    def __init__(self, rel_datapath, data_path=None, output=None, file_suffix=None):
        self.data_path = data_path
        self.rel_datapath = rel_datapath
        self.output = output if output is not None else Path.cwd() / "results"
        self.file_suffix = file_suffix if file_suffix is not None else ".npy"

        self.fpath = readlistFiles(self.rel_datapath, self.file_suffix).file_with_Path()

    def load_file(self, file_withpath):
        data = np.load(file_withpath)
        return data

    def find_diff(self, data):
        data = data[data > 0]
        data = data.flatten()
        data_val_range = max(data) - min(data)
        sorted_data = np.sort(data)
        # first difference between va;ues of teh sotrted data array
        df1 = np.diff(sorted_data)
        print(f"max of 1st diff array : (<{data_val_range}): {max(df1):.5f}")
        print(f"min of 1st diff array : ({data_val_range}): {min(df1):.8e}")
        return df1

    def plot_sharp_gradient_changes(self, arr, gradient_threshold=0.01):
        
        """
        Compute the gradient of the input array, detect sharp rises,
        and plot them along with the original data.
        Parameters:
        -----------
        arr : array-like
            1D numeric array (list or numpy array)
        gradient_threshold : float
            The threshold above which gradient changes are considered sharp.
        """
        arr = np.asarray(arr)
        grad = np.gradient(arr)

        # Identify indices where gradient exceeds threshold
        sharp_indices = np.where(np.abs(grad) > gradient_threshold)[0]

        # # Plot original data
        # plt.figure(figsize=(8, 4))
        # plt.plot(arr, label="Original Data", linewidth=1.5)
        # plt.scatter(sharp_indices, arr[sharp_indices], color='red', label="Sharp Rises", zorder=5)
        # plt.title("Sharp gradient changes")
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        # plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()
        # plt.show()

        # Also plot gradient itself
        plt.figure(figsize=(8, 3))
        plt.plot(grad, label="Gradient", color="purple")
        plt.axhline(
            gradient_threshold,
            color="gray",
            linestyle="--",
            label=f"Threshold (+{gradient_threshold})",
        )
        plt.axhline(
            -gradient_threshold,
            color="gray",
            linestyle="--",
            label=f"Threshold (-{gradient_threshold})",
        )
        plt.scatter(
            sharp_indices,
            grad[sharp_indices],
            color="red",
            label="Sharp Points",
            zorder=5,
        )
        plt.title("Gradient Values")
        plt.xlabel("Index")
        plt.ylabel("Gradient")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def save_plot(self, plot_save_path=None, Title=None, filename=None, pdfsave=None):

        """
        To save the plot produce for each file:
        args:
        plot_save_path : str -> only , it will work as child after  results/ 'plot_save_path'
        Title of each plot: filename[:-12] after removing the mask...
        filename : filename[:-12] -> To save each plot with this individual name.
        pdfsave : True | None -> save in .pdf = True else it will save .png

        """

        # result_dir = Path.cwd()
        result_dir = (
            self.output / plot_save_path
            if plot_save_path is not None
            else self.output / "group_sort_data"
        )

        print(f"parents: {result_dir}")
        result_dir.mkdir(parents=True, exist_ok=True)

        if pdfsave:
            filename = (
                filename
                if filename is not None
                else "sortdata_all_oneExceptgrafene.pdf"
            )
            # file_fig_name = os.path.join(result_dir,filename)
        else:
            filename = (
                filename
                if filename is not None
                else "sortdata_all_oneExceptgrafene.png"
            )

        file_fig_name = os.path.join(result_dir, filename)

        plt.savefig(file_fig_name, dpi=600)
        print(f"file is saved at : {file_fig_name}\n")


    # @staticmethod
    def test_sliding_window_variance(self,data=None, sliding_window_size =100, fname=None, plot_save_path=None):
    # In practice, replace this with your actual flattened, sorted data array
        # data = np.concatenate([
        #     np.random.normal(loc=10, scale=1, size=300),    # cluster around 10
        #     np.random.normal(loc=50, scale=2, size=300),    # cluster around 50
        #     np.random.normal(loc=100, scale=2, size=300)    # cluster around 100
        # ])

        # data = data[data > 0]        # remove zeros or invalid values
        # data.sort()                  # sort the data

        N = data.size
        # Choose a sliding window size (e.g., 50 in this example)
        w = 50 if sliding_window_size is None else sliding_window_size

        # Compute cumulative sums for data and data^2
        cumsum = np.concatenate([[0], np.cumsum(data, dtype=np.float64)])
        cumsum_sq = np.concatenate([[0], np.cumsum(data**2, dtype=np.float64)])

        # Compute local variance for each window position
        window_variances = np.empty(N - w + 1, dtype=np.float64)
        for start in range(0, N - w + 1):
            end = start + w
            # Sum and sum of squares in window [start, end)
            window_sum   = cumsum[end]   - cumsum[start]
            window_sum_sq= cumsum_sq[end] - cumsum_sq[start]
            μ = window_sum / w
            var = (window_sum_sq / w) - μ**2     # variance formula
            window_variances[start] = var

        # Determine a variance threshold (e.g., mean + 1*std in this case)
        thr = window_variances.mean() + window_variances.std()
        high_var_indices = np.where(window_variances > thr)[0]

        print(f"Chosen window size: {w}")
        print(f"Variance threshold: {thr:.3f}")
        print(f"Number of high-variance window positions: {high_var_indices.size}")
        print(f"Sample of high-variance indices: {high_var_indices[:10]}")

        # Plot the sorted data and local variance profile
        fig, ax1 = plt.subplots(figsize=(8,4))
        ax1.set_xlabel('Index in sorted data')
        ax1.set_ylabel('Data value', color='blue')
        ax1.plot(data, color='blue', label='Sorted data values')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel(f'Local variance (window={w})', color='red')
        ax2.plot(np.arange(w//2, w//2 + len(window_variances)), window_variances, 
                color='red', label='Local variance')
        ax2.axhline(y=thr, color='gray', linestyle='--', label='Threshold')
        ax2.tick_params(axis='y', labelcolor='red')

        # plt.title('Sliding Window Variance Profile')
        plt.title(f"{fname[:-12]}_swv{w}")
        fig.tight_layout()
        plt.legend(loc='upper right')
       
        self.save_plot(plot_save_path=None, Title=f"{fname[:-12]}_swv", filename=f"{fname[:-12]}sw_var{w}.png", pdfsave=False)
        
        plt.pause(3)
        plt.close()

    def grouping_by_slopes(self, data=None, tolerance = None):

        """" 
         np.diff(y) / np.diff(x) computes discrete gradient.

        Then we iterate through each slope, comparing it to the previous one. If it's within ± tolerance, we keep it in the same group. If not, we start a new group.

        At the end, groups is a list of lists of indices, each representing a segment with approximately constant slope.

        """

        x = range(1,len(data)+1,1)  # Assuming data is a 1D array
        x  = np.array(x)  # Ensure x is a numpy array for consistency
        y = data
        gradient = np.diff(y) / np.diff(x)

        # Group by similar slope within tolerance

        tolerance = 0.001 if tolerance is None else tolerance

        groups = []
        current_group = [0]
        for i in range(1, len(gradient)):
            if abs(gradient[i] - gradient[i-1]) <= tolerance:
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
        groups.append(current_group)

        # # Print group details
        # for i, grp in enumerate(groups):
        #     print(f"Group {i}: gradient indices {grp}, approx slope: {np.mean(gradient[grp]):.3f}")

        # Plot
        plt.figure(figsize=(8,4))
        plt.plot(x, y, marker='o')
        colors = ['r', 'g', 'b', 'm', 'c', 'y']
        for i, grp in enumerate(groups):
            indices = np.array(grp)
            plt.plot(x[indices], y[indices], color=colors[i % len(colors)], linewidth=0.1, label=f'Group {i}')
        plt.legend()
        plt.title("Grouping by approximate gradient")
        plt.show()

    # it is not possible to have the inflexiopn point sin the sorted increasing data because data is always increasing or flat some time.
    # so it is not possible to have the inflexion points in the sorted data.

    def find_inflexion_points(self, data, threshold=None):
        """
        Args:
        - data : 1 D  numpy array or list of numeric values.
        Find inflection points in the data where the gradient exceeds a threshold.
        
        Parameters:
        - data: 1D numpy array or list of numeric values.
        - threshold: float, the minimum change in gradient to consider as an inflection point.
        
        Returns:
        - inflection_points: list of indices where inflection points occur.
        """
        threshold = 0.0001 if threshold is None else threshold

        gradients = np.gradient(data)  # first derivative


        double_gradients = np.gradient(gradients)  # second derivative
       
        # print(
        #     f"Gradients: {gradients} and length: {len(gradients)}"
        #     f"\nDouble Gradients: {double_gradients} and length: {len(double_gradients)}"
        #     f"\nData: {data} and length: {len(data)} "    
        #     )
        
        Signs = np.sign(double_gradients)  # signum of the second derivative: signum function to convert  + -> +1, -ve -> -1 and 0 -> 0 
        print(f"Signs of second derivative: {Signs} and length: {len(Signs)}") 








        double_gradients_zeros_idx = np.where(Signs == 0)[0]  # list of indices where the second derivative is zero
        print(f"Indices where second derivative is zero: {double_gradients_zeros_idx}")
        # print(f"Signs of second derivative: {Signs[idx_extraction+1] for idx_extraction in double_gradients_zeros_idx}")
        for idx_extraction in double_gradients_zeros_idx:
            if idx_extraction == 0 or idx_extraction == len(data) - 1:
                continue
            print(
                f"\n before : {idx_extraction-1}: {Signs[idx_extraction-1]}"
                f"\n Signs of second derivative at index {idx_extraction}: {Signs[idx_extraction]}"
                f"\n after : {idx_extraction+1}: {Signs[idx_extraction+1]}"
                f"\n and gradient: idxbefore: {idx_extraction-1}, Value gradient: {gradients[idx_extraction-1]:.4} to idx_after :{idx_extraction+1},and Value gradient: {gradients[idx_extraction+1]:.4e}"
                f"\n and double gradient: {double_gradients[idx_extraction-1]:.4} to {double_gradients[idx_extraction+1]:.4e}"
            )

        # Filter out indices where the second derivative is zero
        inflection_point_idx = []  # list to store inflection point indices

        if len(double_gradients_zeros_idx) > 0:
                
            for inflexion_idx in double_gradients_zeros_idx:
                # print(f"Inflection point index: {inflexion_idx}, value: {data[inflexion_idx]}")
                # Check if the inflection point is at the start or end of the data
                # if inflexion_idx == 0 or inflexion_idx == len(data) - 1:
                #     continue            
                if Signs[inflexion_idx - 1] and Signs[inflexion_idx + 1] == -1:  # check if the sign changes from + to - or - to + around the zero crossing
                    print(f"Inflection point found at index {inflexion_idx} with value {Signs[inflexion_idx]} , signumidx -1 :{Signs[inflexion_idx-1]}  signumidx+1 :{Signs[inflexion_idx+1]}  \n and gradient {gradients[inflexion_idx-1]} to {gradients[inflexion_idx+1]}")
                    inflection_point_idx.append(inflexion_idx)

        # inflection_points = []

        # else:
            
        # inflection_point_idx = np.where(np.diff(Signs) != 0)[0] + 1

        

        return inflection_point_idx, gradients, double_gradients, Signs
    

    def segment_data_based_inflexion_points(self,data,filename = None):

        """
        Segment data based on inflection points where the gradient exceeds a threshold.
        
        Parameters:
        - data: 1D numpy array or list of numeric values.
        - threshold: float, the minimum change in gradient to consider as an inflection point.
        
        Returns:
        - segments: list of tuples, each containing the start and end indices of segments.
        """
        index_inflection_points, gradients, double_gradients, Signs = self.find_inflexion_points(data)

        segments = []
        start_idx = 0

        for idx in index_inflection_points:
            if start_idx < idx:  # Ensure we don't create empty segments
                segments.append(data[start_idx:idx])
                start_idx = idx

        segments.append(data[start_idx:])  # Add the last segment from the last inflection point to the end of the data

        
        # plot all segments
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Data')
        plt.plot(gradients, label='Approx. slope', alpha=0.5)
        plt.plot(double_gradients, label='Second derivative', alpha=0.5)
        plt.scatter(index_inflection_points, data[index_inflection_points], color='black', label='Inflection points', zorder=5)
        plt.legend()
        plt.title("Data with Inflection Points")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        # Print summary
        print(f"Found {len(index_inflection_points)} inflection points, divided into {len(segments)} segments.")

        # plt.figure(figsize=(10, 6))
        # for i, segment in enumerate(segments):     
        #     plt.plot(segment, label=f'Segment {i+1}', linewidth=1.5)

        # plt.title("Data Segmentation Based on Inflexion Points")    
        # plt.xlabel("Index")
        # plt.ylabel("Value")     
        # plt.legend()
        # plt.grid(True)  
        # plt.tight_layout()
        self.save_plot(plot_save_path=None, Title="Segmented Data", filename=f"{filename[:-12]}segmented_data.png", pdfsave=False)
        # plt.show()      

        return segments
    
    # this function is not working properly because the data is sorted and it is not possible to have local maxima in the sorted data. it's  taking too long.
    def find_local_maxima_basedon_variance(self, data, window_size= None):
        
        # Example sorted data (could be your dataset)

        # data = np.sort(np.random.normal(0, 1, 50))  # Replace with your sorted data
        data = np.asarray(data)  # Ensure data is a numpy array
        # Store results
        max_variances = []
        indices = []
        window_sizes = []

        # Iterate over increasing window sizes
        for w in range(2, len(data)//2):  # you can decide max window size
            variances = []
            for i in range(len(data) - w + 1):
                window = data[i:i+w]
                var = np.var(window)
                variances.append(var)
            variances = np.array(variances)

            # Find local maxima in the variance array
            # (compare with neighbors)
            for j in range(1, len(variances)-1):
                if variances[j] > variances[j-1] and variances[j] > variances[j+1]:
                    max_variances.append(variances[j])
                    indices.append((j, w))  # start index + window size
                    window_sizes.append(w)

        # Print results
        for var, (idx, w) in zip(max_variances, indices):
            print(f"Local max variance: {var:.4f} in window starting at index {idx} (size {w}), values: {data[idx:idx+w]}")

## <-----------  for smoothing the data, specially for finding the inflexion points, we can use Savitzky-Golay filter. ----------->
    def savgol_smoothening(self, data = None, window_length=None, polyorder=None):
        """
        Apply Savitzky-Golay filter to smooth the data.
        
        Parameters:
        - data: 1D numpy array or list of numeric values.
        - window_length: int, the length of the filter window (must be odd).
        - polyorder: int, the order of the polynomial used to fit the samples.
        
        Returns:
        - smoothed_data: 1D numpy array of smoothed values.
        """

        

        window_length = 5 if window_length is None else window_length
        polyorder = 4 if polyorder is None else polyorder

        smoothed_data = savgol_filter(data, window_length, polyorder)
        
        return smoothed_data





if __name__ == "__main__":

    # from pathlib import Path
    # from group_based_inter_intra_diff import GroupingBasedDistance
    # import os

    relative_datapath = r"data\processed\main_fgdata"
    gbd = GroupingBasedDistance(rel_datapath=relative_datapath)

    # plt.figure(figsize=(12,10))  # make figure only once  # plotting figure
    plt.figure(figsize=(11.7, 8.3))

    for idx, fwpath in enumerate(gbd.fpath):
        filewpath = Path(fwpath)
        filename = filewpath.name
        fname = filewpath.stem

        data = gbd.load_file(file_withpath=filewpath)
        print(f"\n I am processing with filename: {filename}")
        df1 = gbd.find_diff(data)

        data_1d = data[data > 0].flatten()
        data_1d = np.sort(data_1d)
        # gbd.plot_sharp_gradient_changes(data_1d,gradient_threshold=0.0001)

        if fname == "tomo_Grafene_24h_masked_data":
            continue
            # plt.show()

        # plt.subplot(4, 2, idx + 1)
        plt.plot(data_1d, label=f"{fname[:-12]}")
        # plt.plot(df1, label='Gradient')
        ytickvalues = np.linspace(min(data_1d), max(data_1d), 5)
        # xtickvalues = range(len(data_1d))
        # xtickvalues = f"{xtickvalues:.2e}"
        # plt.xticks(xtickvalues)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        plt.yticks(ytickvalues)
        plt.title(fname)
        plt.grid(True, "both")
        # plt.grid(True)
        plt.legend()

        gbd.save_plot(plot_save_path=None, Title=fname[:-12], filename=fname[:-12])

        # plt.show(block=False)
        plt.pause(3)  # shows the figure non-blocking for 4 seconds
        plt.close()   # closes it

        # gbd.test_sliding_window_variance(data=data_1d)
        
    # plt.tight_layout()  # adjust subplots to fit into figure area.
    # when enable these below 4 lines. then enable also the: --> plt.subplot(4, 2, idx + 1)
    # plt.tight_layout()
    # gbd.save_plot(plot_save_path=None,
    #               Title=fname[:-12], filename=f"sortdata_all_oneExceptgrafene_final")
    # plt.show()
