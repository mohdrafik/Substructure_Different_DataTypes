# import matplotlib.pyplot as plt  #
# import seaborn as sns
# Re-import necessary libraries after code execution state reset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from scipy.stats import gaussian_kde, norm
from scipy.signal import find_peaks


def plot_normalizedata_hist(normalized_data, datakey, output_path=None):

    """Plot histograms and envelopes by removing the highest peak via Gaussian fitting."""
    sample_data = normalized_data.flatten()
    
    # Estimate KDE for the full data
    kde = gaussian_kde(sample_data)
    x_vals = np.linspace(sample_data.min(), sample_data.max(), 1000)
    kde_vals = kde(x_vals)
    
    # Find the highest peak using KDE
    peaks, _ = find_peaks(kde_vals)
    peak_idx = peaks[np.argmax(kde_vals[peaks])]
    peak_x = x_vals[peak_idx]
    peak_y = kde_vals[peak_idx]

    # Fit Gaussian around the highest peak to estimate width
    sigma_estimate = 0.05 * (sample_data.max() - sample_data.min())  # heuristic sigma
    lower_bound = peak_x - 2 * sigma_estimate
    upper_bound = peak_x + 2 * sigma_estimate

    # Filter out the data in the peak range
    remaining_data = sample_data[(sample_data < lower_bound) | (sample_data > upper_bound)]

    # Envelope for filtered data
    kde_remaining = gaussian_kde(remaining_data)
    kde_remaining_vals = kde_remaining(x_vals)

    # Create the plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Histogram and Envelope Analysis: {datakey}", fontsize=14)

    # Full histogram with KDE
    sns.histplot(sample_data, bins=50, kde=True, ax=axs[0, 0], color='skyblue')
    axs[0, 0].set_title("Full Data Histogram with KDE")
    axs[0, 0].axvline(peak_x, color='red', linestyle='--', label=f'Peak at {peak_x:.2f}')
    axs[0, 0].legend()

    # Filtered histogram
    sns.histplot(remaining_data, bins=50, kde=True, ax=axs[0, 1], color='lightgreen')
    axs[0, 1].set_title("Histogram Without Highest Peak")
    axs[0, 1].set_xlim(sample_data.min(), sample_data.max())

    # KDE Envelope Full
    axs[1, 0].plot(x_vals, kde_vals, color='blue', label='Original KDE')
    axs[1, 0].axvline(peak_x, color='red', linestyle='--', label='Highest Peak')
    axs[1, 0].set_title("Envelope (KDE) - Full Data")
    axs[1, 0].legend()

    # KDE Envelope Filtered
    axs[1, 1].plot(x_vals, kde_remaining_vals, color='green', label='Filtered KDE')
    axs[1, 1].set_title("Envelope (KDE) - Without Peak")
    axs[1, 1].set_xlim(sample_data.min(), sample_data.max())
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.set_xlabel("Normalized Value")
        ax.set_ylabel("Density")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to: {output_path}")
    plt.show()


# Sample call
# from pathlib import Path
# sample_array = np.random.normal(loc=0.5, scale=0.1, size=(10, 100, 100))
# datakey = "rafik"
# plot_normalizedata_hist(sample_array, datakey, output_path=Path.cwd() / f"{datakey}_filtered_hist.png")


if __name__=="__main__":
    # Example usage with simulated data

    # data = np.random.randint(1, 4, (10, 100, 100))
    import os 
    from pathlib import Path
    dataPath = r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\data\normalized_npyData"

    plot_outputPtah = Path(__file__).resolve().parent.parent.parent    # Will return src/
    plot_outputPtah = plot_outputPtah/"plots"
    dataPath = os.path.normpath(dataPath)
    datalist = os.listdir(dataPath)
    print(f"normpath after : {dataPath}")

    for data in datalist: 
        
        dataValues = np.load(os.path.join(dataPath,data)) 
        print(f"data: {data} and Values:{dataValues[1,1,1]}")
        datakey = data[:-4]

        plotFilename  = f"{datakey}.png"
        plot_path = os.path.join(plot_outputPtah,plotFilename)

        plot_normalizedata_hist(dataValues, datakey=f"{datakey}", output_path = plot_path)

# <--------------------------------------------------------------------------------------------->
# def plot_normalizedata_hist(normalized_data, datakey):
#     """Plot histogram of normalized data."""
    
#     sample_data = normalized_data.flatten()  # Ensure data is in 1D
#     plt.figure(figsize=(6, 4))
#     sns.histplot(sample_data, bins=50, kde=True)
#     plt.title(f"Distribution of Normalized Data: {datakey}")
#     plt.xlabel("Normalized Value")
#     plt.ylabel("Frequency")
#     plt.show()

# if __name__=="__main__":
#     import random
#     import numpy as np

#     nmpyarraydata = np.random.randint(1,4,(10,100,100))
#     datakeys = "rafik"
#     plot_normalizedata_hist(nmpyarraydata,datakeys)