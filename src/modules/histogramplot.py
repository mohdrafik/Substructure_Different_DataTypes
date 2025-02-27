import matplotlib.pyplot as plt  #
import seaborn as sns

def plot_normalizedata_hist(normalized_data, datakey):
    """Plot histogram of normalized data."""
    
    sample_data = normalized_data.flatten()  # Ensure data is in 1D
    plt.figure(figsize=(6, 4))
    sns.histplot(sample_data, bins=50, kde=True)
    plt.title(f"Distribution of Normalized Data: {datakey}")
    plt.xlabel("Normalized Value")
    plt.ylabel("Frequency")
    plt.show()

if __name__=="__main__":
    import random
    import numpy as np

    nmpyarraydata = np.random.randint(1,4,(10,100,100))
    datakeys = "rafik"
    plot_normalizedata_hist(nmpyarraydata,datakeys)