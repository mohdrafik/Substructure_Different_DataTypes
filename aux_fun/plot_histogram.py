import matplotlib as plt
import seaborn as sns
def plot_normalizedata_hist(normalized_data,datakey):
    data = normalized_data
    # sample_file = list(data .keys())[keynumber]
    sample_data = data.flatten()
    plt.figure(figsize=(6, 4))
    sns.histplot(sample_data, bins=50, kde=True)
    plt.title(f"Distribution of Normalized Data: {datakey}")
    plt.xlabel("Normalized Value")
    plt.ylabel("Frequency")
    plt.show()
