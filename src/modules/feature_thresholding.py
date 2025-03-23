import os
import numpy as np
import matplotlib.pyplot as plt
from listspecificfiles import readlistFiles

class FeatureQuantileThresholding:
    """
    Now that your 3D datasets are normalized, we will proceed with Feature Extraction & Quantile-Based Thresholding
    to identify meaningful substructures.

    Why This Step is Important?
    - Feature Extraction helps in understanding the distribution of voxel intensities.
    - Quantile-Based Thresholding helps to filter noise and identify significant regions in the dataset.

    What I Will Do?
    Step 1: Extract statistical features (mean, variance, quantiles)
    Step 2: Apply Quantile-Based Thresholding (0.95, 0.99 quantiles)
    Step 3: Visualize the thresholded regions in 3D slices
    """

    def __init__(self, data_dir, save_dir=None, BASE_DIR=None, keyword='normalized.npy'):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.keyword = keyword
        # self.keyword = keyword.lower()  # Normalize keyword to lowercase

        # Use readlistFiles class to get matched files
        self.dataset_names = readlistFiles(self.data_dir, self.keyword).matched_Files
        self.features = {}
        self.save_res_plot = BASE_DIR

    def load_normalized_data(self, filename):
        return np.load(os.path.join(self.data_dir, filename))

    def extract_features(self, data):
        mean_val = np.mean(data)
        std_dev = np.std(data)
        q95 = np.quantile(data, 0.95)
        q99 = np.quantile(data, 0.99)
        return mean_val, std_dev, q95, q99

    def visualize_thresholds(self, dataset, data, q95, q99, save_plot=False):
        mask_q95 = data > q95
        mask_q99 = data > q99
        slice_index = data.shape[2] // 2

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(data[:, :, slice_index], cmap="gray")
        plt.title("Original Slice")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_q95[:, :, slice_index], cmap="hot")
        plt.title("Thresholded @ Q95")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(mask_q99[:, :, slice_index], cmap="hot")
        plt.title("Thresholded @ Q99")
        plt.axis("off")

        plt.suptitle(dataset)
        plt.tight_layout()

        if save_plot and self.save_res_plot:
            os.makedirs(self.save_res_plot, exist_ok=True)
            png_path = os.path.join(self.save_res_plot, f"{dataset}_thresh.png")
            plt.savefig(png_path, dpi=300)
            print(f"🖼️ Saved PNG: {png_path}")

        plt.show()

    def process(self, visualize=True, save_features=True):
        for dataset in self.dataset_names:
            data = self.load_normalized_data(dataset)
            mean_val, std_dev, q95, q99 = self.extract_features(data)
            self.features[dataset] = {
                "Mean": mean_val,
                "Std Dev": std_dev,
                "Q95": q95,
                "Q99": q99
            }
            print(f"{dataset}: Mean={mean_val:.4f}, Std Dev={std_dev:.4f}, Q95={q95:.4f}, Q99={q99:.4f}")

            if visualize:
                self.visualize_thresholds(dataset, data, q95, q99, save_plot=True)

        if save_features and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, "ThresQuntFeature.txt")
            with open(save_path, 'w') as f:
                f.write(str(self.features))
            print(f"✅ Features saved to: {save_path}")


if __name__ =="__main__":
    from pathlib import Path  # type: ignore

    from feature_thresholding import FeatureQuantileThresholding  # type: ignore
    from listspecificfiles import readlistFiles  # type: ignore

    data_dir = r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\data\normalized_npyData"
    save_dir = r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\results\featureQuantileThres"
    cwd = Path.cwd().parent.parent
    BASE_DIR = cwd/ "results"
    print(f"----------< {BASE_DIR}")
    fq = FeatureQuantileThresholding(data_dir, save_dir, BASE_DIR=BASE_DIR)
    fq.process(visualize=True, save_features=True)

