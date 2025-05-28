import os
from pathlib import Path
import numpy as np
import cupy as cp
from skimage.filters import threshold_otsu
import open3d as o3d


def apply_otsu_segmentation_with_cupy(npy_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npy_path)
    if data.ndim != 3:
        raise ValueError("Expected a 3D array")

    flat_data = cp.asarray(data[data > 0].flatten())
    threshold = float(threshold_otsu(cp.asnumpy(flat_data)))
    print(f"Otsu Threshold: {threshold:.3f}")

    foreground_mask = data > threshold
    background_mask = ~foreground_mask

    base_name = Path(npy_path).stem
    np.save(output_dir / f"{base_name}_fg_mask.npy", foreground_mask)
    np.save(output_dir / f"{base_name}_bg_mask.npy", background_mask)

    return threshold, foreground_mask, background_mask


def visualize_with_open3d(fg_mask, bg_mask, title="3D Visualization", downsample=True, max_points=50000):
    fg_coords = np.argwhere(fg_mask)
    bg_coords = np.argwhere(bg_mask)

    if downsample:
        if len(fg_coords) > max_points:
            fg_coords = fg_coords[np.random.choice(len(fg_coords), max_points, replace=False)]
        if len(bg_coords) > max_points:
            bg_coords = bg_coords[np.random.choice(len(bg_coords), max_points, replace=False)]

    fg_pcd = o3d.geometry.PointCloud()
    fg_pcd.points = o3d.utility.Vector3dVector(fg_coords)
    fg_colors = np.tile([0.3, 0.5, 1.0], (fg_coords.shape[0], 1))
    fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors)

    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(bg_coords)
    bg_colors = np.tile([0.6, 0.6, 0.6], (bg_coords.shape[0], 1))
    bg_pcd.colors = o3d.utility.Vector3dVector(bg_colors)

    o3d.visualization.draw_geometries([fg_pcd, bg_pcd], window_name=title)


def batch_otsu_segmentation(input_dir, output_dir, use_open3d=True, downsample=True, max_points=50000):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    npy_files = list(input_dir.glob("*.npy"))

    print(f"🔍 Found {len(npy_files)} .npy files in: {input_dir}")

    for file in npy_files:
        print(f"\n🚀 Processing: {file.name}")
        try:
            threshold, fg_mask, bg_mask = apply_otsu_segmentation_with_cupy(file, output_dir)
            if use_open3d:
                visualize_with_open3d(fg_mask, bg_mask, title=file.stem, downsample=downsample, max_points=max_points)
            print(f"✅ Finished {file.name}")
        except Exception as e:
            print(f"❌ Failed {file.name}: {e}")


if __name__ == "__main__":
    PROJECT_PATH = Path.cwd().parent
    input_npy_path = PROJECT_PATH / "data" / "raw_npyData"
    result_path = PROJECT_PATH / "results" / "otsu_gpu"

    batch_otsu_segmentation(
        input_dir=input_npy_path,
        output_dir=result_path,
        use_open3d=True,
        downsample=True,
        max_points=50000
    )

"""
Example usage of this below code in help string quotes:
--------------------------
data = np.load("your_data.npy")
explorer = BinWidthExplorer(data, save_dir="results")
result = explorer.explore_binwidth_and_detect_peak(decimal_place=3)
mu, std, peak_data = explorer.fit_gaussian_to_peak(result["peak_edges"])
kde_vals, kde_x = explorer.plot_kde_comparison()
gmm, labels, sil_score, db_score = explorer.fit_gmm_and_save(n_components=2)
best_method, scores = explorer.compare_methods(peak_data, kde_vals, kde_x, gmm, labels, auto_select=True)
--------------------------

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

class BinWidthExplorer:
    def __init__(self, data, save_dir=None):
        self.data = data.flatten() if data.ndim == 3 else data
        self.save_dir = save_dir or os.getcwd()
        os.makedirs(self.save_dir, exist_ok=True)

    def explore_binwidth_and_detect_peak(self, decimal_place=3, plot=True):
        min_bw = 10 ** -(decimal_place + 1)
        max_bw = 10 ** -(decimal_place)
        step = (max_bw - min_bw) / 20
        best_binwidth = None
        best_peak = -np.inf
        results = {}

        for bw in np.arange(min_bw, max_bw, step):
            bins = np.arange(self.data.min(), self.data.max() + bw, bw)
            counts, edges = np.histogram(self.data, bins=bins)
            peak_idx = np.argmax(counts)
            peak_val = counts[peak_idx]

            if peak_val > best_peak:
                best_peak = peak_val
                best_binwidth = bw
                best_edges = (edges[peak_idx], edges[peak_idx + 1])

            results[bw] = (counts, edges, peak_val)

        if plot:
            plt.figure(figsize=(10, 6))
            for bw, (counts, edges, _) in results.items():
                plt.plot(edges[:-1], counts, label=f"bw={bw:.5f}", alpha=0.4)
            plt.title("Binwidth Exploration")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "binwidth_exploration.png"))
            plt.close()

        return {
            "best_binwidth": best_binwidth,
            "peak_edges": best_edges,
            "peak_value": best_peak
        }

    def fit_gaussian_to_peak(self, peak_edges, save_plot=True):
        mask = (self.data >= peak_edges[0]) & (self.data <= peak_edges[1])
        peak_data = self.data[mask]
        mu, std = norm.fit(peak_data)

        x = np.linspace(min(peak_data), max(peak_data), 1000)
        pdf = norm.pdf(x, mu, std)

        if save_plot:
            plt.figure(figsize=(8, 4))
            plt.hist(peak_data, bins=50, density=True, alpha=0.6, label="Histogram")
            plt.plot(x, pdf, 'r--', label=f"Gaussian Fit\nμ={mu:.4f}, σ={std:.4e}")
            plt.title("Gaussian Fit to Peak")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "gaussian_fit_peak.png"))
            plt.close()

        np.save(os.path.join(self.save_dir, "peak_data.npy"), peak_data)
        return mu, std, peak_data

    def plot_kde_comparison(self):
        kde = gaussian_kde(self.data)
        x = np.linspace(min(self.data), max(self.data), 1000)
        kde_values = kde(x)

        plt.figure(figsize=(8, 4))
        plt.hist(self.data, bins=100, density=True, alpha=0.5, label="Histogram")
        plt.plot(x, kde_values, label="KDE", color="green")
        plt.title("KDE vs Histogram")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "kde_comparison.png"))
        plt.close()

        return kde_values, x

    def fit_gmm_and_save(self, n_components=2):
        data_reshaped = self.data.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, random_state=0).fit(data_reshaped)
        labels = gmm.predict(data_reshaped)

        for i in range(n_components):
            cluster_data = self.data[labels == i]
            np.save(os.path.join(self.save_dir, f"gmm_cluster_{i}_data.npy"), cluster_data)

        silhouette = silhouette_score(data_reshaped, labels)
        db_index = davies_bouldin_score(data_reshaped, labels)

        return gmm, labels, silhouette, db_index

    def compare_methods(self, peak_data, kde_values, kde_x, gmm, labels, auto_select=True):
        gmm_score = silhouette_score(self.data.reshape(-1, 1), labels)
        kde_peak = kde_x[np.argmax(kde_values)]
        kde_peak_count = max(kde_values)
        hist_peak_count = len(peak_data)

        scores = {
            "GMM_Silhouette": gmm_score,
            "KDE_PeakHeight": kde_peak_count,
            "Histogram_Count": hist_peak_count
        }

        if auto_select:
            selected = max(scores, key=scores.get)
        else:
            selected = None

        with open(os.path.join(self.save_dir, "method_comparison.txt"), "w") as f:
            for method, score in scores.items():
                f.write(f"{method}: {score:.4f}\n")
            if selected:
                f.write(f"\nBest Method: {selected}\n")

        return selected, scores

# Example Usage:
if __name__ == "__main__":
    data = np.load("your_data.npy")
    explorer = BinWidthExplorer(data, save_dir="results")

    result = explorer.explore_binwidth_and_detect_peak(decimal_place=3)
    mu, std, peak_data = explorer.fit_gaussian_to_peak(result["peak_edges"])
    kde_vals, kde_x = explorer.plot_kde_comparison()
    gmm, labels, sil_score, db_score = explorer.fit_gmm_and_save(n_components=2)

    best_method, score_dict = explorer.compare_methods(peak_data, kde_vals, kde_x, gmm, labels, auto_select=True)

    print(f"Best Method: {best_method}\nScores: {score_dict}")


"""



# import numpy as np

# data = np.random.randint(0,10,(1,10))  # Normal-distributed random numbers
# print(f"data is :{data}")
# # Quartiles (4-tiles)
# quartiles = np.quantile(data, [0.25, 0.5, 0.75])
# print("Quartiles:", quartiles)

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import RobustScaler

# # Generate sample data with outliers
# np.random.seed(42)
# normal_data = np.random.normal(50, 5, 1000)
# outliers = np.random.normal(100, 1, 20)
# data = np.concatenate([normal_data, outliers]).reshape(-1, 1)

# # Apply Robust Scaler
# scaler = RobustScaler()
# data_robust_scaled = scaler.fit_transform(data)

# # Plot original vs robust scaled
# # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# x1 = range(len(data))
# # First scatter plot

# ax1.scatter(x1,data, color='blue', label='Data 1')
# ax1.set_title("Scatter Plot 1")
# ax1.set_xlabel("X")
# ax1.set_ylabel("Y")
# ax1.legend()

# x2 = range(len(data_robust_scaled))
# # Second scatter plot
# ax2.scatter(x2,data_robust_scaled , color='green', label='Data 2')
# ax2.set_title("Scatter Plot 2")
# ax2.set_xlabel("X")
# ax2.set_ylabel("Y")
# ax2.legend()

# plt.tight_layout()
# plt.show()


# # sns.histplot(data.ravel(), bins=50, kde=True, ax=axs[0])
# # axs[0].set_title('Original Data with Outliers')
# # axs[0].set_xlabel('Value')

# # sns.histplot(data_robust_scaled.ravel(), bins=50, kde=True, ax=axs[1])
# # axs[1].set_title('Robust Scaled Data')
# # axs[1].set_xlabel('Scaled Value')

# plt.tight_layout()
# plt.show()
