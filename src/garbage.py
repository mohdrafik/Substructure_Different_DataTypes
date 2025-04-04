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
