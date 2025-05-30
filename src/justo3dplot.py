import numpy as np
import open3d as o3d
import os

# Load saved cluster labels and coordinates
# labels = np.load("clustering_output/cluster_labels.npy")
# coords = np.load("clustering_output/voxel_coords.npy")

labels = np.load(r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\src\clustering_output\cluster_labels.npy")
coords = np.load(r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\src\clustering_output\voxel_coords.npy")

# Get unique cluster labels (excluding noise label -1)
unique_labels = np.unique(labels)
unique_labels = unique_labels[unique_labels != -1]

# Directory to optionally save .ply files
# os.makedirs("o3d_clusters", exist_ok=True)

for label in unique_labels:
    cluster_coords = coords[labels == label]
    print(f"I am inside the {label}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster_coords)

    # Optional: assign color per cluster
    color = np.random.rand(3)  # random RGB
    pcd.paint_uniform_color(color)

    # Visualize
    print(f"Showing cluster {label} with {len(cluster_coords)} points...")
    o3d.visualization.draw_geometries([pcd])

    # Save if you want
    # o3d.io.write_point_cloud(f"o3d_clusters/cluster_{label}.ply", pcd)
