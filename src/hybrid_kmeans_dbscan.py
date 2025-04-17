import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# THRESHOLD_VALUE =  1.334
def load_volume(filepath,THRESHOLD_VALUE):
    if filepath.endswith('.npy'):
        volume = np.load(filepath)
        volume[volume <= THRESHOLD_VALUE] = 0  # Threshold to remove background
    elif filepath.endswith('.mat'):
        mat = sio.loadmat(filepath)
        # Assuming your volume variable is named 'volume' in .mat
        volume = next(v for v in mat.values() if isinstance(v, np.ndarray) and v.ndim == 3)

        volume[volume <= THRESHOLD_VALUE] = 0  # Threshold to remove background
    else:
        raise ValueError("Unsupported file format. Use .mat or .npy")
    return volume


def extract_features(volume):
    coords = np.array(np.nonzero(volume)).T
    intensities = volume[volume > 0].flatten().reshape(-1, 1)
    return np.hstack((coords, intensities))

def run_kmeans(X_scaled, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, init= 'k-means++',random_state=42)
    return kmeans.fit_predict(X_scaled)

def run_dbscan_per_cluster(X_scaled, kmeans_labels, eps=0.6, min_samples=5):
    final_labels = -np.ones(len(X_scaled), dtype=int)
    label_offset = 0
    for cluster_id in np.unique(kmeans_labels):
        indices = np.where(kmeans_labels == cluster_id)[0]
        db = DBSCAN(eps=eps, min_samples=min_samples) # using the DBSCAN 
        sub_labels = db.fit_predict(X_scaled[indices])
        sub_labels[sub_labels != -1] += label_offset
        final_labels[indices] = sub_labels
        label_offset += sub_labels.max() + 1 if sub_labels.max() != -1 else 0
    return final_labels

def save_results(output_dir, labels, coords):
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "cluster_labels.npy"), labels)
    sio.savemat(os.path.join(output_dir, "cluster_labels.mat"), {"labels": labels})
    np.save(os.path.join(output_dir, "voxel_coords.npy"), coords)

def plot_clusters(coords, labels, title="Cluster Visualization"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=labels, cmap='tab20', s=2)
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()


# def VisualizeOpen3d(clusteredNPY_Path, clusteredNPY_Coords):
#     # import numpy as np
#     # import open3d as o3d
#     # import os

#     # Load saved cluster labels and coordinates
#     labels = np.load(clusteredNPY_Path)
#     coords = np.load(clusteredNPY_Coords)

#     # labels = np.load(r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\src\clustering_output\cluster_labels.npy")
#     # coords = np.load(r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\src\clustering_output\voxel_coords.npy")

#     # Get unique cluster labels (excluding noise label -1)
#     unique_labels = np.unique(labels)
#     unique_labels = unique_labels[unique_labels != -1]

#     # Directory to optionally save .ply files
#     # os.makedirs("o3d_clusters", exist_ok=True)

#     for label in unique_labels:
#         cluster_coords = coords[labels == label]

#         # Create Open3D point cloud
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(cluster_coords)

#         # Optional: assign color per cluster
#         color = np.random.rand(3)  # random RGB
#         pcd.paint_uniform_color(color)

#         # Visualize
#         print(f"Showing cluster {label} with {len(cluster_coords)} points...")
#         o3d.visualization.draw_geometries([pcd])

        # Save if you want
        # o3d.io.write_point_cloud(f"o3d_clusters/cluster_{label}.ply", pcd)


# Example use:
# volume = load_volume("yourfile.mat")
# X = extract_features(volume)
# X_scaled = StandardScaler().fit_transform(X)
# kmeans_labels = run_kmeans(X_scaled, n_clusters=4)
# final_labels = run_dbscan_per_cluster(X_scaled, kmeans_labels)
# plot_clusters(X[:, :3], final_labels)
# save_results("output_dir", final_labels, X[:, :3])