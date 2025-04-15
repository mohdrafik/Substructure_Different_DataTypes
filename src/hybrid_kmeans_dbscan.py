import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(X_scaled)

def run_dbscan_per_cluster(X_scaled, kmeans_labels, eps=0.6, min_samples=5):
    final_labels = -np.ones(len(X_scaled), dtype=int)
    label_offset = 0
    for cluster_id in np.unique(kmeans_labels):
        indices = np.where(kmeans_labels == cluster_id)[0]
        db = DBSCAN(eps=eps, min_samples=min_samples)
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


# Example use:
# volume = load_volume("yourfile.mat")
# X = extract_features(volume)
# X_scaled = StandardScaler().fit_transform(X)
# kmeans_labels = run_kmeans(X_scaled, n_clusters=4)
# final_labels = run_dbscan_per_cluster(X_scaled, kmeans_labels)
# plot_clusters(X[:, :3], final_labels)
# save_results("output_dir", final_labels, X[:, :3])