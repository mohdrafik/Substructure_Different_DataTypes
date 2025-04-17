
# Hybrid Clustering with K-Means + DBSCAN for 3D Volume Data

import os
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import open3d as o3d
import matplotlib.pyplot as plt

# --- USER OPTIONS ---

import os 
from pathlib import Path
import random

# SRCFILES  = Path.cwd().parent
SRCFILES  = Path(__file__).resolve().parent.parent

DATAFILES  = SRCFILES/"data"/"raw_npyData"

listfiles = os.listdir(str(DATAFILES))
# Datafile = random.choice(listfiles)
DATAFILE_NAME = "Tomogramma_BuddingYeastCell.npy"

for file in listfiles:
    if file == DATAFILE_NAME:
        DATAFILES = str(DATAFILES)
        input_path = os.path.join(DATAFILES,DATAFILE_NAME)
        print(f"input path : {input_path}")
        
# print(f" --------------->  check step by step : SRCFILES: {SRCFILES} --> DATAFILES: {DATAFILES} --> \n listfiles: {listfiles} \n --> Datafile random choice: {Datafile}")
# Datafile = r"C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\data\raw\Tomogramma_BuddingYeastCell.mat"

# Choose input type
input_type = "npy"   #"mat"   or "npy"
input_path = input_path  # or .npy
volume_key = "volume"  # for .mat file: key inside the .mat dict

output_dir = "cluster_output"
output_dir = SRCFILES/"results"/"hybrid_Kdbcluster"
kmeans_k = 7

# --- LOAD VOLUME DATA ---
if input_type == "mat":
    mat_data = sio.loadmat(input_path)
    volume = mat_data[volume_key]
elif input_type == "npy":
    volume = np.load(input_path)
else:
    raise ValueError("Invalid input_type. Choose 'mat' or 'npy'.")

# --- EXTRACT NONZERO VOXELS AS POINT CLOUD ---
coords = np.array(np.nonzero(volume)).T  
# it will returns the coordinate of each nonzero values in volume and formate will be like this [[]
# coords =
# [z1,x1,y1]
# [z2,x2,y2]
# ........
# [zn,xn,yn]]

intensities = volume[volume > 0].reshape(-1, 1)
X = np.hstack((coords, intensities))  # shape: (N, 4)
# np.hstack() horizontally stacks arrays (i.e., along columns / axis=1), meaning it concatenates them side by side. a = [[1],[2],[3]] , b = [[10],[20],[30]]
# np.hstack(a,b) --> results will be [[1,10],[2,20],[3,30]]

# --- SCALE FEATURES ---
X_scaled = StandardScaler().fit_transform(X)

# --- APPLY K-MEANS ---
kmeans = KMeans(n_clusters=kmeans_k,init = 'k-means++',random_state=42).fit(X_scaled)
# kmeans = KMeans(n_clusters=kmeans_k, init = 'k-means++', random_state=42).fit(X)
kmeans_labels = kmeans.labels_

# --- HYBRID: DBSCAN WITHIN EACH K-MEANS CLUSTER ---
final_labels = -np.ones(len(X), dtype=int)
label_offset = 0

for cluster_id in np.unique(kmeans_labels):
    print(f"i have compl k-means now in dbsacn, cluster_id: {cluster_id}")
    indices = np.where(kmeans_labels == cluster_id)[0]
<<<<<<< HEAD
    X_sub = X_scaled[indices]
    # X_sub = X[indices]
    db = DBSCAN(eps=1.5, min_samples=10).fit(X_sub)
=======
    # X_sub = X_scaled[indices]
    X_sub = X[indices]
    db = DBSCAN(eps=1.5, min_samples=10).fit(X_sub)  # using the DBS
>>>>>>> f4b18b3cf10e5b90342cad42609c959f16d1df35
    db_labels = db.labels_
    db_labels[db_labels != -1] += label_offset
    final_labels[indices] = db_labels
    label_offset += db_labels.max() + 1

# --- SAVE RESULTS ---
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "cluster_labels.npy"), final_labels)
sio.savemat(os.path.join(output_dir, "cluster_labels.mat"), {"labels": final_labels})
np.save(os.path.join(output_dir, "voxel_coords.npy"), coords)

# Save each cluster as separate file
for label in np.unique(final_labels):
    print(f"label: {label}")

    if label == -1:
        continue  # skip noise
    cluster_points = coords[final_labels == label]
    cluster_intensity = intensities[final_labels == label]
    cluster_data = np.hstack((cluster_points, cluster_intensity))
    subdir = os.path.join(output_dir, f"cluster_{label}")
    os.makedirs(subdir, exist_ok=True)
    np.save(os.path.join(subdir, f"cluster_{label}.npy"), cluster_data)
    sio.savemat(os.path.join(subdir, f"cluster_{label}.mat"), {"data": cluster_data})
    

    # # --- VISUALIZE USING Open3D ---
    # print(f"visualizing the label: {label}")
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cluster_points)
    # colors = np.tile(np.random.rand(3), (len(cluster_points), 1))
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # # Estimate normals for mesh
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    # pcd.orient_normals_consistent_tangent_plane(10)

    # # Create mesh using Ball Pivoting
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 2 * avg_dist
    # try:
    #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #         pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    #     mesh.compute_vertex_normals()
    #     o3d.visualization.draw_geometries([pcd, mesh], window_name=f"Cluster {label} (Mesh + PointCloud)")
    # except:
    #     o3d.visualization.draw_geometries([pcd], window_name=f"Cluster {label} (PointCloud Only)")




# if __name__ =="__main__":

#     import numpy as np
#     import open3d as o3d
#     import os
#     from pathlib import Path

#     SRCFILES  = Path(__file__).resolve().parent
#     print(f"------------> {SRCFILES} AND \n")

#     RESFilesINsrc = SRCFILES/"clustering_output"
#     RESFilesINsrc = os.path.normpath(RESFilesINsrc)
#     print(f"------------> {SRCFILES} AND \n {RESFilesINsrc}")
    
#     clusteredNPY_Path = os.path.join(str(RESFilesINsrc),'cluster_labels.npy')
#     clusteredNPY_Coords = os.path.join(str(RESFilesINsrc),'voxel_coords.npy')

#     print(f" here to check the final path -------> \n  clusteredNPY_Path: {clusteredNPY_Path},\n  clusteredNPY_Coords :{clusteredNPY_Coords} \n" )


#     label_path = clusteredNPY_Path
#     coord_path = clusteredNPY_Coords 

#     # from meshvisClustCordLabels import *
#     visualize_and_save_clusters(label_path, coord_path, output_dir="o3d_clusters")

#     # Example usage:
#     # visualize_and_save_clusters("clustering_output/cluster_labels.npy", "clustering_output/voxel_coords.npy")

