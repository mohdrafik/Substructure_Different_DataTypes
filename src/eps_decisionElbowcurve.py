# Re-import necessary libraries after kernel reset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

import sys

def analyze_dbscan_parameters_knn(X_scaled, n_neighbors=50, save_plot=False, save_dir=None):
    """
    Efficient analysis using k-nearest neighbors distances (memory-friendly)
    """
    print(f" Computing {n_neighbors} nearest neighbors distances for {X_scaled.shape[0]} points...")

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)

    # Use the distance to the k-th nearest neighbor
    k_dists = distances[:, -1]
    k_dists_sorted = np.sort(k_dists)

    # Plot the elbow
    plt.figure(figsize=(10, 5))
    plt.plot(k_dists_sorted, label=f'{n_neighbors}th Nearest Neighbor Distance')
    plt.xlabel("Points (sorted)")
    plt.ylabel("Distance")
    plt.title(f"DBSCAN Elbow Plot (k = {n_neighbors})")
    plt.grid(True)
    plt.legend()

    if save_plot and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"dbscan_elbow_k{n_neighbors}.png"))
        print(f"Saved elbow plot to: {save_dir}")
    
    plt.show()

    suggested_eps = np.percentile(k_dists, 90)
    print(f"\n Suggested `eps` (90th percentile): {suggested_eps:.3f}")
    return suggested_eps

# ||| BELOW  CODE shows the memory error: 
# def analyze_dbscan_parameters(X_scaled, k=10):
#     """ 
#     K-distance Elbow Plot: Helps find a natural "elbow" point to choose eps.
#     Histogram of Pairwise Distances: Shows the overall distance distribution.
#     How to Interpret:
#     min_distance: Smallest non-zero pairwise distance in the normalized data.
#     max_distance: Largest pairwise distance.
#     median_distance: Middle value of all distances.
#     recommended_eps_range: A good starting point for choosing eps:
#     Lower bound: slightly above min distance (e.g. min + 0.2)
#     Upper bound: up to the median distance.     
#     """
#     # Compute pairwise distances
#     dists = squareform(pdist(X_scaled))
#     min_dist = np.min(dists[np.nonzero(dists)])
#     max_dist = np.max(dists)
#     median_dist = np.median(dists)

#     # k-distance graph
#     neighbors = NearestNeighbors(n_neighbors=k)
#     neighbors_fit = neighbors.fit(X_scaled)
#     distances, indices = neighbors_fit.kneighbors(X_scaled)
#     k_distances = np.sort(distances[:, k-1])  # k-th NN distances

#     # Plot elbow (k-distance)
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(k_distances)
#     plt.title(f"K-distance Graph (k={k})")
#     plt.xlabel("Points sorted by distance")
#     plt.ylabel(f"{k}-th Nearest Distance")
#     plt.grid(True)

#     # Histogram of distances (after normalization)
#     plt.subplot(1, 2, 2)
#     plt.hist(dists[np.triu_indices_from(dists, k=1)], bins=50, color='skyblue', edgecolor='black')
#     plt.title("Histogram of Pairwise Distances")
#     plt.xlabel("Distance")
#     plt.ylabel("Frequency")
#     plt.tight_layout()
#     plt.show()

#     return {
#         "min_distance": min_dist,
#         "max_distance": max_dist,
#         "median_distance": median_dist,
#         "recommended_eps_range": (round(min_dist + 0.2, 2), round(median_dist, 2)),
#         "recommended_min_samples": k
#     }


if __name__ =="__main__":
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
    X_scaled = StandardScaler().fit_transform(X)   # Z-score scaling/normalization -> zero mean, unit variance

    eps = analyze_dbscan_parameters_knn(X_scaled, n_neighbors=20, save_plot=True, save_dir="results/eps_analysis")
    
    # command = input("if priority is for Proximity of points compare  to intensity enter d = 0 otherwise for both distance and intensity enter d = 1")
    # if command == 0 :
    #     coords_scaled = StandardScaler().fit_transform(coords)
    #     intensity_scaled = intensity / 10.0  # down-weight intensity
    #     X_weighted = np.hstack((coords_scaled, intensity_scaled))
    # else:
        
    # analyze_dbscan_parameters(X_scaled)

    # coords_scaled = StandardScaler().fit_transform(coords)
    # intensity_scaled = intensity / 10.0  # down-weight intensity
    # X_weighted = np.hstack((coords_scaled, intensity_scaled))

#  Suggested `eps` (90th percentile): 0.030