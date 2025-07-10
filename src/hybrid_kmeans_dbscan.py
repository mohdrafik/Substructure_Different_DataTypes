import matplotlib.cm as cm
import math
from functools import wraps
from pathlib import Path
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

from matplotlib import colormaps

from path_manager import addpath
addpath()

from listspecificfiles import readlistFiles

# @staticmethod  # on top if we are defining do't need @staticmethod.

def logfunction(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(
            f"\n ---------------------> /// Implementing method: {func.__name__} \\\ <------------------------------------------------------- \n")
        results = func(*args, **kwargs)
        # print(f"\n ---------------------> /// Finished executing method: {func.__name__} \\\ <--------------------------------------------------\n")
        return results
    # print(f"\n ---------------------> /// Finished executing method:  \\\ <--------------------------------------------------\n")
    return wrapper


class EnhancedClustering:

    def __init__(self, relativeDataPath, output_dir= None, datapath=None, filesuffix=None, intensityBased_Kmeans=True, THRESHOLD_VALUE=None, n_clusters=None, eps_extractedFromData=None, min_samples_loadFromData=None):

        """ 
        it will use dteh both kmeans and dbscan to cluster the data
        Args:
            datapath (str): Path to the input data file (.npy or .mat).
            output_dir (str): Directory where the output files will be saved.
            relativeDataPath (str): Relative path to the data directory.
            filesuffix (str, optional): Suffix of the files to be read. Defaults to None, which will use '.npy' or '.mat'.
            intensityBased_Kmeans (bool, optional): Whether to use intensity-based KMeans. Defaults to True.
            THRESHOLD_VALUE (float, optional): Threshold value for filtering the volume. Defaults to None.
            eps_extractedFromData  (bool, optional): Load epsilon value from JSON. Defaults to False.
            min_samples_loadFromData  (bool, optional): Load min_samples value from JSON. Defaults to False.
        """
        self.datapath = Path(datapath) if datapath is not None else Path.cwd()/relativeDataPath
        self.output_dir = Path(output_dir) if output_dir is not None else Path.cwd().parent/"results"/"kmeans_fgdata"
        self.THRESHOLD_VALUE = THRESHOLD_VALUE
        self.n_clusters = n_clusters if n_clusters is not None else 6
        self.relativeDataPath = relativeDataPath
        self.filesuffix = filesuffix if filesuffix is not None else '.npy'

        self.intensityBased_Kmeans = intensityBased_Kmeans
        self.eps_extractedFromData = eps_extractedFromData if eps_extractedFromData is not None else 0.06
        self.min_samples_loadFromData = min_samples_loadFromData if min_samples_loadFromData is not None else 20

        # self.fpath = readlistFiles(self.relativeDataPath,'npy').file_with_Path()
        self.fpath = readlistFiles(
            self.relativeDataPath, self.filesuffix).file_with_Path()

    # THRESHOLD_VALUE =  1.334

    @logfunction
    def load_volume(self, filepath):
        filepath = Path(filepath)

        filename = filepath.name

        if filename.endswith('.npy'):
            volume = np.load(filepath)

        elif filename.endswith('.mat'):
            mat = sio.loadmat(filepath)
            # Assuming your volume variable is named 'volume' in .mat
            volume = next(v for v in mat.values() if isinstance(
                v, np.ndarray) and (v.ndim == 3 or v.ndim == 2))
        else:
            raise ValueError(
                "Unsupported file format. Use .mat or check the key for .mat File or .npy")

        if self.THRESHOLD_VALUE is not None:
            # THRESHOLD_VALUE = 1.334
            # Threshold to remove background
            volume[volume == self.THRESHOLD_VALUE] = 0

        else:
            print(
                f" \n data volume is alreday thresholded and proceesed as Can see the proceesd path:{self.fpath[0]}\n")

        return volume

    @logfunction
    def extract_feature_data(self, volume, onlyPositiveValues=False, Nonzeros=True):
        # coordinates of non-zero/mainData/foreground/signal values, output will be -->  [z1,x1,y1]

        if  onlyPositiveValues and Nonzeros:
            raise ValueError(
                "onlyPositiveValues and Nonzeros cannot be both True. Choose one of them.")

        if onlyPositiveValues:
            intensities = volume[volume > 0].flatten().reshape(-1, 1)
            coords = np.where(volume > 0)  # coordinates of non-zero values
            coords = np.array(coords).T  # transpose to get shape (N, 3)
        elif Nonzeros:
            intensities = volume[volume != 0].flatten().reshape(-1, 1)
            coords = np.where(volume != 0)  # coordinates of non-zero values
            coords = np.array(coords).T  # transpose to get shape (N, 3)
        else:
            intensities = volume.flatten().reshape(-1, 1)
            coords = np.where(np.isfinite(volume))  # coordinates of finite values
            coords = np.array(coords).T  # transpose to get shape (N, 3)

        # return like this [0,1,2, 1.334]
        coords_and_intensities_both = np.hstack((coords, intensities))

        return  coords, coords_and_intensities_both

    @logfunction
    def run_kmeans(self, All_extracted_faetureData, scaling=False, intensityFeature_only=True, allFeatures=False):
        """ 
        All_extracted_featureData: is return from the extract_faetures() :  [[0,1,2, 1.334] ... [coords,intensity_value] .. []] of size Nx4 
        allFeatures and intensityFeature_only is complementary to ecah other means one True and other is False.

       """
        if intensityFeature_only and scaling:

            data_scaled = StandardScaler().fit_transform(
                All_extracted_faetureData[:, 3])

            # DataFor_kmeans = data_scaled[:,3]  # intensities[:,0] or X[:,3]  # Extracting only the intensity values from X, which is the last column(here is 4th column) of X
            DataFor_kmeans = data_scaled.reshape(-1, 1)

        elif intensityFeature_only:

            # intensities[:,0] or X[:,3]  # Extracting only the intensity values from X, which is the last column(here is 4th column) of X
            DataFor_kmeans = All_extracted_faetureData[:, 3].reshape(-1, 1)

        elif allFeatures and scaling:

            data_scaled = StandardScaler().fit_transform(All_extracted_faetureData)

            DataFor_kmeans = data_scaled

        elif allFeatures:

            # intensities[:,0] or X[:,3]  # Extracting only the intensity values from X, which is the last column(here is 4th column) of X
            DataFor_kmeans = All_extracted_faetureData
        else:

            print(f" entering wrong conditions: ! check")

        kmeans = KMeans(n_clusters=self.n_clusters,
                        init='k-means++', random_state=42)

        # kmeans_labels -> returns labels directly
        return kmeans.fit_predict(DataFor_kmeans) 
        # kmeans.labels_ --> kmeans_labels is one row (1xN) of labels (0,1,.., n_clusters -1) , N is the number row = 200x200x200 for data size of z= 200,y = 200, x =200, each coordinates has a label. 
        # as output → array of cluster assignments for each data point, storing the cluster labels for all samples in the kmeans_labels variable, so you can use them later for saving or analyzing clusters / future use.

    # def run_dbscan_per_cluster(self, feature_data, kmeans_labels, eps=0.06, min_samples=5):

    @logfunction
    def run_dbscan_per_cluster(self, feature_data, kmeans_labels, DBonlyintensity = True, withNormalization = True):
        dbscan_final_labels = -np.ones(len(feature_data), dtype=int) # array([-1, -1, -1, -1, -1 .......... ]) of size of featutr_data. 
        label_offset = 0
        for cluster_id in np.unique(kmeans_labels):
            indices = np.where(kmeans_labels == cluster_id)[0]   # np.where(condition) -> Tuple of arrays;  np.where(condition)[0] -> 1D array of indices
            # example. array([2, 3]),)-> without zero. , # array([2, 3]) -> with zero.
            db = DBSCAN(eps=self.eps_extractedFromData,
                        min_samples=self.min_samples_loadFromData)  # using the DBSCAN
            
            if DBonlyintensity:
                if withNormalization:
                    data_scaled = StandardScaler().fit_transform(feature_data[indices, 3].reshape(-1,1))
                    # sub_labels = db.fit_predict(data_scaled).reshape(-1, 1) # want to apply dbscan only on intensity/RI values.
                    sub_labels = db.fit_predict(data_scaled) # .reshape(-1, 1)removed here want to apply dbscan only on intensity/RI values.
                else:
                    sub_labels = db.fit_predict(feature_data[indices, 3].reshape(-1, 1)) # want to apply dbscan only on intensity/RI values. 
            else:
                if withNormalization :
                    data_scaled = StandardScaler().fit_transform(feature_data[indices])
                    sub_labels = db.fit_predict(data_scaled)  # dbscan on all  
                else:
                    sub_labels = db.fit_predict(feature_data[indices])  # dbscan on all  

            # feature_data[indices] -> returns the indices corresponding rows from feature data. let'say if indices = [0,2,5]-> it return the 0,2,5th rows from the data. 
            # feature_data has shape (N, 4): [x, y, z, value]
            # indices contains the row indices in feature_data that belong to the current KMeans cluster,  

            sub_labels[sub_labels != -1] += label_offset
            dbscan_final_labels[indices] = sub_labels
            label_offset += sub_labels.max() + 1 if sub_labels.max() != -1 else 0

        return dbscan_final_labels

    @logfunction
    def save_results(self, dbscan_final_labels= None, kmeans_labels= None, coords = None, fileFullpath = None,saveAllfilesClustOneDir=None):

        # os.makedirs(self.output_dir, exist_ok=True)
        # np.save(os.path.join(self.output_dir, "cluster_labels.npy"), dbscan_final_labels)
        # sio.savemat(os.path.join(self.output_dir,
        #             "cluster_labels.mat"), {"labels": dbscan_final_labels})
        # np.save(os.path.join(self.output_dir, "voxel_coords.npy"), coords)

        if dbscan_final_labels is not None:
            fullFilepath = Path(fileFullpath)
            kdb_coords_with_final_labels = np.hstack((coords, dbscan_final_labels.reshape(-1, 1)))  # [x, y, z, dbscan_final_labels]
            output_dir_db = self.output_dir/ f"coord_dbfinal_Labels_{fullFilepath.stem}"
            output_dir_db.mkdir(parents=True, exist_ok=True)
            kmeans_dbres = output_dir_db/f"dbfinal{fullFilepath.stem}"
            # np.save(f"{kmeans_dbres}.npy", kdb_coords_with_final_labels)  # save .npy
            sio.savemat(f"{kmeans_dbres}.mat", {"labels": kdb_coords_with_final_labels})  # save .mat
            print(f"\n \n \n Saving the dbscan final labels and coordinates in the path: {kmeans_dbres} \n \n \n")


        # For saving the k-means cluster only and corresponding coordinates and labels results -------- >
        # kmeans_labels = kmeans_labels
        if kmeans_labels is not None:
            kmeans_coords_with_labels = np.hstack((coords, kmeans_labels.reshape(-1, 1)))  # [x, y, z, kmeans_label]
            # Save as .npy and .mat for all kmeans labels with coordinates.
            fullFilepath = Path(fileFullpath)
            if saveAllfilesClustOneDir:
                kmeans_intResultDir = self.output_dir / f"kmIntensity{fullFilepath.stem}"
                kmeans_intResultDir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir_sep = self.output_dir/ f"coord_kmeansLabels_{fullFilepath.stem}"
                output_dir_sep.mkdir(parents=True, exist_ok=True)
                kmeans_intResultDir = output_dir_sep/f"kmIntensity{fullFilepath.stem}"
                
            # np.save(f"{kmeans_intResultDir}.npy", kmeans_coords_with_labels)  # save .npy
            sio.savemat(f"{kmeans_intResultDir}.mat", {"kmeans_coords_labels": kmeans_coords_with_labels})  # save .mat
            print(f"\n \n \n Saving the kmeans labels and coordinates in the path: {kmeans_intResultDir} \n \n \n")

        # kmeans_intResultDir = os.path.join(self.output_dir,f"kmIntensity{self.fpath.stem}")
        # os.makedirs(kmeans_intResultDir,exist_ok=True)
        # np.save(os.path.join(kmeans_intResultDir, "kmeans_coords_labels.npy"), kmeans_coords_with_labels)
        # # Save as .mat
        # sio.savemat(os.path.join(kmeans_intResultDir, "kmeans_coords_labels.mat"), {"kmeans_coords_labels": kmeans_coords_with_labels})

    @logfunction
    def plot_clusters(self, coords, labels, title= None, savefilenamepng = None):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2], c=labels, cmap='tab20', s=2, alpha=0.3)
        plt.title(title)
        plt.colorbar(scatter)
        if savefilenamepng is not None:
            save_path = self.output_dir / f"{savefilenamepng}.png"
        else:
            save_path = self.output_dir / f"{all}_allclusterSimple.png"

        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close(fig)

    @logfunction
    def plot_dbscan_clusters_subplot(self, coords, dbscan_labels, filename_stem):
        """
        Plot all DBSCAN clusters (excluding noise) in subplots with custom colors.

        Args:
            coords (np.ndarray): Nx3 array of voxel coordinates.
            dbscan_labels (np.ndarray): Cluster labels assigned by DBSCAN.
            filename_stem (str): Base filename used for saving plots.
        """
        unique_labels = [label for label in np.unique(dbscan_labels) if label != -1]
        n_clusters = len(unique_labels)

        if n_clusters == 0:
            print(f"No valid clusters to plot for {filename_stem}.")
            return

        # Define color map
        # colormap = cm.get_cmap('tab20', n_clusters)
        # from matplotlib import colormaps  # New module introduced in recent versions
        colormap = colormaps.get_cmap('tab20')


        # Calculate subplot grid layout
        cols = math.ceil(math.sqrt(n_clusters))
        rows = math.ceil(n_clusters / cols)

        fig = plt.figure(figsize=(5 * cols, 4 * rows))
        for idx, label in enumerate(unique_labels):
            indices = np.where(dbscan_labels == label)[0]
            cluster_coords = coords[indices]

            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            # color = colormap(idx)  # Get a unique color for each cluster
            color = colormap(idx / n_clusters)  # fixed here
            ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2], s=2, c=[color])
            ax.set_title(f"Cluster {label}", fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        save_path = self.output_dir / f"{filename_stem}_dbscan_allclusters_colored.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)



if __name__ == "__main__":
    
    fpath = readlistFiles(r'data\processed\main_fgdata',
                          '.npy').file_with_Path()
    fpathanyfile = Path(fpath[0])
    datapath = fpathanyfile.parent
    print(datapath)
    RES_DIR = Path.cwd()/"results"
    kmeansdbscan_Results = Path(RES_DIR) / "kmenasdbscan_Results"
    # kmeansdbscan_Results = Path(kmeansdbscan_Results)
    relativeDataPath = r'data\processed\main_fgdata'

    test_instance = EnhancedClustering(datapath=datapath, output_dir=kmeansdbscan_Results,
                                       relativeDataPath=relativeDataPath, filesuffix='.npy', THRESHOLD_VALUE=None, eps_extractedFromData=0.01,min_samples_loadFromData=10, n_clusters=6)

    # Run all methods in sequence on each file
    count = 0
    for filewithpath in test_instance.fpath:
        count +=1
        filewithpath = Path(filewithpath)
        data = np.load(filewithpath)
        print(
            f"\n File: {filewithpath.name} and \n stem:{filewithpath.stem} \n Data shape: {data.shape}")

        # Run all methods in sequence on each file's data
        # Pass string path to load_volume to avoid Path/str issues with endswith
        volume = test_instance.load_volume(str(filewithpath))

        feature_data = test_instance.extract_feature_data(volume)

        kmeans_labels = test_instance.run_kmeans(feature_data)

        dbscan_labels = test_instance.run_dbscan_per_cluster(
            feature_data, kmeans_labels, DBonlyintensity=True, withNormalization=True)
        
        test_instance.save_results(
            dbscan_labels, kmeans_labels, feature_data[:, :3], fileFullpath=filewithpath)

        # test_instance.plot_clusters(
        #     feature_data[:, :3], dbscan_labels, title=f"{filewithpath.stem}", filename_stem = filewithpath.stem)
        
        test_instance.plot_dbscan_clusters_subplot(
            feature_data[:, :3], dbscan_labels, filewithpath.stem)
        if count ==10:
            break


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
# X = extract_feature_data(volume)
# feature_data = StandardScaler().fit_transform(X)
# kmeans_labels = run_kmeans(feature_data, n_clusters=4)
# dbscan_final_labels = run_dbscan_per_cluster(feature_data, kmeans_labels)
# plot_clusters(X[:, :3], dbscan_final_labels)
# save_results("output_dir", dbscan_final_labels, X[:, :3])