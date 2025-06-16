import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from path_manager import addpath
addpath()
from listspecificfiles import readlistFiles

from pathlib import Path

from functools import wraps

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

    def __init__(self, datapath, output_dir, relativeDataPath, filesuffix = None, intensityBased_Kmeans = True, THRESHOLD_VALUE = False, n_clusters = None, eps_extractedFromData =None, min_samples_loadFromData = None):

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
        self.datapath = Path(datapath)
        self.output_dir = Path(output_dir)
        self.THRESHOLD_VALUE = THRESHOLD_VALUE 
        self.n_clusters = n_clusters if n_clusters is not None else  6
        self.relativeDataPath = relativeDataPath
        self.filesuffix = filesuffix if filesuffix is not None else '.npy'  

        self.intensityBased_Kmeans = intensityBased_Kmeans
        self.eps_extractedFromData  = eps_extractedFromData  if eps_extractedFromData  is not None else 0.06
        self.min_samples_loadFromData  = min_samples_loadFromData  if min_samples_loadFromData  is not None else 5
        
        # self.fpath = readlistFiles(self.relativeDataPath,'npy').file_with_Path() 
        self.fpath = readlistFiles(self.relativeDataPath,self.filesuffix).file_with_Path() 


    # THRESHOLD_VALUE =  1.334
    @logfunction
    def load_volume(self,filepath):
        if filepath.endswith('.npy'):
            volume = np.load(filepath)

        elif filepath.endswith('.mat'):
            mat = sio.loadmat(filepath)
            # Assuming your volume variable is named 'volume' in .mat
            volume = next(v for v in mat.values() if isinstance(v, np.ndarray) and (v.ndim == 3 or v.ndim == 2))
        else:
            raise ValueError("Unsupported file format. Use .mat or check the key for .mat File or .npy")

        if self.THRESHOLD_VALUE is not None:
            # THRESHOLD_VALUE = 1.334
            volume[volume == self.THRESHOLD_VALUE] = 0  # Threshold to remove background

        else:
            print(f"data volume is alreday thresholded and proceesed as Can see the proceesd path:{self.fpath}")

        return volume

    @logfunction
    def extract_features(self,volume, onlyPositiveValues = False, Nonzeros =True ):
        coords = np.array(np.nonzero(volume)).T  # coordinates of non-zero/mainData/foreground/signal values, output will be -->  [z1,x1,y1]

        if onlyPositiveValues: 
            intensities = volume[volume > 0].flatten().reshape(-1, 1)
        elif Nonzeros:
            intensities = volume[volume != 0].flatten().reshape(-1, 1)
        else:
            intensities = volume.flatten().reshape(-1,1)

        return np.hstack((coords, intensities))  # return like this [0,1,2, 1.334]
    @logfunction
    def run_kmeans(self, All_extracted_faetureData, scaling = False, intensityFeature_only = True, allFeatures = False):
        """ All_extracted_featureData: is return from the extract_faetures() :  [[0,1,2, 1.334] ... [coords,intensity_value] .. []] of size Nx4 
        allFeatures and intensityFeature_only is complementary to ecah other means one True and other is False.

     """
        if intensityFeature_only and scaling:

            data_scaled = StandardScaler().fit_transform(All_extracted_faetureData[:,3])

            # DataFor_kmeans = data_scaled[:,3]  # intensities[:,0] or X[:,3]  # Extracting only the intensity values from X, which is the last column(here is 4th column) of X
            DataFor_kmeans = data_scaled.reshape(-1, 1)
        
        elif intensityFeature_only :

            DataFor_kmeans = All_extracted_faetureData[:,3].reshape(-1,1)  # intensities[:,0] or X[:,3]  # Extracting only the intensity values from X, which is the last column(here is 4th column) of X
        elif allFeatures and scaling :

            data_scaled = StandardScaler().fit_transform(All_extracted_faetureData)

            DataFor_kmeans = data_scaled

        elif allFeatures:
            
            DataFor_kmeans = All_extracted_faetureData  # intensities[:,0] or X[:,3]  # Extracting only the intensity values from X, which is the last column(here is 4th column) of X
        else:

            print(f" entering wrong conditions: ! check")


        kmeans = KMeans(n_clusters = self.n_clusters, init= 'k-means++', random_state=42)

        return kmeans.fit_predict(DataFor_kmeans)  # kmeans_labels -> returns labels directly 
    

    # def run_dbscan_per_cluster(self, X_scaled, kmeans_labels, eps=0.06, min_samples=5):
    @logfunction
    def run_dbscan_per_cluster(self, X_scaled, kmeans_labels):
        final_labels = -np.ones(len(X_scaled), dtype=int)
        label_offset = 0
        for cluster_id in np.unique(kmeans_labels):
            indices = np.where(kmeans_labels == cluster_id)[0]
            db = DBSCAN(eps=self.eps_extractedFromData, min_samples= self.min_samples_loadFromData) # using the DBSCAN 
            sub_labels = db.fit_predict(X_scaled[indices])
            sub_labels[sub_labels != -1] += label_offset
            final_labels[indices] = sub_labels
            label_offset += sub_labels.max() + 1 if sub_labels.max() != -1 else 0
        return final_labels
    @logfunction
    def save_results(self,labels, coords,fileFullpath):
        os.makedirs(self.output_dir, exist_ok=True)
        np.save(os.path.join(self.output_dir, "cluster_labels.npy"), labels)
        sio.savemat(os.path.join(self.output_dir, "cluster_labels.mat"), {"labels": labels})
        np.save(os.path.join(self.output_dir, "voxel_coords.npy"), coords)


        # For saving the k-means cluster only and corresponding coordinates and labels results -------- >
        kmeans_labels = labels
        kmeans_coords_with_labels = np.hstack((coords, kmeans_labels.reshape(-1, 1)))  # [x, y, z, kmeans_label]
        # Save as .npy
        fullFilepath = Path(fileFullpath) 
        self.output_dir = self.output_dir/"seperatematplot"
        self.output_dir.mkdir(parents=True,exist_ok = True)
        kmeans_intResultDir = self.output_dir/f"kmIntensity{fullFilepath.stem}"
        np.save(kmeans_intResultDir,kmeans_coords_with_labels) # save .npy
        sio.savemat(kmeans_intResultDir,{"labels":kmeans_coords_with_labels})  # save .mat 

        # kmeans_intResultDir = os.path.join(self.output_dir,f"kmIntensity{self.fpath.stem}")
        # os.makedirs(kmeans_intResultDir,exist_ok=True)
        # np.save(os.path.join(kmeans_intResultDir, "kmeans_coords_labels.npy"), kmeans_coords_with_labels)
        # # Save as .mat
        # sio.savemat(os.path.join(kmeans_intResultDir, "kmeans_coords_labels.mat"), {"kmeans_coords_labels": kmeans_coords_with_labels})


    @logfunction
    def plot_clusters(self, coords, labels, title):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=labels, cmap='tab20', s=2)
        plt.title(title)
        plt.colorbar(scatter)
        plt.show()


if __name__=="__main__":
    # Example usage of the class
  
    fpath = readlistFiles(r'data\processed\main_fgdata','.npy').file_with_Path()
    fpathanyfile = Path(fpath[0])
    datapath= fpathanyfile.parent
    print(datapath)
    RES_DIR = Path.cwd()/"results"
    kmeansdbscan_Results = Path(RES_DIR)/ "kmenasdbscan_Results"
    kmeansdbscan_Results = Path(kmeansdbscan_Results)
    relativeDataPath = r'data\processed\main_fgdata'

    test_instance = EnhancedClustering(datapath=datapath, output_dir=kmeansdbscan_Results,
                                   relativeDataPath=relativeDataPath, filesuffix='.npy', THRESHOLD_VALUE=None, n_clusters=6)
    
    # Run all methods in sequence on each file
    for filewithpath in test_instance.fpath:
    
        filewithpath = Path(filewithpath)
        data = np.load(filewithpath)
        print(f"\n File: {filewithpath.name} and \n stem:{filewithpath.stem} \n Data shape: {data.shape}")

        # Run all methods in sequence on each file's data
        # Pass string path to load_volume to avoid Path/str issues with endswith
        volume = test_instance.load_volume(str(filewithpath))
        features = test_instance.extract_features(volume)
        kmeans_labels = test_instance.run_kmeans(features)
        dbscan_labels = test_instance.run_dbscan_per_cluster(features, kmeans_labels)
        test_instance.save_results(dbscan_labels,features[:, :3],fileFullpath=filewithpath)

        test_instance.plot_clusters(features[:, :3], dbscan_labels, title=f"{filewithpath.stem}")



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