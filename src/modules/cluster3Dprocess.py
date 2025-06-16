# Install required packages
# !pip install faiss-cpu annoy plotly hdbscan

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors


try:
    import hdbscan # type: ignore
    hdbscan_available = True
except ImportError:
    hdbscan_available = False

try:
    import faiss # type: ignore
    faiss_available = True
except ImportError:
    faiss_available = False

class Cluster3DProcessor:
    """
     New Features Added:
         Feature	           Description
        Method Selection	 Choose between dbscan, hdbscan, or faiss
        Downsampling	     Prevents memory issues with large data
        Save Subvolumes	     Extracts and saves cluster-specific sub-volumes
        Bounding Box	     Each subvolume is saved within its bounding box
        Interactive Plot	 Generates interactive 3D visualization with Plotly
        Output Directory	 All results saved in cluster_output/
    
    """

    def __init__(self, npy_file,Baseoutput_dir, method='dbscan', save_outputs=True, downsample=True, max_points=100000,save_subvolumes=True, visualize_interactive=True):
        self.npy_file = npy_file
        self.method = method.lower()
        self.save_outputs = save_outputs
        self.downsample = downsample
        self.max_points = max_points
        self.save_subvolumes = save_subvolumes
        self.visualize_interactive = visualize_interactive
        self.coords = None
        self.coords_scaled = None
        self.labels = None
        self.data = None
        self.output_dir = os.path.normpath(Baseoutput_dir/"cluster_output")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_preprocess(self):
        # filename = self.npy_file
        # BASE_DIR = Path(__resolve__)
        # filepath =  
        self.data = np.load(self.npy_file)

        if self.data.ndim != 3:
            raise ValueError("Input .npy file is not a 3D array.")

        self.coords = np.argwhere(self.data > 0)
        if self.coords.shape[0] == 0:
            raise ValueError("No non-zero voxels found.")

        self.coords_scaled = MinMaxScaler().fit_transform(self.coords)

        if self.downsample and self.coords_scaled.shape[0] > self.max_points:
            idx = np.random.choice(self.coords_scaled.shape[0], size=self.max_points, replace=False)
            self.coords_scaled = self.coords_scaled[idx]
            self.coords = self.coords[idx]

    def cluster(self):
        if self.method == 'dbscan':
            self.labels = DBSCAN(eps=0.05, min_samples=10).fit_predict(self.coords_scaled)
        elif self.method == 'hdbscan' and hdbscan_available:
            self.labels = hdbscan.HDBSCAN(min_cluster_size=20).fit_predict(self.coords_scaled)
        elif self.method == 'faiss' and faiss_available:
            self.labels = self.cluster_with_faiss()
        else:
            raise NotImplementedError(f"Clustering method '{self.method}' not supported or dependencies missing.")

    def cluster_with_faiss(self):
        data = self.coords_scaled.astype('float64')
        index = faiss.IndexFlatL2(data.shape[1])
        index.add(data)
        neighbors = 10
        _, I = index.search(data, neighbors)
        model = DBSCAN(eps=0.05, min_samples=10, metric='euclidean')
        return model.fit_predict(data)

    def save_results(self):
        base_name = os.path.splitext(os.path.basename(self.npy_file))[0]

        result_df = pd.DataFrame(self.coords, columns=['X', 'Y', 'Z'])
        result_df['Cluster'] = self.labels
        csv_path = os.path.join(self.output_dir, f'{base_name}_labels.csv')
        result_df.to_csv(csv_path, index=False)

        if self.visualize_interactive:
            fig = px.scatter_3d(result_df, x='X', y='Y', z='Z', color='Cluster',
                                title=f'{base_name} - Interactive Clusters')
            fig.write_html(os.path.join(self.output_dir, f'{base_name}_interactive.html'))

        if self.save_subvolumes:
            for label in np.unique(self.labels):
                if label == -1:
                    continue
                mask = self.labels == label
                cluster_coords = self.coords[mask]
                min_coords = cluster_coords.min(axis=0)
                max_coords = cluster_coords.max(axis=0)
                bbox = self.data[min_coords[0]:max_coords[0]+1,
                                 min_coords[1]:max_coords[1]+1,
                                 min_coords[2]:max_coords[2]+1]
                np.save(os.path.join(self.output_dir, f'{base_name}_cluster_{label}_subvolume.npy'), bbox)

    def run(self):
        self.load_and_preprocess()
        self.cluster()
        if self.save_outputs:
            self.save_results()
        return self.coords, self.labels
    


if __name__ == "__main__":
    import os
    # PROJECT_DIR = Path.cwd().parent
    PROJECT_DIR = Path.cwd()
    input_dir = PROJECT_DIR/"data"/"raw_npyData"
    # print(f" input path of the raw npy files: {input_dir}")
    Baseoutput_dir = PROJECT_DIR/"results"
    # Example usage:

    # input_dir= input_dir
    Baseoutput_dir= Baseoutput_dir
    method='dbscan'
    save_outputs=True
    downsample=True
    max_points=100000
    save_subvolumes=True
    visualize_interactive=True

    filesnpy = os.listdir(input_dir)
    
    for npyfile in filesnpy:
        # npyfilePath = Path(__file__).resolve().parent.parent.parent
        # npyfileDataFull = npyfilePath/"data"/"raw_npyData"/npyfile
        # print(f" \n see here the used Path(__file__).resolve().parent.parent: {npyfilePath} \n and with filename path : {npyfileDataFull} \n ")
        npyfile_path = input_dir / npyfile

        processor = Cluster3DProcessor(
                    npy_file = npyfile_path,
                    Baseoutput_dir= Baseoutput_dir,
                    method=method,
                    save_outputs=save_outputs,
                    downsample= downsample,
                    max_points=max_points,
                    save_subvolumes=save_subvolumes,
                    visualize_interactive=visualize_interactive
                )
        
        coords, labels = processor.run()
        print(f"Finished {os.path.basename(npyfile)} - Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")

