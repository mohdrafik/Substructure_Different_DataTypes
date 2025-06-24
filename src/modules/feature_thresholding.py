import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from path_manager import addpath
addpath()

from listspecificfiles import readlistFiles




class FeatureQuantileThresholding:
    """
    Now that your 3D datasets are normalized, we will proceed with Feature Extraction & Quantile-Based Thresholding
    to identify meaningful substructures.

    Why This Step is Important?
    - Feature Extraction helps in understanding the distribution of voxel intensities.
    - Quantile-Based Thresholding helps to filter noise and identify significant regions in the filename.

    What I Will Do?
    Step 1: Extract statistical features (mean, variance, quantiles)
    Step 2: Apply Quantile-Based Thresholding (0.95, 0.99 quantiles)
    Step 3: Visualize the thresholded regions in 3D slices
    """

    def __init__(self, relative_datadir, BASE_DIR=None, keyword=None, QunatilePercentile = None):

        self.relative_datadir = relative_datadir
        # self.save_dir = save_dir
        self.keyword = keyword if keyword is not None else '.npy'

        self.BASE_DIR = BASE_DIR if BASE_DIR is not None else Path.cwd()
        
        self.QuantilePercentile = QunatilePercentile if QunatilePercentile is not None else 0.99

        # Use readlistFiles class to get matched files
        self.filewithpath = readlistFiles(self.relative_datadir, self.keyword).file_with_Path()
        self.features = {}

        self.save_res_plot = self.BASE_DIR / "results" / "Quantilefeature"

    def load_data(self, filewithpath):
        return np.load(filewithpath)

    def extract_features(self, data):
        mean_val = np.mean(data)
        std_dev = np.std(data)
        # QunatPercentile quantile(data, 0.95)
        QuantileThres= np.quantile(data, self.QuantilePercentile)
        
        return mean_val, std_dev, QuantileThres

    def visualize_thresholds(self, filename, data, QunatileThres, save_plot=None,durationplotshow=None):
        # IEEE single column: ~3.5 inches wide, so use (3.5, 2.5) or (3.5, 3)
        figsize = (3.5, 3)
        fontsize = 8
        titlesize = 9

        maskQ = data > QunatileThres
        slice_index = data.shape[2] // 2

        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.imshow(data[:, :, slice_index], cmap="gray")
        plt.title("Original Slice", fontsize=titlesize)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(maskQ[:, :, slice_index], cmap="hot")
        plt.title(f"Thresholded @ Q={QunatileThres:.2f}", fontsize=titlesize)
        plt.axis("off")

        plt.suptitle(str(filename[:-4]), fontsize=titlesize)
        plt.tight_layout(pad=1.0)

        # Set font sizes for all text in the figure
        plt.rcParams.update({'font.size': fontsize})

        if save_plot and self.save_res_plot:
            visualize_thres_Save_path = self.save_res_plot/"Quantilemiddleslice_z"
            os.makedirs(visualize_thres_Save_path, exist_ok=True)
            # Clean filename for saving
            base_filename = os.path.basename(str(filename)).replace('.npy', '')
            png_path = os.path.join(visualize_thres_Save_path, f"{base_filename}_thresh.png")
            plt.savefig(png_path, dpi=600, bbox_inches='tight')
            print(f"Saved PNG: {png_path}")

        plt.show()
        pauseTime = durationplotshow if durationplotshow is not None else 3
        plt.pause(pauseTime)
        plt.close()
        



    def visualize_thresholds_slice_dynamic(self, filename, data, QunatileThres, save_plot=False, num_slices=None, durationplotshow = None):
        """
        Visualize original and thresholded slices from 3D data, and optionally save the plots.

        Parameters:
            filename (str): Used in the plot title and optional saved filename.
            data (np.ndarray): 3D volumetric data (shape: [x, y, z]).
            QunatileThres (float): Threshold value for segmentation.
            save_plot (bool): Whether to save the plot as an image.
            output_dir (str): Directory to save the plots if enabled.
            num_slices (int): Number of equally spaced slices to visualize.
        """
        # Configuration
        figsize = (7, 3)  # For side-by-side view
        fontsize = 8
        titlesize = 9

        # Generate binary mask
        maskQ = data > QunatileThres

        # Determine slice indices
        z_dim = data.shape[2]
        step = max(z_dim // (num_slices + 1), 1)
        slice_indices = list(range(step, z_dim - step, step))[:num_slices]

        for idx, slice_index in enumerate(slice_indices):
            fig, ax = plt.subplots(1, 2, figsize=figsize)

            # Original slice
            ax[0].imshow(data[:, :, slice_index], cmap="gray")
            ax[0].set_title(f"Original Slice @ z={slice_index}", fontsize=titlesize)
            ax[0].axis("off")

            # Thresholded slice
            ax[1].imshow(maskQ[:, :, slice_index], cmap="hot")
            ax[1].set_title(f"Thresholded @ Q={QunatileThres:.2f}", fontsize=titlesize)
            ax[1].axis("off")

            fig.suptitle(f"{filename} (Slice {idx+1})", fontsize=titlesize)
            plt.tight_layout(pad=1.5)
            plt.rcParams.update({'font.size': fontsize})

            # Save plot
            if save_plot:
                Qslicedynamic = self.save_res_plot/ "fixQSliceZ"
                os.makedirs(Qslicedynamic, exist_ok=True)
                safe_filename = f"{os.path.splitext(filename)[0]}_slice{slice_index}_thres.png"
                full_path = os.path.join(Qslicedynamic, safe_filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')

            # plt.show()
            # plt.close(fig)
            
            plt.show(block=False) # - displays the figure but allows code to continue running.
            durationplotshow = durationplotshow if durationplotshow is not None else 3
            plt.pause(durationplotshow) # - keeps the figure open for 3 seconds.
            plt.close() # - closes the figure window automatically.


    
    def visualize_all_views(self, data, filename, QunantileThresV, save_plot=False, durationplotshow = None, num_slices = None, figsize=(7, 5), dpi=300):
        
        """
        Visualize axial, coronal, and sagittal slices of a 3D data side-by-side in IEEE paper format.
        Parameters:
        - data: 3D NumPy array
        - filename: base filename for saving
        - save_dir: directory to save the output image (defaults to self.save_res_plot)
        - figsize: size of the figure (default fits IEEE single column width)
        - dpi: resolution of saved image
        """

        maskQ = data > QunantileThresV
        # volume = data*maskQ
        volume = maskQ
          

        x_dim = data.shape[0]
        y_dim = data.shape[1]
        z_dim = data.shape[2]

        step_x = max(x_dim//(num_slices+1), 1) # it will divide the whole size(index) in parts = (num_slices+1) if num_slicdes+1 > x_dim, -> 0 then it choose 1.
        step_y = max(y_dim//(num_slices+1), 1) # it will divide the whole size(index) in parts = (num_slices+1) if num_slicdes+1 > y_dim, -> 0 then it choose 1.
        step_z = max(z_dim//(num_slices+1), 1) # it will divide the whole size(index) in parts = (num_slices+1) if num_slicdes+1 > z_dim, -> 0 then it choose 1.
        
        slice_index_x = list(range(step_x, x_dim-step_x, step_x))[:num_slices] # why x_dim - step_x -->  because idx_x,starts, from step_x,2*step_x,3*step_x...
        # x_dim - step_x -> stop before the last few slices , [:num_slices] → first slices numbers values (if available)
        slice_index_y = list(range(step_y, y_dim-step_y, step_y))[:num_slices] # why x_dim - step_x -->  because idx_x,starts, from step_x,2*step_x,3*step_x...
        slice_index_z = list(range(step_z, z_dim-step_z, step_z))[:num_slices] # why x_dim - step_x -->  because idx_x,starts, from step_x,2*step_x,3*step_x...
        print(f"size of x,y,z slices Total:{len(slice_index_x)} ,{len(slice_index_y)},{len(slice_index_z)} \n")

        for idx, (x_index, y_index, z_index) in enumerate(zip(slice_index_x, slice_index_y, slice_index_z)):
            # Extract slices
            print(f"idx: {idx}, x_index: {x_index}, y_index: {y_index}, z_index: {z_index}")
            sagittal_slice = volume[x_index, :, :]
            coronal_slice = volume[:, y_index, :]
            axial_slice = volume[:, :, z_index]

            # Create figure
            fig, ax = plt.subplots(2, 2, figsize=figsize)

            ax[0, 0].imshow(data[:, :, z_index], cmap='gray')
            ax[0, 0].set_title(f"Original Slice@ z={z_index} Q={QunantileThresV:.2f}", fontsize=9)
            ax[0, 1].imshow(axial_slice.T, cmap="gray", origin="lower")
            ax[0, 1].set_title(f"Axial View Qth:{QunantileThresV:.2f}", fontsize=9)
            ax[1, 0].imshow(coronal_slice.T, cmap="gray", origin="lower")
            ax[1, 0].set_title(f"Coronal View Qth:{QunantileThresV:.2f}", fontsize=9)
            ax[1, 1].imshow(sagittal_slice.T, cmap="gray", origin="lower")
            ax[1, 1].set_title(f"Sagittal View Qth:{QunantileThresV:.2f}", fontsize=9)

            for ax in ax.flat:
                ax.axis("off")

            fig.suptitle(f"{filename} QunatilePercentile:{self.QuantilePercentile:.2f} (Slice {idx+1})", fontsize=10)
            plt.tight_layout(pad=1.0)

            # Save image before closing
            if save_plot:
                save_dirxyzSlice = self.save_res_plot / "saveSlicexyzFixQ"
                os.makedirs(save_dirxyzSlice, exist_ok=True)
                save_path = os.path.join(save_dirxyzSlice, f"{filename}_slice{z_index}_numslice{idx+1}allviews.png")
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

            plt.show(block=False)
            durationplotshow = durationplotshow if durationplotshow is not None else 3
            plt.pause(durationplotshow)
            plt.close(fig)

            # return save_path

   




    # def process(self, visualize=True, save_features=True):
    #     for filename in self.filewithpath:
    #         data = self.load_data(filename)
    #         mean_val, std_dev, QunatilethresValue = self.extract_features(data)
    #         self.features[filename] = {
    #             "Mean": mean_val,
    #             "Std Dev": std_dev,
    #             f"Q{self.QunatPercentile}": QunatilethresValue,            
    #         }
    #         print(f"{filename}: Mean={mean_val:.4f}, Std Dev={std_dev:.4f}, Q{self.QunatPercentile}={QunatilethresValue:.4f}")

    #         if visualize:
    #             self.visualize_thresholds(filename, data, QunatilethresValue, save_plot=True)

    #     if save_features and self.save_dir:
    #         os.makedirs(self.save_dir, exist_ok=True)
    #         save_path = os.path.join(self.save_dir, "ThresQuntFeature.txt")
    #         with open(save_path, 'w') as f:
    #             f.write(str(self.features))
    #         print(f"Features saved to: {save_path}")




class StaticsData(FeatureQuantileThresholding):

    def __init__(self, relative_datadir, save_dir=None, BASE_DIR=None, keyword=None):
        super().__init__(relative_datadir, save_dir, BASE_DIR, keyword)
        
    def stat_extract_features(self, data, dataset_name, save_features=True):
        """
            Extracts extended statistical features including min and max,
            combines with parent features and saves all to a unified file.
        """
    # Inherit base features
        base_features = super().extract_features(data)
        max_val = np.max(data)
        min_val = np.min(data)

        # Combine all features in a dictionary
        feature_dict = {
            "Mean": base_features[0],
            "Std Dev": base_features[1],
            "Q95": base_features[2],
            "Q99": base_features[3],
            "Min": min_val,
            "Max": max_val
        }

        # Save to a single file
        if save_features and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"AllFeatures_Stats{self.keyword[:-4]}.csv")
            with open(save_path, 'a') as f:  # append mode to collect all datasets
                f.write(f"{dataset_name}: {feature_dict}\n")
            print(f"Saved features for {dataset_name} to: {save_path}")

        return feature_dict



if __name__ =="__main__":
    import json
    
    relative_datadir = r'data/raw_npyData'

    ft = FeatureQuantileThresholding(relative_datadir=relative_datadir)

    pathjsonbgpFile = ft.BASE_DIR/"data"/"processed"
    pathjsonbgpFile = Path(pathjsonbgpFile)
    filename_json = 'QunatileP_ofbg_allfiles.json'

    with open(os.path.join(pathjsonbgpFile,filename_json),'r') as fr:
        QuantileP_EachFile = json.load(fr)
    
       
    
    for filewithpath in ft.filewithpath:
        filewithpath = Path(filewithpath)
        filename  = filewithpath.name
        # print(f"old Qunatile value: .99 --> {ft.QuantilePercentile}")
        old_QunatileP = ft.QuantilePercentile

        ft.QuantilePercentile = QuantileP_EachFile[filename[:-4]]/100
       
        print(f"filename : which is current in process:{filename} and \n old | new QunatileP :{old_QunatileP, ft.QuantilePercentile}")
        data  = ft.load_data(filewithpath)
        
        mean_val, std_dev, QuantileThres = ft.extract_features(data)

        QunatilethresValue = QuantileThres
        # ft.visualize_thresholds(filename,data,QunatileThres=QunatilethresValue,save_plot=True)
        # ft.visualize_thresholds_slice_dynamic(filename,data,QunatileThres = QunatilethresValue,save_plot=True,num_slices=3)
        
        ft.visualize_all_views(data, filename, QunantileThresV=QunatilethresValue,num_slices=7)
    


    # relative_datadir = r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\data\raw_npyData"
    # save_dir = r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\results\featureQuantileThres"
    # cwd = Path.cwd().parent.parent
    # BASE_DIR = cwd / "results"

    # print(f"----------< {BASE_DIR}")

    # stat_data = StaticsData(relative_datadir, save_dir, BASE_DIR=BASE_DIR, keyword='.npy')

    # for dataset_name in stat_data.filewithpath:
    #     data = stat_data.load_data(dataset_name)
    #     features = stat_data.stat_extract_features(data, dataset_name)
    #     print(f"Features for {dataset_name}: {features}")

