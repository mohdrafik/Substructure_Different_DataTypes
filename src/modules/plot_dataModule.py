import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from matplotlib.patches import Patch

import open3d as o3d
import imageio.v2 as imageio
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter

import open3d as o3d
import imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# from path_manager import addpath
# addpath()  # custom module to add paths this is not needed in this case because this module is in the same directory as the plot_dataModule.py script

from functools import wraps

def decoratorLog(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        print(f"\n ---------------------> /// Implementing method: {func.__name__} \\\ <------------------------------------------------------- \n")
        results = func(*args, **kwargs)
        # print(f"\n ---------------------> /// Finished executing method: {func.__name__} \\\ <--------------------------------------------------\n")
        return results
    return wrapper


from listspecificfiles import readlistFiles  # custom module to list files

class DataPlotter:
    """
    Class to visualize .npy data and significant digit distributions.

    Args:
        data_dir (str or Path): Path to directory with .npy files.
        base_dir (str or Path, optional): Root directory for results. Auto-detected from data_dir if None.
        save_results (bool): Whether to save plots to disk. Default is True.
        save_dir (str or Path, optional): Custom save directory. Defaults to 'results/rawData_XYPlot' inside base_dir.

    Methods:
        plot_simple(file, save_name=None): Basic scatter plot.
        plot_complex(file, save_name=None): Colored scatter plot with min/mean/max lines.
        plot_histogram_with_annotate_counts(counts, values, ...): Colored 1-height bar plot with counts annotated.
        plot_horizontal_split_histogram(sig_digit_data, ...): Horizontal bar plot split by frequency category.
        run_all(complex_plot=False): Process and plot all files.
    """

    def __init__(self, data_dir, base_dir=None, save_results=True, save_dir=None):
        self.data_dir = Path(data_dir)
        self.base_dir = Path(base_dir) if base_dir else self.data_dir.parent.parent
        self.save_results = save_results

        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = self.base_dir / "results" / "rawData_XYPlot"

        if self.save_results:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.files = readlistFiles(data_path=self.data_dir.relative_to(self.base_dir), keyword=".npy")

    def _generate_save_path(self, filename_stem, suffix=".png"):
        return self.save_dir / f"{filename_stem}{suffix}"

#################################################################################################################
    @staticmethod
    @decoratorLog
    def create_mesh_visualize_from_volumeData(volumeData, grid_factor, simplify=True, threshold ='percentile', percentileValue= None,  color_mode='gradient'):
        volume = volumeData
        flat = volume[volume > 0].flatten()
        if threshold == 'percentile':
            if percentileValue: 
                threshold = np.percentile(flat, percentileValue)  
            else:
                threshold = np.percentile(flat, 50)  # or any fixed value like threshold = 1.0

            print(f" Percentile Based Threshold: {threshold:.4f}")
        elif threshold == 'otsu':
            threshold = threshold_otsu(flat)
            print(f" Otsu Threshold: {threshold:.4f}")
        else:
            threshold = 1.334
            print(f" Manual Threshold: {threshold:.4f}")


        print("[INFO] Extracting mesh...")
        verts, faces, _, _ = marching_cubes(volume, level=threshold)
        print(f"[INFO] Mesh before simplification: {len(verts)} vertices, {len(faces)} faces")

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        if simplify:
            voxel_size = max(volume.shape) / grid_factor
            mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size)
            print(f"[INFO] Mesh after simplification: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

        if color_mode == 'gradient':
            z_vals = np.asarray(mesh.vertices)[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            norm_z = (z_vals - z_min) / (z_max - z_min + 1e-8)
            colors = np.stack([norm_z, 0.6 * np.ones_like(norm_z), 1.0 - norm_z], axis=1)
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        else:
            mesh.paint_uniform_color([0.6, 0.7, 1.0])

        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


        return mesh
    
    @staticmethod
    @decoratorLog
    def save_mesh_views_as_gif_and_png(mesh, save_dir, savefilenamepng , savefilenamegif):
        """
        Save the mesh visualization in isometric form as PNG and animated GIF.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(mesh)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        vis.poll_events()
        vis.update_renderer()

        # Isometric angles to simulate rotation
        angles = list(range(0, 360, 10))
        images = []

        for angle in angles:
            ctr.rotate(10.0, 0.0)  # Horizontal rotation
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            image_np = (np.asarray(image) * 255).astype(np.uint8)
            images.append(image_np)

            if angle == 0:
                imageio.imwrite(os.path.join(save_dir, savefilenamepng), image_np)

        gif_path = os.path.join(save_dir, savefilenamegif)
        print(f"Saving GIF and png to {gif_path}")
        imageio.mimsave(gif_path, images, fps=10)
        vis.destroy_window()

        return gif_path

        
    # @staticmethod
    # def visualize_mesh(mesh):
    #     o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

#################################################################################################################
    def plot_simple(self, file, save_name=None):
        data = np.load(file).flatten().reshape(-1, 1)
        x_data = np.linspace(data.min(), data.max(), len(data))

        plt.plot(x_data, data, '.', color='blue', markersize=0.4)
        plt.title(f"{file.stem} Linearly Spaced Vector")
        plt.xlabel('linearly spaced vector from data itself')
        plt.ylabel('original Values')
        plt.grid(True)

        if self.save_results:
            save_path = self._generate_save_path(save_name or file.stem)
            plt.savefig(save_path, dpi=300)

        plt.close()

#################################################################################################################
    def plot_complex(self, file, save_name=None):
        data = np.load(file).flatten()
        x_data = np.linspace(data.min(), data.max(), len(data))

        hist_vals, bin_edges = np.histogram(data, bins=10000)
        dominant_bin_index = np.argmax(hist_vals)
        bin_start, bin_end = bin_edges[dominant_bin_index], bin_edges[dominant_bin_index + 1]
        flat_band = data[(data >= bin_start) & (data < bin_end)]

        if flat_band.size == 0:
            print(f"No flat band detected in file: {file.name}")
            return

        flat_min, flat_max, flat_mean = flat_band.min(), flat_band.max(), flat_band.mean()

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(x_data, data, c=data, cmap='viridis', s=0.4)

        flat_y_vals = [("Min", flat_min, 'red'), ("Mean", flat_mean, 'blue'), ("Max", flat_max, 'black')]
        used_y = []
        x_text_position = x_data[0] - 0.0002*(x_data[-1] - x_data[0])  # shift right
    
        if file[-5:-4] =='h':
            text_gap = 0.12* max(abs(flat_max - flat_min), 1e-1)  # vertical spacing
        else:
            text_gap = 0.05* max(abs(flat_max - flat_min), 1e-1)  # vertical spacing
            

        for i, (label, y_val, color) in enumerate(flat_y_vals):
            y_text = y_val + i * text_gap  # stagger vertically (fixed gap)
            plt.axhline(y=y_val, color=color, linestyle='--', linewidth=0.6)

            plt.text(x_text_position, y_text, f'{label}: {y_val:.16f}',
                    color=color, fontsize=8,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        

        # for label, y_val, color in flat_y_vals:
        #     plt.axhline(y=y_val, color=color, linestyle='--', linewidth=1)
        #     offset = 0
        #     while any(abs((y_val + offset) - y) < 0.002 * (flat_max - flat_min) for y in used_y):
        #         offset += 0.002 * (flat_max - flat_min)
        #     y_text = y_val + offset
        #     used_y.append(y_text)
        #     plt.text(x_data[0], y_text, f'{label}: {y_val:.8f}', color=color, fontsize=8, verticalalignment='bottom')

        plt.title(f"{file.stem}")
        plt.xlabel('Linearly spaced vector from data')
        plt.ylabel(f" RI value ")
        plt.grid(True)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Intensity')

        if self.save_results:
            save_path = self._generate_save_path(save_name or file.stem + '_marked')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved at: {save_path}")

        plt.show()
        plt.close()


#################################################################################################################
    @staticmethod
    def plot_histogram_with_annotate_counts(counts, values, title=None, xlabel=None, ylabel=None, saveplot=False, filename=None, save_dir=None, dpi=600):
        
        """
        Plot a clean, publication-ready histogram with counts mapped to color and bold annotations.
        Necessity:
        Data counts are differ in Counts from very high to very low. So I make all counts to 1 coorresponding to each unique value and annotate the original counts on top of each bar.
        This is useful for visualizing the distribution of significant digits in the data.
        Plot histogram with original counts annotated on top of each bar.

        Args:
            Counts_OFUniqueValues (list or array): Count values corresponding to unique values.
            UniqueValues (list or array): List of unique values (e.g., significant digits).
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            saveplot (bool): If True, saves the figure.
            filename (str): Base filename (no extension).
            save_dir (str): Directory to save the plot.
            dpi (int): Dots per inch for saving resolution.
        """
        x = np.array(values)
        counts = np.array(counts)
        y = np.ones_like(counts)

        norm = plt.Normalize(counts.min(), counts.max())
        cmap = plt.cm.viridis
        colors = cmap(norm(counts))

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(x, y, width=0.9, color=colors, edgecolor='black', linewidth=1.2)

        for xi, count in zip(x, counts):
            ax.text(xi, 1.008, f"{count}", ha='center', va='bottom', fontsize=4, fontweight='bold')

        ax.set_xlabel(xlabel if xlabel else "Significant Digits", fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel if ylabel else "Normalized Count (1 per bin)", fontsize=11, fontweight='bold')
        ax.set_title(title if title else "Histogram with Annotated Counts", fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_yticks([])
        ax.tick_params(axis='x', length=5, width=1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Original Count Value", fontsize=11, fontweight='bold')

        plt.tight_layout()

        if saveplot:
            if filename and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{filename}_histogram.png")
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                print(f" Plot saved at: {save_path}")
            else:
                print(" Cannot save: filename and save_dir must be specified.")
        else:
            print("saveplot is False; plot will not be saved.")

        plt.show()

#################################################################################################################
    @staticmethod
    def plot_horizontal_split_histogram(significant_digit_data, title=None, xlabel=None, ylabel=None, saveplot=False, filename=None, save_dir=None, dpi=600):
        """
        here significant_digit_data is a list of significant digits in my data values, this is counted and plot as horizontal split histogram 
        Plot a horizontal histogram with bars color-coded by frequency level:
        very low (highlighted), low, moderate, and high.

        Args:
            significant_digit_data (list): Input digit list.
            title (str): Plot title. If None, default will be used.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            saveplot (bool): Whether to save the plot.
            filename (str): Filename without extension.
            save_dir (str): Directory to save the plot.
            dpi (int): Resolution for saved plot.
        """

        counter = Counter(significant_digit_data)
        digits = np.array(sorted(counter.keys()))
        counts = np.array([counter[d] for d in digits])

        very_low_thresh = np.percentile(counts, 10)
        low_thresh = np.percentile(counts, 33)
        high_thresh = np.percentile(counts, 66)

        colors = []
        for c in counts:
            if c <= very_low_thresh:
                colors.append("darkred")
            elif c < low_thresh:
                colors.append("skyblue")
            elif c < high_thresh:
                colors.append("gold")
            else:
                colors.append("salmon")

        sns.set(style="whitegrid", font_scale=1.1, rc={
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'font.family': 'serif'
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(digits, counts, color=colors, edgecolor='black', height=0.7)

        for d, c in zip(digits, counts):
            ax.text(c + max(counts)*0.01, d, f"{c}", va='center', fontsize=9)

        ax.set_title(title if title else "Horizontal Histogram", pad=10)
        ax.set_xlabel(xlabel if xlabel else "Count", labelpad=6)
        ax.set_ylabel(ylabel if ylabel else "Significant Digit", labelpad=6)
        ax.set_yticks(digits)

        legend_elements = [
            Patch(facecolor='darkred', edgecolor='black', label='Very Low'),
            Patch(facecolor='skyblue', edgecolor='black', label='Low'),
            Patch(facecolor='gold', edgecolor='black', label='Moderate'),
            Patch(facecolor='salmon', edgecolor='black', label='High')
        ]
        ax.legend(handles=legend_elements, title="Frequency Level", loc='best')

        plt.tight_layout()

        if saveplot:
            if filename and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{filename}_horizontal_histogram.png")
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                print(f" Plot saved at: {save_path}")
            else:
                print(" Cannot save: provide both filename and save_dir.")
        else:
            print("Plot not saved (saveplot=False).")

        plt.show()

    def run_all(self, complex_plot=False):
        print(f"Found {len(self.files)} .npy files.")
        t_start = time.time()

        for file in self.files:
            if complex_plot:
                self.plot_complex(file)
            else:
                self.plot_simple(file)

        print(f"All plots completed in {time.time() - t_start:.2f} seconds.")

################################################################################################################# does not working  on cnr computer.
    @staticmethod
    def visualize_and_export_3d_mesh(fg_mask, data, smoothing=None, title=None,
                                     save_obj_path=None,
                                     save_png_path=None,
                                     save_gif_path=None,
                                     rotate_and_capture=False,
                                     gif_frames=12):
        """
        Visualize a 3D mesh from a 3D numpy array using Open3D. 
        Optionally save the mesh as an OBJ file and capture images or GIFs.
        Args:
            fg_mask (numpy.ndarray): Foreground mask (3D numpy array).
            data (numpy.ndarray): 3D numpy array to visualize.                      
            smoothing (float, optional): Smoothing factor for Gaussian filter. smoothing= 0.5,1.0 etc
            title (str, optional): Title for the visualization window.
            save_obj_path (str, optional): Path to save the OBJ file.
            save_png_path (str, optional): Path to save the PNG image.
            save_gif_path (str, optional): Path to save the GIF.
            rotate_and_capture (bool, optional): Whether to rotate the view and capture frames for GIF.
            gif_frames (int, optional): Number of frames for the GIF.

        """


        if not np.any(fg_mask):
            print(" Foreground mask is empty. Skipping visualization.")
            return

        if smoothing is not None:
            data = gaussian_filter(data * fg_mask, sigma=smoothing)

        else:
            data = data * fg_mask

        verts, faces, _, _ = marching_cubes(data, level=0)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        if smoothing is not None:
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
        
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.9, 1.0])

        fg_points = np.argwhere(fg_mask)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fg_points)
        pcd.paint_uniform_color([1.0, 0.7, 0.3])

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1024, height=768, visible=True)
        vis.add_geometry(mesh)
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.mesh_show_back_face = True
        opt.point_size = 2.5

        vis.poll_events()
        vis.update_renderer()
        ctr = vis.get_view_control()

        if save_png_path:
            vis.capture_screen_image(save_png_path)
            print(f"PNG saved: {save_png_path}")

        if rotate_and_capture and save_gif_path:
            images = []
            tmp_paths = []
            for i in range(gif_frames):
                ctr.rotate(10.0, 0.0)
                vis.poll_events()
                vis.update_renderer()
                tmp_path = f"_tmp_frame_{i:03d}.png"
                vis.capture_screen_image(tmp_path)
                images.append(imageio.imread(tmp_path))
                tmp_paths.append(tmp_path)

            imageio.mimsave(save_gif_path, images, duration=0.1)
            print(f" GIF saved: {save_gif_path}")

            for p in tmp_paths:
                if os.path.exists(p):
                    os.remove(p)

        if save_obj_path:
            o3d.io.write_triangle_mesh(save_obj_path, mesh)
            print(f" OBJ saved: {save_obj_path}")

        # these two lines require manuual closing of the window
        vis.run()  #  This keeps the window open until you close it manually
        vis.destroy_window()

        # Instead, we can use a loop to keep the window open for a short time
        # and then close it automatically
        # vis.poll_events()
        # vis.update_renderer()
        # time.sleep(10)  # Keep the window open for 2 seconds
        # vis.destroy_window()

#################################################################################################################
    # @staticmethod
    # def QunatileBasedThrseholdingVisualization():



if __name__ == "__main__":

    from plot_dataModule import DataPlotter
    # Example data usage
    plotter = DataPlotter(
        # __init__(self, data_dir, base_dir=None, save_results=True, save_dir=None)
        data_dir="data/raw_npyData",  # relative to BASE_DIR
        base_dir="E:/Projects/substructure_3d_data/Substructure_Different_DataTypes"
    )
    plotter.run_all(complex_plot=True)


    # Example data usage
    from plot_dataModule import DataPlotter
    counts = [1, 10, 50, 200, 800, 3000]
    values = [1, 2, 3, 4, 5, 6]
    significant_digits = [1, 2, 2, 3, 3, 3, 4, 5, 6]

    # Call directly on the class (no instance required)
    DataPlotter.plot_histogram_with_annotate_counts(
        counts, values,
        title="Annotated Histogram",
        saveplot=False
    )

    DataPlotter.plot_horizontal_split_histogram(
        significant_digits,
        title="Horizontal Split Histogram",
        saveplot=False
    )

# counts = np.array([      1,      20,     253,    2466,   25466,  257152, 2461352,  5670279, 1041250,   97875,   10690,     838])
# values = [val for val in range(1, 13)]
# significant_digit_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
