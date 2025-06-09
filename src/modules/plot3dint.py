import numpy as np
import plotly.graph_objects as go

from functools import wraps
# @staticmethod  # on top if we are defining do't need @staticmethod.
def logfunction(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n ---------------------> /// Implementing method: {func.__name__} \\\ <------------------------------------------------------- \n")
        results = func(*args, **kwargs)
        # print(f"\n ---------------------> /// Finished executing method: {func.__name__} \\\ <--------------------------------------------------\n")
        return results
    # print(f"\n ---------------------> /// Finished executing method:  \\\ <--------------------------------------------------\n")
    return wrapper


@logfunction
def plot3dinteractive(voldata, keyvalue, sample_fraction):

    """Interactive 3D scatter plot for large 3D NumPy arrays.
    
    - `voldata`: Input 3D NumPy array.
    - `keyvalue`: Label for the figure.
    - `sample_fraction`: Fraction of points to randomly plot (Default: 1% of the points) sample_fraction = 0.01 .
    """
    array_3d = voldata
    x1, y1, z1 = array_3d.shape
    print(f"Shape of input 3D NumPy array: {x1, y1, z1}")

    # Create a 3D meshgrid
    x, y, z = np.meshgrid(np.arange(x1), np.arange(y1), np.arange(z1))

    # Filter out zero values
    mask = array_3d > 0
    x_vals = x[mask].flatten()
    y_vals = y[mask].flatten()
    z_vals = z[mask].flatten()
    values = array_3d[mask].flatten()

    # **Randomly sample** points to reduce memory usage
    num_points = len(values)
    sample_size = int(num_points * sample_fraction)
    
    if sample_size > 0:
        indices = np.random.choice(num_points, sample_size, replace=False)
        x_vals = x_vals[indices]
        y_vals = y_vals[indices]
        z_vals = z_vals[indices]
        values = values[indices]
    else:
        print("⚠ Warning: Not enough non-zero points for plotting.")
        return

    print(f"Plotting {sample_size} points out of {num_points} ({sample_fraction * 100}% sampled)")
    print(f"Generating figure for: {keyvalue}")

    # Create a 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=1,
            color=values,
            colorscale='Viridis',
            opacity=0.5
        )
    ))

    # Set axis labels
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    # Show the interactive plot
    fig.show()

if __name__ =="__main__":
    import os
    from pathlib import Path
  
    cws = Path(__file__).resolve().parent   # use it everywhere for current working scripts path (cws)
    BASE_DIR = cws.parent.parent
    DATA_PATH = BASE_DIR/"data" 
    rawnypyData_Path = DATA_PATH/"raw_npyData"

    data_dir = str(rawnypyData_Path)
    # print(f"here we will see step by step all mid paths --> \n cws :{cws} \n BASE_DIR:{BASE_DIR} \n DATA_PATH: {DATA_PATH} \n rawnypyData_Path:{rawnypyData_Path} \n ")

    # List all .npy files
    npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

    # Iterate through all .npy files and plot
    for filename in npy_files:
        file_path = os.path.join(data_dir, filename)
        voldata = np.load(file_path)  # Load .npy file
        plot3dinteractive(voldata, filename, sample_fraction=0.02)  # Only 0.5% of points




#  already imported numpy as np and plotly.graph_object as go. 
# we already have 3D numpy array with shape (201, 201, 201)
# def plot3dinteractive(voldata,keyvalue):
#     """ vodata should be numpy 3d array and keyvalue and writing the name just for knowing.   """
#     array_3d = voldata
#     # Create a 3D meshgrid
#     x1,y1,z1 = array_3d.shape
#     print(f"shape of th einput 3d numpy array: {x1, y1, z1}")
#     # x, y, z = np.meshgrid(np.arange(201), np.arange(201), np.arange(201))
#     x, y, z = np.meshgrid(np.arange(x1), np.arange(y1), np.arange(z1))

#     # Get the values and coordinates for the points with values greater than 0
#     x_vals = x[array_3d > 0].flatten()
#     y_vals = y[array_3d >0].flatten()
#     z_vals = z[array_3d > 0].flatten()
#     values = array_3d[array_3d > 0].flatten()
#     # custom_colormap = np.where(array_3d > 0, 'red', 'blue')
#     # Create a 3D scatter plot
#     print(f"figure for the :{keyvalue}")
#     fig = go.Figure(data=go.Scatter3d(
#         x=x_vals,
#         y=y_vals,
#         z=z_vals,
#         mode='markers',
#         marker=dict(
#             size=1,
#             color=values,
#             colorscale='Viridis',
#             opacity=0.5
            
            
#         )
#     ))

#     # Set labels for each axis
#     fig.update_layout(scene=dict(
#         xaxis_title='X',
#         yaxis_title='Y',
#         zaxis_title='Z'
#     ))

#     # Show the interactive plot
#     fig.show()
