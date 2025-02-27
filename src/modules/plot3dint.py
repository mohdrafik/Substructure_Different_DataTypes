import numpy as np
import plotly.graph_objects as go
#  already imported numpy as np and plotly.graph_object as go. 
# we already have 3D numpy array with shape (201, 201, 201)
def plot3dinteractive(voldata):
    """ vodata should be numpy 3d array     """
    array_3d = voldata
    # Create a 3D meshgrid
    x, y, z = np.meshgrid(np.arange(201), np.arange(201), np.arange(201))

    # Get the values and coordinates for the points with values greater than 0
    x_vals = x[array_3d > 0].flatten()
    y_vals = y[array_3d >0].flatten()
    z_vals = z[array_3d > 0].flatten()
    values = array_3d[array_3d > 0].flatten()
    # custom_colormap = np.where(array_3d > 0, 'red', 'blue')
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

    # Set labels for each axis
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    # Show the interactive plot
    fig.show()