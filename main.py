
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import sys
plt.style.use("ggplot")

# ----------- all wriiten module imports after adding the path -----------

BASE_DIR = Path.cwd()
os.sys.path.append(str(BASE_DIR / "src"))  # Add the src directory to the Python path
from path_manager import addpath
addpath()


# ----------- here we define the results directory data directory and save directory -----------
RES_DIR = BASE_DIR / "results"
base_dir = BASE_DIR
data_dir = BASE_DIR / "data" / "processed" / "main_fgdata"  # CHNAGE WHEN DIFFERENT FILES NEEDED ! -- Path to the directory containing .npy files
data_dir = Path(data_dir)  # Convert to Path object for consistency

save_dir = RES_DIR / "procesed_fgData_XYPlot"
save_dir = Path(save_dir)  # Convert to Path object for consistency
print(f"save_dir: {save_dir}")  # Print the relative path for clarity

from plot_dataModule import DataPlotter
from pathlib import Path 
from listspecificfiles import readlistFiles



save_dir.mkdir(parents = True, exist_ok=True)  # Create if missing

plotter = DataPlotter(data_dir=data_dir, base_dir=base_dir, save_results=True, save_dir=save_dir) 
# because self.files is already defined in the DataPlotter class, constructor.
# it will read all files from the data_dir and save them in self.files. can call  it using:  plotter.files
# Ensure plotter.files is iterable (list of files)
# files = plotter.files if isinstance(plotter.files, (list, tuple)) else [plotter.files]

for file in plotter.files:
    
    filename = file.split('.')[0]  # Get the filename without extension
    print(f"file name with extension: {file} \n filename without extension : {filename}\n")

    # DataPlotter(data_dir = data_dir, base_dir = base_dir, save_results=True, save_dir = save_dir).plot_complex(file = fileWithpath, save_name = filename)

    # plotter.plot_simple(file = file)
    plotter.plot_complex(file = file, withMarkers=False, save_name = None)
print("< --------------------------------------------------  All plots generated successfully. -------------------------------------------------- >\n \n")

