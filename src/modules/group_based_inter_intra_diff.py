import numpy as np
from pathlib import Path
import os
import sys
sys.path.append(r'E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\src')
from path_manager import addpath
addpath()

from listspecificfiles import readlistFiles

class GroupingBasedDistance:
    def __init__(self, rel_datapath, data_path = None, output = None, file_suffix = None):
        self.data_path = data_path
        self.rel_datapath = rel_datapath
        self.output = output
        self.file_suffix = file_suffix if file_suffix is not None else '.npy'

        self.fpath = readlistFiles(self.rel_datapath,self.file_suffix).file_with_Path()


    def load_file(self, file_withpath):
        data = np.load(file_withpath) 
        return data
    
    def find_diff(self, data):
        data = data[data > 0]
        data = data.flatten()
        data_val_range = max(data) - min(data)
        sorted_data = np.sort(data)
        df1 = np.diff(sorted_data)  # first difference between va;ues of teh sotrted data array
        print(f"max of 1st diff array : (<{data_val_range}): {max(df1):.5f}")
        print(f"min of 1st diff array : ({data_val_range}): {min(df1):.8e}")
        return df1


if __name__== "__main__":
    
    # from pathlib import Path
    # from group_based_inter_intra_diff import GroupingBasedDistance
    # import os
    relative_datapath = r"data\processed\main_fgdata"
    gbd = GroupingBasedDistance(rel_datapath=relative_datapath)

    for fwpath in gbd.fpath:
        filewpath = Path(fwpath)
        filename = filewpath.name
        fname = filewpath.stem

        data = gbd.load_file(file_withpath=filewpath)
        print(f"\n I am processing with filename: {filename}")
        gbd.find_diff(data)

        