import os
import glob
from pathlib import Path
import os

class readlistFiles:
    """
    want full filePath with name use -->  fpath = readlistFiles(Relative_data_path,'.npy').file_with_Path() 
    want only filename in directory -->  # files = readlistFiles(Relative_data_path,'.npy')

    Returns a list of filenames in the given filepath that end with the specified keyword.

    Args:
        filepath (str): Path to the relative filepath "data\normalized_npydata\" to search in.
        keyword (str): keyword to filter files by (e.g., 'normalized.npy').

    Returns:
        list: Filenames ending with the given keyword.
    """

    cws = Path(__file__).resolve().parent.parent   # use it everywhere for current working scripts path (cws)
    # cws = Path.cwd()
    BASE_DIR = cws.parent        # \Projects\substructure_3d_data\Substructure_Different_DataTypes\
    # DATA_PATH = BASE_DIR/"data"  #  Projects\substructure_3d_data\Substructure_Different_DataTypes\data\
    
    def __init__(self,data_path,keyword):
        self.data_path = self.BASE_DIR/data_path
        self.keyword = keyword

        self.files_matched = self.matchedFiles()  # Why use self.files_matched_e_Path = self.matchedFiles() instead of just files_matched_e_Path = self.matchedFiles() inside __init__()? or we can write: files_matched_e_Path = self.matchedFiles()
        #  --> but this variable will be local to the __init__() method only. ➡️ So: it won’t be accessible outside the __init__() method ; ➡️ You will lose access to it once the constructor finishes running.      
        # self.fandPath = self.file_with_Path()
        
    def matchedFiles(self):
        matched_Files = []
        Files = os.listdir(self.data_path)
        for file in Files:
            if file.endswith(self.keyword):
                matched_Files.append(file)
                
        return matched_Files
    
    def file_with_Path(self):
        self.fileFullPath = [self.data_path/f for f in self.files_matched]
        return self.fileFullPath

    def __iter__(self):
        return  iter(self.files_matched)
        
    def __getitem__(self,index):
        return self.files_matched[index]
    
    def __len__(self):
        return len(self.files_matched)

    
    def __str__(self):
        return f"list of files:{self.matchedFiles()}"



if __name__ == "__main__":
    # datapath =r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\data\normalized_npyData" 
    datapath =r"data\raw_npyData" 
    datapath = os.path.normpath(datapath)
    print(datapath)
    # datapath = Normalized_data
    # k1 = 'Copy.txt'
    # k1 = 'NORMALIZED.NPY'
    k1 = '.npy'
    k1 = k1.lower()
    d1 = readlistFiles(datapath,k1)
    
    # d1.matchedFiles()
    print(f" {d1.files_matched} \n and \n base path: {d1.file_with_Path()}")

    # print(d1)
    import numpy as np
    import os
    count = 0
    # for file in d1:
    #     count +=1
    #     f1 = file
    #     if count ==1:
    #         break

    data = np.load(os.path.join(datapath,d1[3]))
    print(f"datashape: {d1[3]}  and {data.shape}")
