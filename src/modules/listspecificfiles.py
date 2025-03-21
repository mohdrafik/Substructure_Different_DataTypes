import os
import glob


def readlistFiles(filepath,keyword):
    """ Read the list of specific files based on the key word given by users:
        Enter the filepath and keyword (e.g. "normalized.npy", ...)
    """
    # keyword = "*keyword";
    retSpecificFiles = []
    Files = glob.glob(os.path.join(filepath,f"*{keyword}"))
    for file in Files:
        Filename = os.path.basename(file)
    
        retSpecificFiles.append(Filename)
        # os.path.normpath

    return retSpecificFiles;

if __name__=="__main__":
    
    # Keyword = 'normalized.npy'
    Keyword = input("enter keyword:")
    # if not Keyword:
    # filepath = input("enter Filepath:")    
    filepath = r'E:\\Projects\\substructure_3d_data\\Substructure_Different_DataTypes\\data\\intermdata1'
    print("output res: --> \n",readlistFiles(filepath=filepath,keyword=Keyword))


# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent  # Gets the folder containing this script
# print("Base Directory:", BASE_DIR)
