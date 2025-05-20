# createmat2npyViceVersa 
""" convert the matfile to numpy array file and numpy array to matfile  data """

import scipy.io as sio
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import json
# from basicstatics import basicstat
from scipy.io import savemat

# Load the .npy file
def basicstat(data):
    # data = np.load("Tomogramma_Cell1.npy")

    # Print basic statistics
    print("Shape of Data:", data.shape)
    print("Data Type:", data.dtype)
    print("Min Value:", np.min(data))
    print("Max Value:", np.max(data))
    print("Mean Value:", np.mean(data))
    print("Standard Deviation:", np.std(data))

def load_and_normalize_npy(npyfile_path):

    """
    Loads a .npy file and applies Min-Max normalization.
    """
    data = np.load(npyfile_path)  # Load the file
    min_val, max_val = np.min(data), np.max(data)  # Get min-max values
    normalized_data = (data - min_val) / (max_val - min_val)  # Normalize to [0,1]
    return normalized_data, min_val, max_val



def mat2npy(datapath,outputdatapath):
    """
    convert all the .mat file corresponding to the main key( 'Tom_RI') of ecah .mat file to the .npy (numpy array file) 
    and save in npy array with the same name as .mat  
    
    """

    # # files = os.listdir(r'C:\Users\mrafik\OneDrive - C.N.R. STIIMA\tomogram all data\all_tomogram_data')
    # datapath = input("enter the data path:(press enter for use default path):").strip()
    # if not datapath:
    #     datapath = r'C:\Users\mrafik\OneDrive - C.N.R. STIIMA\tomogram all data\all_tomogram_data'
    #     if not os.path.exists(datapath):
    #         datapath = 'C:\Users\Gaetano\Desktop\create_with_codeRafi\SharedContents\OneDrive - C.N.R. STIIMA\tomogram all data\all_tomogram_data'        

    
    # # outputdatapath = input("enter the output path:")
    
    # outputdatapath = input("Enter the output path (Press Enter to use default path): ").strip()  #  # Prompt user for output path
    # # strip() Cleans Input: This removes any accidental spaces before checking for emptiness.
    # # If user just pressed Enter, set default path
    # if not outputdatapath:
    #     outputdatapath = r'E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\data\intermdata1'
    #     if not os.path.exists(outputdatapath):
    #         outputdatapath = 'C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\data\intermdata1' 
        
    # # Ensure the directory exists; create it if it doesn't
    # os.makedirs(outputdatapath, exist_ok=True)



    files = os.listdir(datapath)
    print(f" list of only .mat files: {[ file for file in files if file.endswith('.mat')]}")
    if not (os.path.exists(outputdatapath)):
        os.makedirs(outputdatapath)
        print(f"Directory '{outputdatapath}' created.")
    else:
        print("output directory already exist \n")

    for matfile in files:
        if matfile[-4:]=='.mat' :
            print(f"just a matfile name is printed not proceesed yet, for checking name only :--> {matfile}")
            # Load .mat file
            matdata = os.path.join(datapath,matfile)
            mat = sio.loadmat(matdata)
            # Extract the 3D matrix
            tomogram_name = matfile[0:-4]
            # print(tomogram_name)
            fileexistCheckname = os.path.join(outputdatapath,f"{tomogram_name}.npy")
            if not os.path.exists(fileexistCheckname):
                for key in mat.keys():
                    # print(key)
                    if key in ['Tom_RI','Tom3']:
                        tomogram_data = mat[key]  # Extract the refractive index data
                    # Save as a .npy file
                        numpyarrfile = f"{tomogram_name}.npy"
                        numpyarrfilename = os.path.join(outputdatapath,numpyarrfile)
                        np.save(numpyarrfilename, tomogram_data)
                        # data = np.load(numpyarrfile)
                        # basicstat(data)
                        basicstat(tomogram_data)
            else:
                print(f"file already exist:{tomogram_name}.npy")     

    print(f"process completed,.mat2 .npy is done and saved at:  {outputdatapath}  ]")    

def npy2mat(datapath,outputdatapath):
    
    files = os.listdir(datapath)
    print(f" list of only .mat files: {[ file for file in files if file.endswith('.npy')]}")
    
    if not (os.path.exists(outputdatapath)):
        os.makedirs(outputdatapath)
        print(f"Directory '{outputdatapath}' created.")
    else:
        print("output directory already exist \n")


    for npyfile in files:
        if npyfile[-4:]=='.npy' :
            print(f"just a npyfile name is printed not proceesed yet, for checking name only :--> {npyfile}")
            # Load .npy file
            npydata = os.path.join(datapath,npyfile)
            npydata = np.load(npydata)
            # Extract the 3D matrix
            tomogram_name = npyfile[0:-4]
            # print(tomogram_name)
            fileexistCheckname = os.path.join(outputdatapath,f"{tomogram_name}.mat")
    

    # Load the .npy file
    # array_data = np.load(npy_file_path)
            if not os.path.exists(fileexistCheckname):
                    # Save as a .mat file
                    numpyarrfile = f"{tomogram_name}.mat"
                    numpyarrfilename = os.path.join(outputdatapath,numpyarrfile)
                    # savemat(outputdatapath, {"data": npydata})
                    savemat(numpyarrfilename, {"data": npydata})
                    # data = np.load(numpyarrfile)
                    # basicstat(data)
                    basicstat(npydata)
            else:
                print(f"file already exist:{tomogram_name}.mat")     

    print(f"process completed,.npy2.mat is done and saved at:  {outputdatapath}  ]")    
    # Save to .mat file (with a variable name)


if __name__ == "__main__":

    # datapath = input("enter the data path:(press enter for use default path):").strip()
    # if not datapath:
    #     datapath = r'C:\Users\mrafik\OneDrive - C.N.R. STIIMA\tomogram all data\all_tomogram_data'
    
    # # outputdatapath = input("enter the output path:")
    
    # outputdatapath = input("Enter the output path (Press Enter to use default path): ").strip()  #  # Prompt user for output path
    # # strip() Cleans Input: This removes any accidental spaces before checking for emptiness.
    # # If user just pressed Enter, set default path
    # if not outputdatapath:
    #     outputdatapath = r'E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\data\intermdata1' 
        

    # # Ensure the directory exists; create it if it doesn't
    # os.makedirs(outputdatapath, exist_ok=True)

    # print(f"Output directory set to: {outputdatapath}")

    # # Call function with the paths


    datapath = input("enter the data path:(press enter for use default path):").strip()
    if not datapath:
        datapath = r'C:\Users\mrafik\OneDrive - C.N.R. STIIMA\tomogram all data\all_tomogram_data'
        if not os.path.exists(datapath):
            datapath = r'C:\Users\Gaetano\Desktop\create_with_codeRafi\SharedContents\OneDrive - C.N.R. STIIMA\tomogram all data\all_tomogram_data'        


    # outputdatapath = input("enter the output path:")

    outputdatapath = input("Enter the output path (Press Enter to use default path): ").strip()  #  # Prompt user for output path
    # strip() Cleans Input: This removes any accidental spaces before checking for emptiness.
    # If user just pressed Enter, set default path
    if not outputdatapath:
        outputdatapath = r'E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\data\raw_npyData'
        if not os.path.exists(outputdatapath):
            outputdatapath = r'C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\data\raw_npyData' 
        
    # Ensure the directory exists; create it if it doesn't
    os.makedirs(outputdatapath, exist_ok=True)

    mat2npy(datapath=datapath, outputdatapath=outputdatapath)
    
    npy2mat(datapath=datapath, outputdatapath=outputdatapath)

