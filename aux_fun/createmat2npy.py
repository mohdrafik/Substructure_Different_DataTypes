import scipy.io as sio
import numpy as np
import os
# from basicstatics import basicstat

import numpy as np

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

def mat2npy(datapath,outputdatapath):

    # files = os.listdir(r'C:\Users\mrafik\OneDrive - C.N.R. STIIMA\tomogram all data\all_tomogram_data')
    files = os.listdir(datapath)
    print(f" list of all files: {files}")
    if not (os.path.exists(outputdatapath)):
        os.makedirs(outputdatapath)
        print(f"Directory '{outputdatapath}' created.")
    else:
        print("output directory already exist \n")

    for matfile in files:
        if matfile[-4:]=='.mat':
            print(matfile)
            # Load .mat file
            matdata = os.path.join(datapath,matfile)
            mat = sio.loadmat(matdata)
            # Extract the 3D matrix
            tomogram_name = matfile[0:-4]
            # print(tomogram_name)
            
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
    print(f"process completed,.mat2 .npy is done and saved at:  {outputdatapath}  ]")                


if __name__=="__main__":

    datapath = input("enter the data path:(press enter for use default path):").strip()
    if not datapath:
        datapath = r'C:\Users\mrafik\OneDrive - C.N.R. STIIMA\tomogram all data\all_tomogram_data'
    
    # outputdatapath = input("enter the output path:")
    
    outputdatapath = input("Enter the output path (Press Enter to use default path): ").strip()  #  # Prompt user for output path
    # strip() Cleans Input: This removes any accidental spaces before checking for emptiness.
    # If user just pressed Enter, set default path
    if not outputdatapath:
        outputdatapath = r'E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\data\intermdata1' 
        

    # Ensure the directory exists; create it if it doesn't
    os.makedirs(outputdatapath, exist_ok=True)

    print(f"Output directory set to: {outputdatapath}")

    # Call function with the paths
    mat2npy(datapath=datapath, outputdatapath=outputdatapath)
