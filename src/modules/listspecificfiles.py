import os
import glob

class readlistFiles:
    """
    Returns a list of filenames in the given filepath that end with the specified keyword.

    Args:
        filepath (str): Path to the filepath to search in.
        keyword (str): keyword to filter files by (e.g., 'normalized.npy').

    Returns:
        list: Filenames ending with the given keyword.
    """
 
    
    def __init__(self,data_path,keyword):
        self.data_path = data_path
        self.keyword = keyword
        self.matched_Files = self.matchedFiles()
        # dataFiles_list = self.matchedFiles()
        
        
    def matchedFiles(self):
        matched_Files = []
        Files = os.listdir(self.data_path)
        for file in Files:
            if file.endswith(self.keyword):
                matched_Files.append(file)
        return matched_Files 



if __name__ == "__main__":
    datapath =r"E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\data\normalized_npyData" 
    datapath = os.path.normpath(datapath)
    print(datapath)
    # datapath = Normalized_data
    # k1 = 'Copy.txt'
    k1 = 'NORMALIZED.NPY'
    k1 = k1.lower()
    d1 = readlistFiles(datapath,k1)
    # d1.matchedFiles()
    print(d1.matched_Files)
    # print(d1)




# # Example usage (can be removed or commented in notebook):
# if __name__ == "__main__":
#     dir_path = input("📁 Enter filepath path: ").strip()
#     keyword = input("🔍 Enter file keyword to filter (e.g., normalized.npy): ").strip()
    
#     try:
#         result_files = readlistFiles(dir_path, keyword)
#         print("\n✅ Matched Files:")
#         for f in result_files:
#             print(" -", f)
#     except Exception as e:
#         print(e)

