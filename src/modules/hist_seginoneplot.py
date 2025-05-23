import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

class HistogramAnalyzer:
    
    def __init__(self, path,bins = 'fd'):
        self.path = Path(path)
        self.results_dir = self.path / "results" / "histogramrawData"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.files = [f for f in os.listdir(self.path) if f.endswith('.mat') or f.endswith('.npy')]
        print(f"Number of data files found: {len(self.files)}")
    
    def load_data(self, filename):
        filepath = self.path / filename
        if filename.endswith('.mat'):
            datastruct = sio.loadmat(filepath)
            choose_keyVal = int(input(f"File {filename}: Enter 1 for 'elem_double' or 0 for 'mat_temp': "))
            if choose_keyVal == 1:
                data = datastruct['elem_double']
            else:
                data = datastruct['mat_temp']
        else:
            data = np.load(filepath)
        data = np.array(data).flatten()
        return data[data != 0]
    
    def plot_histogram_with_envelope(self, data, filename, nbins=800):
        plt.figure(figsize=(6,5))
        counts, edges, _ = plt.hist(data, bins=nbins, color='white', alpha=0.7, edgecolor='white', label='hist')
        plt.fill_between(edges[:-1], counts, color='None', edgecolor='blue', alpha=0.7, label='envelope')
        plt.title('Envelope of data distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        save_path = self.results_dir / f"{filename}_hist_envelope.png"
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"Saved histogram and envelope: {save_path}")
    
    def fit_gaussian_and_plot_peak(self, data, filename, nbins=800):
        counts, bin_edges = np.histogram(data, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Fit a Gaussian
        mu, std = norm.fit(data)
        
        # Plot
        plt.figure(figsize=(6,5))
        plt.hist(data, bins=nbins, density=True, alpha=0.6, color='g', edgecolor='black')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 1000)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        plt.title(f"Gaussian Fit: μ={mu:.4f}, σ={std:.4f}")
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.tight_layout()
        
        save_path = self.results_dir / f"{filename}_gaussian_peak.png"
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()
        print(f"Saved Gaussian peak fit: {save_path}")
        
        return mu, std

    def process_all_files(self):
        for idx, filename in enumerate(self.files, 1):
            print(f"Processing file {idx}: {filename}")
            data = self.load_data(filename)
            clean_filename = filename.replace('.mat', '').replace('.npy', '')
            self.plot_histogram_with_envelope(data, clean_filename)
            self.fit_gaussian_and_plot_peak(data, clean_filename)

if __name__ == "__main__":
    path = input("Enter the path to your .npy or .mat files: ")

    analyzer = HistogramAnalyzer(path)
    analyzer.process_all_files()



# def plot_histogram_auto(data,filename,count):
    
#     """
#     Create a histogram with automatic bin width and edges chosen from the data.

#     this is the complete code to load the all .mat data files automatically and 
#     plot histogram and envelope for each file and save the corresponding plot in the same directory.

#     just give the path for the directory accordingly.
    
#     """

#     print("count no. = ", count)
# #     if count == 1:
# #         pass
# #         plt.figure(figsize=(6,5))  
# #         nbins = 400
# #         plt.subplot(2,1,1)
# #         plt.hist(data,bins=nbins,color='blue',alpha=0.7, edgecolor ='magenta')
# #         plt.title('data distribution')
# #         plt.xlabel('value')
# # #         plt.xlim([1.3326,1.3371])  # this is for manual thresholding
# #         plt.xlim([1.3338,1.3342]) # 1.32951.33851.334N1  8 parts.
# #         plt.tight_layout() 
# #         plt.ylabel('Frequency')


# #         plt.subplot(2,1,2)
# #         counts,edges,_ = plt.hist(data, bins=nbins,color='white',alpha=0.7, edgecolor ='white',label ='hist')
# #         plt.fill_between(edges[:-1],counts,color='None',alpha=0.7, edgecolor ='blue',label ='envelope')
# #         plt.title('Envelope of data distribution')
# #         plt.xlabel('value')
# #         plt.xlim([1.3338,1.3342])  # for the kmeans auto thresholding 8 parts 
# # #         plt.xlim([1.3326,1.3371]) # this is for manual thresholding
# #         plt.ylabel('Frequency')
# #         # Save the figure
# #         filename1 = filename+'.png'
# #         print(filename1)
# #         plt.tight_layout()
# #         plt.savefig(filename1, dpi=400, bbox_inches='tight')
# #         plt.show()
    
# #     else:
#     plt.figure(figsize=(6,5))  
# #     #     binsw = (max(data) -min(data))/200
#     nbins = 800

#     plt.subplot(2,1,1)
#     plt.hist(data,bins=nbins,color='blue',alpha=0.7, edgecolor ='magenta')
#     plt.title('data distribution')
#     plt.xlabel('value')
#     plt.ylabel('Frequency')


#     plt.subplot(1,1,1)
#     # plt.plot()
#     counts,edges,_ = plt.hist(data, bins=nbins,color='white',alpha=0.7, edgecolor ='white',label ='hist')
#     plt.fill_between(edges[:-1],counts,color='None',alpha=0.7, edgecolor ='blue',label ='envelope')
#     plt.title('Envelope of data distribution')
#     plt.xlabel('value')
#     plt.ylabel('Frequency')
# #         plt.ylim([0,100])
# #     plt.legend()
# #     plt.xlim([1.31,1.48])
# # Identify and label extreme values
# #     mean_count = np.mean(counts)
# #     std_count = np.std(counts)
# #     extreme_values = []
# #     for i, count in enumerate(counts):
# #         if count > mean_count + 2 * std_count or count < mean_count - 2 * std_count:
# #             plt.text(bin_edges[i], 0, str(round(bin_edges[i], 2)), rotation=90, va='bottom', ha='center', color='red')
# #             extreme_values.append(bin_edges[i])

#     # Save the figure
#     from pathlib import path
#     # exiting_filepath = path.(__src__).resolve().parent
#     savepath = 
#     filename1 = filename+'.png'
#     os.makedirs(os.path.join('results\histogramrawData', exist_ok =True))

#     filename1 = os.path.join('results', filename1)
#     print(filename1)
#     plt.tight_layout()
#     plt.savefig(filename1, dpi=400, bbox_inches='tight')
#     plt.show()
#     # plt.close()
# # plot_histogram_auto(data, 'histogram.png')
    



# """
# datastruct ==> after loading using sio.loadmat(filename) --> get the datastruct as dictionary format as
# following and our actual data is associated with key 'mat_temp' .
# {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Thu Dec 14 13:37:14 2023',
#  '__version__': '1.0',
#  '__globals__': [],
#  'mat_temp': array([[[0., 0., 0., . .................. 

# """

# # path ='C:\\Users\\mrafik\\Desktop\\conference_data\\paper_data14_122023\\paper_data14_122023\\autothresholdingKmeans'
# path = input("enter the path of your MAT files:")
# print(f"The path is ----------> {path}")
# if not path: 
#     path = "C:\\Users\mrafik\\Desktop\\conference_data\\paper_dataMarch2024\\abstracttomograms\\manualthreshResult"

# import os
# import scipy.io as sio
# # for mat in os.path.join(path,.mat):
# allfiles = os.listdir(path)
# print("Number of files: = ",len(allfiles))
# matfiles = []
# count = 0

# for filename in allfiles:
#     # print(filename)
#     if filename.endswith('.mat') or filename.endswith('.npy'):
#         count =count+1
#         matfiles.append(filename)
#         print(filename)
#         if filename.endswith('.mat'):
#             datastruct = sio.loadmat(filename) # --> readmatfile(filename)
#     #         print(datastruct.items)  --> it will show the actuall data under the key 'mat_temp'
#             choose_keyVal = int(input("enter for threshold matlab segment with key elem_double 1 or for key mat_temp 0:"))
#             if choose_keyVal == 1:
#                 data = datastruct['elem_double'] 
#             if choose_keyVal == 0:
#                 data = datastruct['mat_temp']
#         else:
#             data = np.load(os.path.join(path,filename))

#         data = np.array(data)
#         data = data.flatten()
#         data = data[data!=0]
#         filename = filename[0:-4:1]+f'histseg{filename[-5:-4]}'
#         print("new file name after removing .mat = ", filename)
#         print("max data :",max(data), " ","min data", min(data))
#         print("figure completed and size type of data--> ",count,data.size,data.dtype)
#         if count !=1:
#             plot_histogram_auto(data, filename,count) # --> This function is called,again and again in for loop, which draw the histogram of each seperated data(.mat file) using the  k -means cluster.    

      