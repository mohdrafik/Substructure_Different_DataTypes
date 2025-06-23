import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import ScalarFormatter
from pathlib import Path


class HistogramAnalyzer:
    
    def __init__(self,base_dir, bins = 'fd'):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results" / "hist_peakDetect_plot"
        self.data_dir = self.base_dir/"data"/"raw_npyData"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith('.mat') or f.endswith('.npy')]
        print(f"Number of data files found: {len(self.files)}")
    
    def load_data(self, filename):
        filepath = self.data_dir / filename
        if filename.endswith('.mat'):
            datastruct = sio.loadmat(filepath)
            choose_keyVal = int(input(f"File {filename}: Enter 1 for 'elem_double' or 0 for 'mat_temp': "))
            if choose_keyVal == 1:
                data = datastruct['elem_double']
            else:
                data = datastruct['mat_temp']
        else:
            data = np.load(filepath)
        data = data.flatten()
        return data
    
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
    
  
    def fit_gaussian_and_plot_peak(self, data, filename, nbins=800, fit_gaussian = False):
        counts, bin_edges = np.histogram(data, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Fit a Gaussian
        
        # Plot
        plt.figure(figsize=(6,5))

        if fit_gaussian is True:
            plt.hist(data, bins=nbins, density=True, alpha=0.6, color='g', edgecolor='black')
            mu, std = norm.fit(data)
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



        plt.hist(data, bins=nbins,alpha=0.6, color='g', edgecolor='green')
        plt.title(f"histogram {filename}")
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.tight_layout() 
        save_path = self.results_dir / f"{filename}_histogram.png"
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

        print(f"normal histogram : {save_path}")


     
    def plot_Peak_and_histogram(self, filename, peaks_except_Highest=None, bins=None, singleColumnPlot = None):
        """
        Plots histogram with peak detection and saves the plot.
        Uses the instance's load_data method to load data.
        """
        data = self.load_data(filename)
        data = data.flatten()
        if filename[-6:-4] =='8h':
            data = data[data!=0]
        else:
            data = data[data > 0]

        nbins = bins if bins is not None else 1000
        counts, edges = np.histogram(data, bins=nbins)

        peakCounts = np.max(counts)
        ind_peakCounts = np.where(counts == peakCounts)[0]

        peak_left_edge = edges[ind_peakCounts]
        peak_right_edge = edges[ind_peakCounts + 1]

        # Calculate equivalent count limit for y-axis
        TotalValue_data = data.size
        peak_val_Percentage = (peakCounts / TotalValue_data) * 100
        remainingVal_percentage = 100 - peak_val_Percentage
        if peaks_except_Highest is None or peaks_except_Highest == 0:
            shared_averageValCount_percentage = 0
        else:
            shared_averageValCount_percentage = remainingVal_percentage / peaks_except_Highest

        final_comparableCountLimit = peakCounts * shared_averageValCount_percentage / 100 if peaks_except_Highest else peakCounts

        
        if singleColumnPlot :       
            # best for double column plot.
            # Ideal for single-column research paper
            # plt.figure(figsize=(3.6, 2.5))  # Ideal for single-column research paper
            fig = plt.figure(figsize=(3.6, 2.5), facecolor='white')  # White background
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')  # ensures white background inside plot area


            n, bins_hist, patches = ax.hist(data, bins=nbins, edgecolor='magenta', alpha=0.8, color='magenta', label='Histogram')
            ax.set_xlabel('Values', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.set_title(f'{filename[:-4]}', fontsize=8, fontweight='bold')
            ax.grid(True, linestyle='--', linewidth=0.2, alpha=0.8, color='gray')  # light gray grid color

            
            # for bold ticklabels: 
            # for tick in ax.xaxis.get_major_ticks():
            #     tick.label.set_fontsize(7)
            #     tick.label.set_fontweight('bold')

            # for tick in ax.yaxis.get_major_ticks():
            #     tick.label.set_fontsize(7)
            #     tick.label.set_fontweight('bold')

            # this is valid with plt.
            # plt.xlabel(fontsize=7, fontweight='bold')
            # plt.ylabel(fontsize=7, fontweight='bold')

            # Set tick parameters for paper style
            ax.tick_params(axis='both', labelsize = 7)
            ax.set_ylim(0, final_comparableCountLimit)

            # ax = plt.gca()
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # scientific notation
            # Add border/spine (BOX)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.8)

            # Plot vertical lines and annotations at peak edges
            for left, right in zip(peak_left_edge, peak_right_edge):
                if filename[-6:-4] in ['8h', '4h']:
                # if filename[-6:-4] == '8h' or filename[-6:-4] == '4h':
                    y_position_text = final_comparableCountLimit - 1.2E4
                    xtext_position = ((left+right+0.026)/2)
                else:
                    xtext_position = ((left+right+0.018)/2)
                    y_position_text = final_comparableCountLimit - 190
              
                ax.axvline(left, color='red', linestyle='-', linewidth=1.2, label=f'Peak Left Edge: {left:.6f}')
                ax.axvline(right, color='darkred', linestyle='-', linewidth=1.2, label=f'Peak Right Edge: {right:.6f}')

                ax.annotate(f'Count: {peakCounts}', xy=(xtext_position, y_position_text), xytext=(0, 10),
                            textcoords='offset points', ha='center', color='black', fontsize=5, fontweight='normal',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            # Clean up legend
            # ax = plt.gca()
            # handles, labels = ax.gca().get_legend_handles_labels()
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=5, loc='best', frameon=True)
            # ax.legend(by_label.values(), by_label.keys(), fontsize=3)
           



        else:
            # best for A4 size plot in full width.
            plt.figure(figsize=(11.7, 8.3))
            n, bins_hist, patches = plt.hist(data, bins=nbins, edgecolor='black', alpha=0.4, color='skyblue', label='Histogram')
            plt.xlabel('Values', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.title(f'Histogram: {filename[:-4]}', fontsize=16, fontweight='bold')
            plt.grid(True, linestyle='--', linewidth=1, alpha=0.5)

            # Set tick parameters for research paper style
            plt.xticks(fontsize=14,fontweight='bold')
            plt.yticks(fontsize=14,fontweight='bold')
            plt.ylim(0, final_comparableCountLimit)
            # Format y-axis in scientific notation
        


            # Plot vertical lines at peak edges
            for left, right in zip(peak_left_edge, peak_right_edge):

                if filename[-6:-4] == '8h' or filename[-6:-4] == '4h':
                # if filename[-6:-4] in ['8h', '4h']:
                    y_position_text = final_comparableCountLimit - 0.5E4
                    xtext_position = ((left+right+0.01)/2)
                else:
                    xtext_position = ((left+right+0.009)/2)
                    y_position_text = final_comparableCountLimit - 140
            
            
                plt.axvline(left, color='darkred', linestyle='-', linewidth=2, label=f'Peak Left Edge: {left:.6f}')
                plt.axvline(right, color='darkblue', linestyle='-', linewidth=2, label=f'Peak Right Edge: {right:.6f}')

                plt.annotate(f'Count: {peakCounts}', xy = (xtext_position, y_position_text), xytext=(0, 10),
                    textcoords='offset points', ha='center', color='black', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))

            ax =plt.gca()
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # force scientific notation

            plt.legend(by_label.values(), by_label.keys(), fontsize=12)
            plt.tight_layout()

        
        plt.tight_layout()
        save_dir = self.results_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        clean_filename = filename.replace('.mat', '').replace('.npy', '')
        # save_path = save_dir / f"{clean_filename}_histogram_peaks.eps"
        save_path = save_dir / f"{clean_filename}_histogram_peaks.png"
        plt.savefig(save_path, dpi=600, bbox_inches = 'tight', facecolor='white')
        # plt.savefig(save_path, dpi=600)
        print(f"Plot saved to: {save_path}")

        plt.close()


      

    # def process_all_files(self,fit_gaussian = False):
    #     for idx, filename in enumerate(self.files, 1):
    #         print(f"Processing file {idx}: {filename}")
    #         data = self.load_data(filename)
    #         clean_filename = filename.replace('.mat', '').replace('.npy', '')
    #         self.plot_histogram_with_envelope(data, clean_filename)

    #         self.fit_gaussian_and_plot_peak(data, clean_filename)

if __name__ == "__main__":
    base_dir = input("Enter the base_dir to your .npy or .mat files: ")

    BASE_DIR = base_dir
    # from hist_seginoneplot import HistogramAnalyzer
    hist = HistogramAnalyzer(BASE_DIR, bins = None)

    for file in hist.files:
        print(file)
        hist.load_data(file)
        if file[-6:-4] == '4h':
            hist.plot_Peak_and_histogram(filename=file, peaks_except_Highest = 50, bins = 900, singleColumnPlot=True)
        elif file[-6:-4] =='8h':
            hist.plot_Peak_and_histogram(filename=file, peaks_except_Highest = 1, bins = 900,singleColumnPlot=True)
        else:
            hist.plot_Peak_and_histogram(filename=file, peaks_except_Highest = 100, bins = 900, singleColumnPlot= True)

        # if file[-6:-4] in ['4h','8h']:
        #     hist.plot_Peak_and_histogram(filename=file, peaks_except_Highest = 1, bins = 900)
    print(f"All files processed succesfully -----------> ")


#     def plot_Peak_histogram(data, filename = None, peaks_except_Highest = 4, binning_method = None):
#         data = data.flatten()
#         data = data[data > 0]

#         # counts, edges, _ = plt.hist(data, bin = binning_method)
#         counts, edges = np.histogram(data, bins = binning_method)
#         peakCount = counts.size
#         ind_peakCount = np.where(counts == peakCount)[0]  # without this [0] index, it is tuple.
#         peak_left_edge = edges[ind_peakCount]
#         peak_right_edge = edges[ind_peakCount + 1]

#         def calculate_equivalent_Counts_Peak(data, peakCount, peaks_except_Highest):
#             TotalValue_data = data.size
#             peak_val_Percentage = (peakCount/TotalValue_data)*100
#             remainingVal_percentage = 100 - peak_val_Percentage
#             # let'say number of other peaks except the highest peak  = 5 then share of each peak
#             # shared average counts to each peak is :  remainingVal_percentage / 5 (or take 5-2)
#             shared_averageValCount_percentage = remainingVal_percentage/peaks_except_Highest
#             final_comparableCountLimit = peakCount*shared_averageValCount_percentage/100 # utilize it setting the ylimit.

#             return final_comparableCountLimit
        
#         ylimit = calculate_equivalent_Counts_Peak(data=data,peakCount=peakCount,peaks_except_Highest=peaks_except_Highest)


#         # Plot using bar
#         plt.figure(figsize=(10, 5))
#         # plt.bar(edges[:-1], counts, width=np.diff(edges), align='edge', edgecolor='black', color='skyblue')
#         plt.hist(data,bins=binning_method)
#         plt.xlabel("Value")
#         plt.ylabel("Frequency")
#         plt.title("Histogram from counts and edges")
#         plt.grid(True, linestyle='--', alpha=0.3)
#         plt.tight_layout()
#         plt.ylim(0,ylimit)
#         plt.show()
       