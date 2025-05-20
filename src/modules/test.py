import gc  #  garbage colloctor
import time
import os
import numpy as np
from pathlib import Path
from preprocessAll import DataPreprocessor
# from plot_dataModule import visualize_and_export_3d_mesh
from plot_dataModule import DataPlotter

# from gui_utils import show_progress_gui

from gui_utils import ProgressGUI


# if __name__ == "__main__":

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent

input_dir = PROJECT_PATH / "data" / "raw_npyData"
output_dir = PROJECT_PATH / "results" / "otsu_results"
image_plotpath = PROJECT_PATH / "results" / "otsu_plot"

os.makedirs(image_plotpath, exist_ok=True)

files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
print(f"\n Found {len(files)} .npy files {files} in: {input_dir}")

# window, update_progress = show_progress_gui(len(files))
gui = ProgressGUI(len(files))

# for file in files:
for idx, file in enumerate(files):
    # Update the progress bar and label
    # update_progress(idx + 1, file)
    # update_progress(idx + 1, file)
    if gui.is_cancelled():
        break

    gui.update(idx + 1, file)   

    npy_path = input_dir / file
    save_png_path = image_plotpath / f"{file[:-4]}fg.png"
    save_gif_path = image_plotpath / f"{file[:-4]}fg.gif"
    data = np.load(str(npy_path))
    try:
        print(f"\n Processing: {file}")

        res = DataPreprocessor(data).apply_otsu_segmentation(save_masks_to=None)

        threshold, fg_mask, bg_mask  = res['threshold'], res['fg_mask'], res['bg_mask']
        print(f"Otsu Threshold: {threshold:.7f}")

        DataPlotter.visualize_and_export_3d_mesh(
            fg_mask=fg_mask,
            data=data,
            smoothing=None,
            title=file[:-4],
            save_obj_path=None,
            save_png_path=None,  #save_png_path
            save_gif_path= None, #save_gif_path,
            rotate_and_capture=True,
            gif_frames=36           
        )

        print(f" Finished: {file}")
       
        # Save the threshold value to a text file
        threshold_file = output_dir / f"{file[:-4]}_ostu_threshold.txt"  
        with open(threshold_file, 'w') as f:
            f.write(f"Otsu Threshold: {threshold:.7f}\n")
            f.write(f"Foreground Mask Shape: {fg_mask.shape}\n")
            f.write(f"Background Mask Shape: {bg_mask.shape}\n")

        # Clear memory manually
        # del data, res, fg_mask, bg_mask
        # gc.collect()      
        del data, res, fg_mask, bg_mask
        # Optional: Force garbage collection    
        gc.collect()    

     
        time.sleep(2)  # Optional: Add a delay to observe the output    
        # input(" Press Enter to process the next file...")

    except Exception as e:

        print(f" Failed {file}: {e}")
        
gui.close()

print(f"\n Finished processing all files in: {input_dir} and \n saved results to: {output_dir} \n")
# Close the progress window
# window.destroy()