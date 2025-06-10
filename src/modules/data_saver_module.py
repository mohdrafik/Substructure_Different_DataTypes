import os
import json
import scipy.io as sio
import numpy as np
from pathlib import Path
import json
import pandas as pd

# Decorator to log method execution
# This decorator can be used to log the execution of methods in the DataSaver class.
from functools import wraps

def decoratorLog(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        print(f"\n ---------------------> /// Implementing method: {func.__name__} \\\ <------------------------------------------------------- \n")
        results = func(*args, **kwargs)
        print(f"\n ---------------------> /// Finished executing method: {func.__name__} \\\ <--------------------------------------------------\n")
        return results
    return wrapper

class DataSaver:
    """
    A flexible class to save metadata or results to JSON format with optional filtering of fields.

    Args:
        save_dir (str or Path): Directory where the JSON file should be saved.
        include_fields (list of str): List of fields to include in the saved JSON.
        filename (str): The name of the JSON file to create (default is 'output.json').

    Methods:
        save(data_list): Saves the list of dictionaries to a JSON file, filtered by include_fields.
    """

    def __init__(self, save_dir, include_fields=None, filename="output.json"):
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.include_fields = include_fields if include_fields else []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @decoratorLog
    def save(self, data_list):
        filtered_data = []
        for entry in data_list:
            if self.include_fields:
                filtered_entry = {k: v for k, v in entry.items() if k in self.include_fields}
                # {k: v for k, v in entry.items() if k in self.include_fields} this is equivalent to the below code.
                # filtered_entry = {}
                # for k, v in entry.items():
                #     if k in ["filename", "counts"]:
                #         filtered_entry[k] = v

            else:
                filtered_entry = entry # No filtering if include_fields is empty means save all fields like we can say from example  usage all fields from metadata_list are included.

            filtered_data.append(filtered_entry)

        save_path = self.save_dir / self.filename
        with open(save_path, 'w') as f:
            json.dump(filtered_data, f, indent=4)
        print(f" Saved metadata to: {save_path}")



    @staticmethod
    @decoratorLog
    def save_masked_Unmasked_into_npy_mat(save_dir, base_name,
                            Masked_data, masked_coords,
                            filtered_data, unmasked_coords, mask, bgDataonly):
        """
        Save the masked data and coordinates into .npy and .mat files.

        Parameters:
        -----------
        save_dir : str
            Directory where files will be saved.
        base_name : str
            Base name used for naming files.
        Masked_data : np.ndarray
            The masked array after masking values.
        masked_coords : np.ndarray
            Coordinates of masked values.
        filtered_data : np.ndarray
            Filtered non-zero data after masking with zero.
        unmasked_coords : np.ndarray
            Coordinates of values not masked.
        mask : np.ndarray
            Boolean mask indicating masked positions.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save individual components as .npy
        np.save(os.path.join(save_dir, f"{base_name}_masked_data.npy"), Masked_data)
        np.save(os.path.join(save_dir, f"{base_name}_masked_coords.npy"), masked_coords)
        np.save(os.path.join(save_dir, f"{base_name}_filtered_data.npy"), filtered_data)
        np.save(os.path.join(save_dir, f"{base_name}_unmasked_coords.npy"), unmasked_coords)
        np.save(os.path.join(save_dir, f"{base_name}_mask_bool.npy"), mask)
        np.save(os.path.join(save_dir, f"{base_name}_bgmaskvalues.npy"), bgDataonly)

        # Save .mat file with x,y,z,1/0 (assuming 3D coordinates)
        #  Create labels
        unmasked_labels = np.ones((unmasked_coords.shape[0], 1))
        masked_labels = np.zeros((masked_coords.shape[0], 1))

        # Combine with labels
        unmasked_labeled = np.hstack((unmasked_coords, unmasked_labels))
        masked_labeled = np.hstack((masked_coords, masked_labels))

        # Concatenate all rows: [x, y, z, label]
        all_coords_labeled = np.vstack((unmasked_labeled, masked_labeled))

        # Save as .mat
        save_path = os.path.join(save_dir, f"{base_name}_coords_mask_label.mat")
        sio.savemat(save_path, {'coords_mask_label': all_coords_labeled})

###########################################################################################################
    # This module provides a DataSaver class that can be used to save metadata or results to JSON format.
    @staticmethod
    @decoratorLog
    def save_background_foreground_stats(
        data,
        Masked_data,
        maskedValues_coordsOnly,
        filtered_Data_WithoutZero,
        UnMasked_coords,
        mask,
        bgDataonly,
        filenameWithoutExtension,
        save_dir
    ):
        """
        Calculate and save background/foreground statistics for a given dataset.

        Args:
            data (np.ndarray): Original data array.
            Masked_data (np.ndarray): Foreground-masked data.
            maskedValues_coordsOnly (np.ndarray): Coordinates of masked (background) values.
            filtered_Data_WithoutZero (np.ndarray): Foreground values after masking.
            UnMasked_coords (np.ndarray): Coordinates of unmasked (foreground) values.
            mask (np.ndarray): Boolean mask for background.
            bgDataonly (np.ndarray): Array with only background values.
            filenameWithoutExtension (str): Key for stats dictionary.
            save_dir (Path or str): Directory to save the JSON file.

        Returns:
            dict: The updated all_stats dictionary.
        """
        # Ensure mask is boolean
        mask_bool = mask.astype(bool)

        file_stats = {
            "data_shape_before_removing_zeros": (data.shape),
            "data_shape_after_removing_zeros": (np.array(data[data != 0]).shape),
            "total_points_nonzero": int(np.count_nonzero(data > 0)),
            "background_points": int(np.count_nonzero(mask_bool)),
            "background_percentage": float((np.count_nonzero(mask_bool) / data.size) * 100),
            "foreground_points": int(np.count_nonzero(Masked_data > 0)),
            "foreground_percentage": float((np.count_nonzero(Masked_data > 0) / data.size) * 100),
            "sum_bg_fg": int(np.count_nonzero(mask_bool) + np.count_nonzero(Masked_data > 0)),
            "total_elements": int(data.size),
            "maskedValues_coordsOnly_shape": list(maskedValues_coordsOnly.shape),
            "filtered_Data_WithoutZero_shape": list(filtered_Data_WithoutZero.shape),
            "UnMasked_coords_shape": list(UnMasked_coords.shape),
            "bgDataonly_nonzero_count": int(np.count_nonzero(bgDataonly))
        }

        summary = {filenameWithoutExtension: file_stats}
        json_save_path = os.path.join(str(save_dir), "background_foreground_stats.json")

        if os.path.exists(json_save_path):
            with open(json_save_path, "r") as f:
                all_stats = json.load(f)
        else:
            all_stats = {}

        all_stats.update(summary)

        with open(json_save_path, "w") as f:
            json.dump(all_stats, f, indent=4)

        print(f"Saved stats for {filenameWithoutExtension} to {json_save_path}")
        # Convert all_stats to a DataFrame for CSV export
        df = pd.DataFrame.from_dict(all_stats, orient='index')
        csv_save_path = os.path.join(str(save_dir), "background_foreground_stats.csv")
        df.to_csv(csv_save_path)
        print(f"Saved CSV summary to {csv_save_path}")
        
        return all_stats

# Example usage:
# all_stats = save_background_foreground_stats(
#     data, Masked_data, maskedValues_coordsOnly, filtered_Data_WithoutZero,
#     UnMasked_coords, mask, bgDataonly, filenameWithoutExtension, save_dir
# )    


# Example usage
if __name__ == "__main__":

    from data_saver_module import DataSaver  # Import the DataSaver class

    # Create saver with custom fields to include
    saver = DataSaver(save_dir="results/histogram_significantDigits/", include_fields=["filename", "x_axis_max"])

    # Example metadata list
    metadata_list = [
        {
            "filename": "sample1",
            "unique_values": [1, 2, 3],
            "counts": [100, 200, 150],
            "x_axis_max": 3,
            "significant_digit_data": [1, 1, 2, 3]
        }
    ]
    saver.save(metadata_list)
