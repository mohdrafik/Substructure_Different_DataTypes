import os
import json
import scipy.io as sio
import numpy as np
from pathlib import Path

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
    def save_masked_Unmasked_into_npy_mat(save_dir, base_name,
                            Masked_data, masked_coords,
                            filtered_data, unmasked_coords, mask):
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
