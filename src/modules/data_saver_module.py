import os
import json
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
            else:
                filtered_entry = entry
            filtered_data.append(filtered_entry)

        save_path = self.save_dir / self.filename
        with open(save_path, 'w') as f:
            json.dump(filtered_data, f, indent=4)
        print(f"✅ Saved metadata to: {save_path}")



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
