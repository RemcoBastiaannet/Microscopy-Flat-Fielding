import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from czifile import CziFile

def select_folder():
    """Open a GUI to select a folder containing CZI files."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title="Select Folder Containing CZI Files")
    return folder_path

def process_czi_files(folder_path):
    """Process all CZI files in the selected folder."""
    if not folder_path:
        print("No folder selected.")
        return

    czi_files = [f for f in os.listdir(folder_path) if f.endswith('.czi')]
    if not czi_files:
        print("No CZI files found in the selected folder.")
        return

    print(f"Found {len(czi_files)} CZI files. Processing...")

    # Initialize an array to store the sum of all tiles
    flat_field_sum = None
    tile_count = 0

    for czi_file in czi_files:
        file_path = os.path.join(folder_path, czi_file)
        with CziFile(file_path) as czi:
            # Extract image data (assuming a specific structure, adjust as needed)
            image_data = czi.asarray()
            
            # Iterate over tiles/scenes and add them to the sum
            for tile in image_data:
                if flat_field_sum is None:
                    flat_field_sum = np.zeros_like(tile, dtype=np.float64)
                flat_field_sum += tile
                tile_count += 1

    # Compute the average flat field
    if tile_count > 0:
        flat_field_average = flat_field_sum / tile_count
        print("Flat field average computed.")
        # Ask the user where to save the flat field average
        save_path = filedialog.asksaveasfilename(
            title="Save Flat Field Average",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        if save_path:
            # Save the flat field average as a NumPy file
            np.save(save_path, flat_field_average)
            print(f"Flat field average saved to {save_path}.")
        else:
            print("Save operation canceled.")
    else:
        print("No tiles processed.")

def main():
    """Main function to run the script."""
    folder_path = select_folder()
    process_czi_files(folder_path)

if __name__ == "__main__":
    main()