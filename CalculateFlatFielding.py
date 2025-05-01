import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from aicspylibczi import CziFile
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from tifffile import imwrite


def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder Containing CZI Files")
    return folder_path

def process_czi_files(folder_path):
    if not folder_path:
        print("No folder selected.")
        return

    czi_files = []
    for root_dir, _, files in os.walk(folder_path):
        czi_files.extend([os.path.join(root_dir, f) for f in files if f.endswith('.czi')])

    if not czi_files:
        print("No CZI files found.")
        return

    print(f"Found {len(czi_files)} CZI files. Processing...")

    flat_field_sum = None
    tile_count = None

    for file_path in czi_files:
        print(f"Processing {file_path}")
        czi = CziFile(file_path)

        if not czi.is_mosaic():
            print(f"Warning: {file_path} does not appear to be a mosaic scan. Skipping.")
            continue

        dim_map = dict(zip(czi.dims, czi.size))
        channels = dim_map.get('C', 1)

        tile_bounding_boxes = czi.get_all_tile_bounding_boxes()

        total_tiles = len(tile_bounding_boxes)

        with tqdm(total=total_tiles, desc="Processing Tiles", unit="tile") as pbar:
            for tile_info, bbox in tile_bounding_boxes.items():
                dims = tile_info.dimension_coordinates
                x_start = bbox.x
                y_start = bbox.y
                width = bbox.w
                height = bbox.h

                tile_image, _ = czi.read_image(**dims) #is this a new iamge every time?
                
                tile_data = np.squeeze(tile_image)

                channel_idx = dims.get('C', 0)

                if flat_field_sum is None:
                    flat_field_sum = np.zeros((channels, tile_data.shape[-2], tile_data.shape[-1]), dtype=np.float64)
                    tile_count = np.zeros(channels, dtype=np.int64)

                norm_factor = np.nansum(tile_data)
                tile_data = tile_data / norm_factor if norm_factor != 0 else tile_data

                flat_field_sum[channel_idx] += tile_data
                tile_count[channel_idx] += 1

                pbar.update(1)

    if flat_field_sum is None:
        print("No tiles were processed.")
        return

    flat_field_average = np.zeros_like(flat_field_sum)
    for c in range(flat_field_sum.shape[0]):
        if tile_count[c] > 0:
            flat_field_average[c] = flat_field_sum[c] / tile_count[c]
            print(f"Flat field for channel {c} computed.")
        else:
            print(f"No tiles found for channel {c}.")

    # Apply low-pass filtering to retain only very low spatial frequencies
    sigma = 50  # Adjust this value to control the level of low-pass filtering
    for c in range(flat_field_average.shape[0]):
        flat_field_average[c] = gaussian_filter(flat_field_average[c], sigma=sigma)
        flat_field_average[c] = flat_field_average[c] / np.nansum(flat_field_average[c])  # Normalize to [0, 1]
        print(f"Low-pass filter applied to channel {c}.")

    # Save flat field averages as OME TIFF
    save_path = filedialog.asksaveasfilename(
        title="Save Flat Field Averages as OME TIFF",
        defaultextension=".ome.tiff",
        filetypes=[("OME TIFF files", "*.ome.tiff"), ("All files", "*.*")]
    )
    if save_path:
        flat_field_average = flat_field_average[np.newaxis, np.newaxis]  # [T=1, Z=1, C, Y, X]
        imwrite(
            save_path,
            flat_field_average.astype(np.float32),
            metadata={'axes': 'TZCYX'},
            compression='zlib',
            bigtiff=True,
            ome=True
        )
        print(f"Flat field averages saved to {save_path}.")
    else:
        print("Save canceled.")

    # Optional: Visualize flat field averages
    for c in range(flat_field_average.shape[0]):
        plt.figure()
        plt.imshow(np.squeeze(flat_field_average)[c], cmap='gray')
        plt.title(f'Flat Field Average - Channel {c}')
        plt.axis('off')
        plt.show()
    return

def main():
    folder_path = select_folder()
    process_czi_files(folder_path)

if __name__ == "__main__":
    main()