import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from aicspylibczi import CziFile
from tifffile import imread, imwrite
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize

def main():
    # Select flat field OME TIFF file
    root = tk.Tk()
    root.withdraw()
    flat_field_path = filedialog.askopenfilename(
        title="Select Flat Field Correction (OME TIFF)",
        filetypes=[("OME TIFF Files", "*.ome.tiff"), ("All Files", "*.*")]
    )
    if not flat_field_path:
        print("No flat field selected. Exiting.")
        return

    flat_field = imread(flat_field_path)  # Load OME TIFF
    flat_field = np.squeeze(flat_field)  # Remove singleton dimensions [T=1, Z=1, C, Y, X] -> [C, Y, X]

    # Select CZI file
    czi_file_path = filedialog.askopenfilename(
        title="Select CZI File to Apply Correction",
        filetypes=[("CZI Files", "*.czi")]
    )
    if not czi_file_path:
        print("No CZI file selected. Exiting.")
        return

    czi = CziFile(czi_file_path)
    dim_map = dict(zip(czi.dims, czi.size))
    channels = dim_map.get('C', 1)

    if channels != flat_field.shape[0]:
        print("Mismatch in number of channels between flat field and CZI file. Exiting.")
        return

    tile_bounding_boxes = czi.get_all_tile_bounding_boxes()

    # Find minimum x and y to shift tiles to positive coordinates
    min_x = min(bbox.x for bbox in tile_bounding_boxes.values())
    min_y = min(bbox.y for bbox in tile_bounding_boxes.values())

    # Determine full stitched image size
    full_x = max((bbox.x - min_x) + bbox.w for bbox in tile_bounding_boxes.values())
    full_y = max((bbox.y - min_y) + bbox.h for bbox in tile_bounding_boxes.values())

    stitched_image = np.zeros((channels, full_y, full_x), dtype=np.float32)
    weight_map = np.zeros((channels, full_y, full_x), dtype=np.float32)

    with tqdm(total=len(tile_bounding_boxes), desc="Applying Flatfield", unit="tile") as pbar:
        for tile_info, bbox in tile_bounding_boxes.items():
            dims = tile_info.dimension_coordinates
            x_start = bbox.x - min_x
            x_end = bbox.x - min_x + bbox.w
            y_start = bbox.y - min_y
            y_end = bbox.y - min_y + bbox.h

            for c in range(flat_field.shape[0]):  # Loop over channels
                dims_with_c = dims.copy()
                dims_with_c['C'] = c

                tile_image, _ = czi.read_image(**dims_with_c)
                tile_data = np.squeeze(tile_image)

                corrected_tile = np.true_divide(tile_data, flat_field[c])
                corrected_tile[~np.isfinite(corrected_tile)] = 0

                # Add corrected tile into stitched image and update weight map
                stitched_image[c, y_start:y_end, x_start:x_end] += corrected_tile
                weight_map[c, y_start:y_end, x_start:x_end] += (corrected_tile != 0).astype(np.float32)

            pbar.update(1)

    # Normalize stitched image by weight map to blend overlaps
    with np.errstate(divide='ignore', invalid='ignore'):
        stitched_image = np.true_divide(stitched_image, weight_map)
        stitched_image[~np.isfinite(stitched_image)] = 0

    # Rescale to uint16 to reduce file size
    max_val = np.percentile(stitched_image, 99.9)
    stitched_image = np.clip(stitched_image / max_val, 0, 1) * 65535
    stitched_image = stitched_image.astype(np.uint16)
    stitched_image = stitched_image[np.newaxis, np.newaxis]  # [T=1, Z=1, C, Y, X]

    # Prepare output path
    output_base, _ = os.path.splitext(czi_file_path)
    output_path = f"{output_base}_flatfield_corrected.ome.tiff"

    imwrite(
        output_path,
        stitched_image,
        metadata={'axes': 'TZCYX'},
        compression='zlib',
        bigtiff=True,
        ome=True
    )

    print(f"Flat field correction applied and saved to {output_path}")

    # Generate RGB JPEG preview (downsampled)
    stitched_rgb = stitched_image[0, 0].astype(np.float32) / 65535  # [C, Y, X]
    channels, height, width = stitched_rgb.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    for i in range(min(3, channels)):
        rgb[:, :, i] = stitched_rgb[i]

    rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
    rgb_small = resize(rgb, (height // 4, width // 4), anti_aliasing=True)
    rgb_uint8 = (rgb_small * 255).astype(np.uint8)
    preview_path = f"{output_base}_preview.jpg"
    Image.fromarray(rgb_uint8).save(preview_path)
    print(f"Preview image saved to {preview_path}")

if __name__ == "__main__":
    main()