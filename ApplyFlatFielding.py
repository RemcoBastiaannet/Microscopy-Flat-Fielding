import tkinter as tk
from tkinter import filedialog
import numpy as np
import czifile
import os

def main():
    # Create a GUI to select the flat field effect numpy array
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    flat_field_path = filedialog.askopenfilename(
        title="Select the Flat Field Effect Numpy Array",
        filetypes=[("Numpy Files", "*.npy")]
    )
    if not flat_field_path:
        print("No flat field effect file selected. Exiting.")
        return

    # Load the flat field effect array
    flat_field_effect = np.load(flat_field_path)

    # Create a GUI to select the CZI file to apply the flat field effect
    czi_file_path = filedialog.askopenfilename(
        title="Select the CZI File to Apply Flat Fielding",
        filetypes=[("CZI Files", "*.czi")]
    )
    if not czi_file_path:
        print("No CZI file selected. Exiting.")
        return

    # Open the CZI file
    with czifile.CziFile(czi_file_path) as czi:
        # Read the image data
        image_data = czi.asarray()

    # Apply the flat field effect to each tile
    with np.errstate(divide='ignore', invalid='ignore'):
        corrected_data = np.true_divide(image_data, flat_field_effect)
        corrected_data[~np.isfinite(corrected_data)] = 0  # Replace NaNs and INFs with 0

    # Generate the output file path
    base, ext = os.path.splitext(czi_file_path)
    output_path = f"{base}_flatfield_corrected{ext}"

    # Save the corrected data back to a new CZI file
    with czifile.CziFile(output_path, mode='w') as czi_out:
        czi_out.write(corrected_data)

    print(f"Flat field correction applied and saved to: {output_path}")

if __name__ == "__main__":
    main()