import struct
import numpy as np
import math

# Function to read and process the .int file
def open_int_file(file_path):
    with open(file_path, "rb") as file:
        binary_data = file.read()

    # Unpack the binary data as signed 4-byte integers
    unpacked_data = struct.unpack("<" + "i" * (len(binary_data) // 4), binary_data)

    # Convert the unpacked data to a Pillow image
    image_array = np.array(unpacked_data, dtype=np.int32)

    # Infer dimensions (assuming the image is square as a fallback)
    total_pixels = len(image_array)
    side_length = int(math.sqrt(total_pixels))
    if side_length * side_length != total_pixels:
        print(
            "Warning: The data does not form a perfect square. Adjust dimensions manually if needed."
        )
        side_length = total_pixels

    # Reshape the array to match the inferred dimensions
    image_array = image_array.reshape((side_length, side_length))

    # Normalize the data to 8-bit range (0-255) for visualization
    image_array = 255 - (
        (image_array - image_array.min())
        / (image_array.max() - image_array.min())
        * 255
    ).astype(np.uint8)

    return image_array