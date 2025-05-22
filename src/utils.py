import math
import struct

import numpy as np
from PyQt5.QtGui import QImage


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

def ndarray_to_qimage(arr):
    # Scale floats to 0-255:
    arr = np.ascontiguousarray(arr.T)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = 255 * (arr - arr.min()) / (np.ptp(arr) if np.ptp(arr) > 0 else 1)
        arr = arr.astype(np.uint8)
    h, w = arr.shape
    return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()

def non_max_suppression(coords, px_threshold = 20):
    if not coords:
        return []
    
    filtered = [coords[0]]

    for coord in coords[1:]:
        too_close = False
        for other in filtered:
            distance = np.linalg.norm(np.array(coord) - np.array(other))
            if distance < px_threshold:
                too_close = True
                break
        if not too_close:
            filtered.append(coord)
    return filtered


def unrotate_point_with_reshape(x_rot, y_rot, angle_deg, orig_shape, rotated_shape):
    angle_rad = -np.deg2rad(angle_deg)  # inverse angle
    H_orig, W_orig = orig_shape
    H_rot, W_rot = rotated_shape

    cx_orig, cy_orig = W_orig / 2, H_orig / 2  # center original
    cx_rot, cy_rot = W_rot / 2, H_rot / 2     # center rotated

    # Shift rotated point coordinates to rotation center
    x_shifted = x_rot - cx_rot
    y_shifted = y_rot - cy_rot

    # Inverse rotate
    x_orig_shifted = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
    y_orig_shifted = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)

    # Shift back to original image coordinate system
    x_orig = x_orig_shifted + cx_orig
    y_orig = y_orig_shifted + cy_orig

    return x_orig, y_orig


def play_gdr_song():
    import webbrowser
    from random import choice

    # open youtube link
    songs = [
        "https://youtu.be/la9uHQdFv2U?si=TWJ5VQxdKps9RpCk",
        "https://youtu.be/2flpdqGfAsw?si=FpXoJrBsqAw6aufH",
        "https://youtu.be/61B2oN5tV3M?si=RSXaXFUhfC0Sp4BG",
        "https://youtu.be/oopGFXItVyg?si=6X3RyKh9cTBvTDV5",
        "https://youtu.be/3_o9HkYCtQs?si=1JZP5VSn9b2eyox_",
    ]
    webbrowser.open(choice(songs))
