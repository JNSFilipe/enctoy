from sys import exception
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List

BLOCK_SIZE: int = 8

# Quantization table for JPEG (simplified example)
# In real JPEG, this table is more complex, but for simplicity we'll use a smaller table.
JPEG_QUANTIZATION_TABLE: torch.Tensor = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=torch.float32)

def get_quantization_from_quality(Q:int) -> torch.Tensor:
    # https://stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression

    if Q < 0 or Q > 100:
        raise Exception("Quality factor out of bounds, should be between 0 and 100")

    S:float = 5000/Q if Q<50 else 200 - 2*Q

    Ts: torch.Tensor = torch.floor((S*JPEG_QUANTIZATION_TABLE + 50) / 100.0)
    Ts[Ts<=0] = 1

    # TODO: This statement is so that weights fit in 8 bits, must fix this at some point
    if Q >= 80:
        Ts = torch.floor(Ts*1.4)
    ###

    return Ts

def zigzag_indices(n: int) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []

    for sum_idx in range(2 * n - 1):
        # Determine the starting and ending row and column for the current diagonal
        if sum_idx % 2 == 0:
            # Moving downwards from top-right to bottom-left
            start_row: int = min(sum_idx, n - 1)
            start_col: int = sum_idx - start_row
            while start_row >= 0 and start_col < n:
                result.append((start_row, start_col))
                start_row -= 1
                start_col += 1
        else:
            # Moving upwards from bottom-left to top-right
            start_col: int = min(sum_idx, n - 1)
            start_row: int = sum_idx - start_col
            while start_col >= 0 and start_row < n:
                result.append((start_row, start_col))
                start_row += 1
                start_col -= 1

    return result

ZIGZAG_ORDER: List[Tuple[int, int]] = zigzag_indices(BLOCK_SIZE)

# Function to load and process an image using PIL
def load_image(image_path: str) -> torch.Tensor:
    """Load an image from a file, convert to grayscale, and convert to tensor"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale (L mode)
    img = np.array(img)  # Convert image to numpy array
    img_tensor = torch.tensor(img, dtype=torch.float32)  # Convert to tensor
    return img_tensor

def save_jpg(image: torch.Tensor, path:str, quality:int=95):
    # TODO: Only works for greyscale
    img: Image.Image = Image.fromarray(image.numpy()).convert("L")
    if not (".jpg" in path or ".jpeg" in path):
        raise Exception("path must end in .jpeg or .jpg")
    img.save(path, quality=quality)

def save_png(image: torch.Tensor, path:str):
    # TODO: Only works for greyscale
    img: Image.Image = Image.fromarray(image.numpy()).convert("L")
    if ".png" not in path:
        raise Exception("path must end in .png")
    img.save(path)
