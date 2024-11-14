import torch
import deflate
import numpy as np
from scipy.fftpack import dct
from typing import Tuple, List
from enctoy.commun import ZIGZAG_ORDER, get_quantization_from_quality

# Function to perform Discrete Cosine Transform (DCT)
def dct2(block: torch.Tensor) -> torch.Tensor:
    """Perform 2D DCT on a block"""
    return torch.tensor(dct(dct(block.numpy()-128, axis=0, norm="ortho"), axis=1, norm="ortho"))


# Quantization function
def quantize(block: torch.Tensor, quantization_table: torch.Tensor) -> torch.Tensor:
    """Quantize the DCT coefficients"""
    return torch.round(block / quantization_table)

# Zigzag scanning order for JPEG (simplified)
def zigzag_scan(block: torch.Tensor) -> torch.Tensor:
    """Zigzag scan the 8x8 block"""
    flattened_block = torch.flatten(block)
    return torch.tensor([flattened_block[x[0] * 8 + x[1]] for x in ZIGZAG_ORDER])

# Function to split the image into blocks
def split_into_blocks(image: torch.Tensor, block_size: int = 8) -> List[torch.Tensor]:
    """Split image into 8x8 blocks"""
    blocks = []
    height, width = image.shape
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks

# Function to encode the image using a simplified JPEG encoder
def encode_image(image: torch.Tensor, quality:int, save_path:str):
    """Encode the image using a simplified JPEG encoder"""
    blocks = split_into_blocks(image)
    encoded_data = []

    q_matrix = get_quantization_from_quality(quality)

    # Process each block in parallel using torch
    for block in blocks:
        # Apply DCT (2D)
        dct_block = dct2(block)

        # Quantize the DCT coefficients
        quantized_block = quantize(dct_block, q_matrix)

        # Apply zigzag scan on the quantized block
        zigzagged = zigzag_scan(quantized_block)
        # print(zigzagged)

        # Append the zigzagged coefficients for entropy coding
        encoded_data.append(zigzagged.numpy().astype(np.int8))

    # Now we use constrictor for entropy coding
    encoded_data = np.array(encoded_data).flatten().astype(np.int8)
    bitstream = deflate.deflate_compress(encoded_data, 12)
    with open(save_path, "wb") as f:
        f.write(bitstream)
