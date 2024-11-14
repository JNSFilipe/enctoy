import torch
import deflate
import numpy as np
from scipy.fftpack import idct
from typing import Tuple, List
from enctoy.commun import BLOCK_SIZE, ZIGZAG_ORDER,get_quantization_from_quality

# Inverse zigzag scanning
def zigzagged_to_block(zigzagged: torch.Tensor) -> torch.Tensor:
    """Convert zigzagged scan back to block"""
    block = torch.zeros((8, 8), dtype=torch.float32)
    for i, idx in enumerate(zigzagged):
        x, y = ZIGZAG_ORDER[i]
        block[x, y] = idx
    return block

# Dequantization
def dequantize(block: torch.Tensor, quantization_table: torch.Tensor) -> torch.Tensor:
    """Dequantize the block"""
    return block * quantization_table

# Inverse DCT (simplified)
def idct2(block: torch.Tensor) -> torch.Tensor:
    """Perform inverse 2D DCT"""
    return torch.tensor(idct(idct(block.numpy(), axis=0, norm="ortho"), axis=1, norm="ortho"))+128

# Function to decode the encoded data (simplified)
def decode_image(img_path: str, height: int, width: int, quality: int) -> torch.Tensor:
    """Decode the image (simplified)"""
    # Deflate
    decoded_data = []
    with open(img_path, "rb") as f:
        decoded_data = f.read()
    decoded_data = deflate.deflate_decompress(decoded_data, width*height*8)
    decoded_data = np.frombuffer(decoded_data, dtype=np.int8)
    decoded_data = np.array(decoded_data).reshape(-1, BLOCK_SIZE**2).tolist()

    q_matrix = get_quantization_from_quality(quality)

    # Reconstruct the image from blocks (inverse zigzag, dequantize, and inverse DCT)
    blocks = []
    idx = 0
    for i in range(0, height, 8):
        row_blocks = []
        for j in range(0, width, 8):
            zigzagged = decoded_data[idx]
            block = zigzagged_to_block(zigzagged)
            dequantized_block = dequantize(block, q_matrix)
            idct_block = idct2(dequantized_block)
            row_blocks.append(idct_block)
            idx += 1
        blocks.append(torch.round(torch.cat(row_blocks, dim=1)))

    return torch.cat(blocks, dim=0).type(torch.uint8)
