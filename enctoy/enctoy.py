import torch
import scipy as sp
import numpy as np
import constriction
from PIL import Image
from typing import Tuple, List

# Function to perform Discrete Cosine Transform (DCT)
def dct2(block: torch.Tensor) -> torch.Tensor:
    """Perform 2D DCT on a block"""
    # return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(block)))
    return torch.tensor(sp.fft.dctn(block))

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

# Quantization function
def quantize(block: torch.Tensor, quantization_table: torch.Tensor) -> torch.Tensor:
    """Quantize the DCT coefficients"""
    return torch.round(block / quantization_table)

# Zigzag scanning order for JPEG (simplified)
def zigzag_scan(block: torch.Tensor) -> torch.Tensor:
    """Zigzag scan the 8x8 block"""
    zigzag_order = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4),
        (2, 3), (3, 2), (4, 1), (5, 0), (5, 1), (4, 2), (3, 3), (2, 4),
        (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2),
        (6, 1), (7, 0)
    ]
    flattened_block = torch.flatten(block)
    return torch.tensor([flattened_block[x[0] * 8 + x[1]] for x in zigzag_order])

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
def encode_image(image: torch.Tensor) -> List[int]:
    """Encode the image using a simplified JPEG encoder"""
    blocks = split_into_blocks(image)
    encoded_data = []

    # Process each block in parallel using torch
    for block in blocks:
        # Apply DCT (2D)
        dct_block = dct2(block)

        # Quantize the DCT coefficients
        quantized_block = quantize(dct_block, JPEG_QUANTIZATION_TABLE)

        # Apply zigzag scan on the quantized block
        zigzagged = zigzag_scan(quantized_block)

        # Append the zigzagged coefficients for entropy coding
        encoded_data.append(zigzagged.numpy().astype(int))

    # Now we use constrictor for entropy coding
    # TODO: This needs to be its own function
    encoded_data = np.array([d for d in encoded_data])
    model = constriction.stream.model.QuantizedGaussian(encoded_data.min(), encoded_data.max(), encoded_data.mean(), encoded_data.std())
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(encoded_data.astype(np.int32), model)
    entropy_encoded_data = coder.get_compressed()
    # entropy_encoded_data = constrictor.encode(encoded_data)

    return entropy_encoded_data

# Function to decode the encoded data (simplified)
def decode_image(encoded_data: List[int], height: int, width: int) -> torch.Tensor:
    """Decode the image (simplified)"""
    # Decode using constrictor
    decoded_data = constrictor.decode(encoded_data)

    # Reconstruct the image from blocks (inverse zigzag, dequantize, and inverse DCT)
    blocks = []
    idx = 0
    for i in range(0, height, 8):
        row_blocks = []
        for j in range(0, width, 8):
            zigzagged = decoded_data[idx]
            block = zigzagged_to_block(zigzagged)
            dequantized_block = dequantize(block, JPEG_QUANTIZATION_TABLE)
            idct_block = idct2(dequantized_block)
            row_blocks.append(idct_block)
            idx += 1
        blocks.append(torch.cat(row_blocks, dim=1))

    return torch.cat(blocks, dim=0)

# Inverse zigzag scanning
def zigzagged_to_block(zigzagged: torch.Tensor) -> torch.Tensor:
    """Convert zigzagged scan back to block"""
    block = torch.zeros((8, 8), dtype=torch.float32)
    zigzag_order = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4),
        (2, 3), (3, 2), (4, 1), (5, 0), (5, 1), (4, 2), (3, 3), (2, 4),
        (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2),
        (6, 1), (7, 0)
    ]
    for i, idx in enumerate(zigzagged):
        x, y = zigzag_order[i]
        block[x, y] = idx
    return block

# Dequantization
def dequantize(block: torch.Tensor, quantization_table: torch.Tensor) -> torch.Tensor:
    """Dequantize the block"""
    return block * quantization_table

# Inverse DCT (simplified)
def idct2(block: torch.Tensor) -> torch.Tensor:
    """Perform inverse 2D DCT"""
    # return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(block)))
    return torch.tensor(sp.fft.idctn(block))

# Function to load and process an image using PIL
def load_image(image_path: str) -> torch.Tensor:
    """Load an image from a file, convert to grayscale, and convert to tensor"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale (L mode)
    img = np.array(img)  # Convert image to numpy array
    img_tensor = torch.tensor(img, dtype=torch.float32)  # Convert to tensor
    return img_tensor

# Example usage:
if __name__ == "__main__":
    image = load_image("./assets/Lena_512.png")

    # Encode the image
    encoded_image = encode_image(image)

    # Decode the image
    decoded_image = decode_image(encoded_image, image.shape[0], image.shape[1])

    # Print the original and decoded image to verify
    print("Original Image:")
    print(image)
    print("\nDecoded Image:")
    print(decoded_image)

    # Encode the image
    encoded_image = encode_image(image)

    # Decode the image
    decoded_image = decode_image(encoded_image, 64, 64)

    # Print the original and decoded image to verify
    print("Original Image:")
    print(image)
    print("\nDecoded Image:")
    print(decoded_image)
