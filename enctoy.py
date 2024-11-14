import torch
import tqdm
import numpy as np
import pandas as pd
import plotly.express as px

from sewar.full_ref import msssim, psnr
from enctoy import load_image, save_jpg, encode_image, decode_image
from pathlib import Path

TESTS_ROOT = "./.test"
REF_JPG_IMGS = f"{TESTS_ROOT}/ref_jpg"
OUR_JPG_IMGS = f"{TESTS_ROOT}/our_jpg"

# Example usage:
if __name__ == "__main__":
    # Create Folders
    Path(REF_JPG_IMGS).mkdir(parents=True, exist_ok=True)
    Path(OUR_JPG_IMGS).mkdir(parents=True, exist_ok=True)

    # Load original image
    # image = load_image("./assets/Lena_512.png")
    image = load_image("./assets/Lena_512_grey.png")

    df = pd.DataFrame(columns=["type", "bpp", "psnr", "msssim"])
    decoded_image = []

    for Q in tqdm.tqdm(range(1, 90, 10)):
    # for Q in tqdm.tqdm(range(1, 40, 10)):
        # Generate Paths
        ref_path: str = f"{REF_JPG_IMGS}/{Q}.jpg"
        our_path: str = f"{OUR_JPG_IMGS}/{Q}.jpg"

        # Encode with PIL JPEG
        save_jpg(image, ref_path, quality=Q)
        ref: torch.Tensor = load_image(ref_path)
        df = df._append({
            "type": "ref",
            "bpp": Path(ref_path).stat().st_size * 8.0 / (image.shape[0] * image.shape[1]),
            "psnr": psnr(image.numpy().astype(np.uint8), ref.numpy().astype(np.uint8)),
            "msssim": msssim(image.numpy().astype(np.uint8), ref.numpy().astype(np.uint8)).real
        }, ignore_index=True)

        # Encode the image
        encode_image(image, Q, our_path)

        # Decode the image
        decoded_image = decode_image(our_path, image.shape[0], image.shape[1], Q)
        df = df._append({
            "type": "our",
            "bpp": Path(our_path).stat().st_size * 8.0 / (image.shape[0] * image.shape[1]),
            "psnr": psnr(image.numpy().astype(np.uint8), decoded_image.numpy().astype(np.uint8)),
            "msssim": msssim(image.numpy().astype(np.uint8), decoded_image.numpy().astype(np.uint8)).real
        }, ignore_index=True)

    print(df.head())
    refs = df[df.type=="ref"]
    ours = df[df.type=="our"]

    fig = px.line(df, x="bpp", y="psnr", color="type", markers=True)
    fig.show()

    fig = px.line(df, x="bpp", y="msssim", color="type", markers=True)
    fig.show()

    # # Print the original and decoded image to verify
    # print("Original Image:")
    # print(image)
    # print("\nDecoded Image:")
    # print(decoded_image)

    # fig = px.imshow(decoded_image)
    # fig.show()
